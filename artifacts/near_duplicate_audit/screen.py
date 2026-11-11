"""Screen PlantSeg RGB photos for cross-partition near duplicates.

Run from the project root with .venv/bin/python. The screen writes candidates
and contact sheets for manual review. It does not modify either manifest or
run a classifier. Only Pillow and NumPy are required.
"""

import argparse
import csv
import hashlib
import json
import os
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
DATA_ROOT = Path(os.environ.get("PLANTSEG_ROOT", PROJECT / "data/plantseg"))
MANIFEST = PROJECT / "artifacts/option2_manifest.json"
SPLITS = {"training": "train", "validation": "val", "test": "test"}
SCREEN_SIZE = 32
CORRELATION_LIMIT = 0.87
CROP_LIMIT = 0.86
PHASH_LIMIT = 14
DHASH_LIMIT = 10
DCT = np.cos(
    np.pi * np.arange(8)[:, None]
    * (np.arange(SCREEN_SIZE)[None, :] + 0.5) / SCREEN_SIZE
).astype(np.float32)
DCT[0] /= np.sqrt(2)


def load_rows():
    manifest = json.loads(MANIFEST.read_text())
    excluded = {row["image_path"] for row in manifest["audit_exclusions"]}
    hosts = {row["host"] for row in manifest["class_roles"]["A"]}
    rows = []
    with (DATA_ROOT / "Metadata.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["Plant"].strip().casefold() not in hosts:
                continue
            split = SPLITS[row["Split"].strip().casefold()]
            path = f"images/{split}/{row['Name'].strip()}"
            if path not in excluded:
                rows.append({
                    "image_path": path,
                    "split": split,
                    "class_id": int(row["Index"]),
                    "class_name": row["Disease"].strip(),
                    "url": row.get("URL", "").strip(),
                })
    if len(rows) != len({row["image_path"] for row in rows}):
        raise ValueError("Image paths are not unique after exact deduplication")
    return sorted(rows, key=lambda row: row["image_path"])


def crop_boxes(width, height):
    yield "full", (0, 0, width, height)
    for fraction, positions in (
        (0.9, [(x, y) for x in (0, 0.5, 1) for y in (0, 0.5, 1)]),
        (0.8, [(x, y) for x in (0, 0.5, 1) for y in (0, 0.5, 1)]),
        (0.65, [(x, y) for x in (0, 0.5, 1) for y in (0, 0.5, 1)]),
    ):
        crop_width = round(width * fraction)
        crop_height = round(height * fraction)
        for x, y in positions:
            left = round((width - crop_width) * x)
            top = round((height - crop_height) * y)
            yield f"{fraction:g}@{x:g},{y:g}", (
                left, top, left + crop_width, top + crop_height
            )


def features(row):
    with Image.open(DATA_ROOT / row["image_path"]) as image:
        image = ImageOps.exif_transpose(image).convert("L")
    # Crop on the EXIF-corrected image after shrinking large inputs once.
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    grids = []
    crop_names = []
    for name, box in crop_boxes(*image.size):
        thumbnail = image.crop(box).resize(
            (SCREEN_SIZE, SCREEN_SIZE), Image.Resampling.LANCZOS
        )
        grids.append(np.asarray(thumbnail, dtype=np.float32))
        crop_names.append(name)
    difference_grid = np.asarray(
        image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16
    )
    dhash = np.where(np.diff(difference_grid, axis=1).ravel() > 0, 1, -1)
    full = grids[0]
    phashes = []
    for orientation in (full, np.fliplr(full)):
        values = (DCT @ orientation @ DCT.T).ravel()[1:]
        phashes.append(np.where(values > np.median(values), 1, -1))
    grids = np.stack(grids)
    grids -= grids.mean(axis=(1, 2), keepdims=True)
    norms = np.linalg.norm(grids.reshape(len(grids), -1), axis=1)
    grids /= np.maximum(norms[:, None, None], 1)
    return grids, np.asarray(phashes, dtype=np.float32), dhash.astype(np.float32), crop_names


def dot_product(left, right):
    # Accelerate can set floating-point warning flags even for finite products.
    # Validate the actual matrix explicitly before using it.
    with np.errstate(all="ignore"):
        product = left @ right.T
    if not np.isfinite(product).all():
        raise ValueError("Non-finite image similarity")
    return product


def cross_candidates(rows, grids, phashes, dhashes, crop_names):
    splits = np.array([r["split"] for r in rows])
    normalized = grids.reshape(len(rows), len(crop_names), -1)
    full = np.ascontiguousarray(normalized[:, 0, :])
    flips = np.ascontiguousarray(grids[:, 0, :, ::-1]).reshape(full.shape)
    matches = defaultdict(dict)

    def record(i, j, **measurements):
        key = tuple(sorted((int(i), int(j))))
        for method, value in measurements.items():
            previous = matches[key].get(method)
            if previous is None or value < previous:
                matches[key][method] = value

    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        cross = splits[start:stop, None] != splits[None, :]
        corr = dot_product(full[start:stop], full)
        corr_flip = dot_product(full[start:stop], flips)
        ph = (63 - dot_product(phashes[start:stop, 0, :], phashes[:, 0, :])) / 2
        ph_flip = (63 - dot_product(phashes[start:stop, 0, :], phashes[:, 1, :])) / 2
        dh = (64 - dot_product(dhashes[start:stop], dhashes)) / 2
        candidates = cross & (
            (corr >= CORRELATION_LIMIT) | (corr_flip >= CORRELATION_LIMIT)
            | (ph <= PHASH_LIMIT) | (ph_flip <= PHASH_LIMIT) | (dh <= DHASH_LIMIT)
        )
        for i, j in zip(*np.where(candidates)):
            if start + i < j:
                record(
                    start + i, j, full_neg_corr=-float(max(corr[i, j], corr_flip[i, j])),
                    phash_distance=int(min(ph[i, j], ph_flip[i, j])),
                    dhash_distance=int(dh[i, j]),
                )

    for crop_index, name in enumerate(crop_names[1:], start=1):
        cropped = np.ascontiguousarray(normalized[:, crop_index, :])
        for start in range(0, len(rows), 128):
            stop = min(start + 128, len(rows))
            cross = splits[start:stop, None] != splits[None, :]
            corr = dot_product(cropped[start:stop], full)
            corr_flip = dot_product(cropped[start:stop], flips)
            for i, j in zip(*np.where(cross & ((corr >= CROP_LIMIT) | (corr_flip >= CROP_LIMIT)))):
                score = -float(max(corr[i, j], corr_flip[i, j]))
                key = tuple(sorted((start + i, j)))
                if score < matches[key].get("crop_neg_corr", 0):
                    record(start + i, j, crop_neg_corr=score)
                    matches[key]["crop"] = f"photo {1 if key[0] == start + i else 2}: {name}"
        print(f"Crop view {name}: {len(matches)} candidate pairs", flush=True)

    # A source page can host many images. It is a review signal only.
    by_url = defaultdict(list)
    for index, row in enumerate(rows):
        if row["url"].startswith(("http://", "https://")):
            by_url[row["url"]].append(index)
    for group in by_url.values():
        for offset, i in enumerate(group):
            for j in group[offset + 1:]:
                if splits[i] == splits[j]:
                    continue
                direct = float(full[i] @ full[j])
                mirror = float(full[i] @ flips[j])
                if max(direct, mirror) >= 0.65:
                    record(i, j, full_neg_corr=-max(direct, mirror), shared_url=True)

    ordered = sorted(
        matches.items(),
        key=lambda item: (
            min(item[1].get("full_neg_corr", 0), item[1].get("crop_neg_corr", 0)),
            item[1].get("phash_distance", 63),
            item[0],
        ),
    )
    return [
        {"pair_id": n, "left": rows[i], "right": rows[j], **measurements}
        for n, ((i, j), measurements) in enumerate(ordered, start=1)
    ]


def render_sheets(pairs):
    directory = OUTPUT / "contact_sheets"
    directory.mkdir(exist_ok=True)
    for offset in range(0, len(pairs), 20):
        page = pairs[offset:offset + 20]
        sheet = Image.new("RGB", (1200, ((len(page) + 3) // 4) * 178), "white")
        draw = ImageDraw.Draw(sheet)
        for index, pair in enumerate(page):
            x, y = (index % 4) * 300, (index // 4) * 178
            for position, key in enumerate(("left", "right")):
                row = pair[key]
                with Image.open(DATA_ROOT / row["image_path"]) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((140, 110), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + position * 148, y + 14))
                label = f"{row['split']} {row['class_name']}"
                draw.multiline_text(
                    (x + position * 148, y + 126),
                    textwrap.fill(label, 22), fill="black",
                )
            score = max(-pair.get("full_neg_corr", 0), -pair.get("crop_neg_corr", 0))
            hash_distance = pair.get("phash_distance", "-")
            draw.text((x, y), f"#{pair['pair_id']} corr={score:.3f} pHash={hash_distance}", fill="black")
        sheet.save(directory / f"pairs_{page[0]['pair_id']:04d}-{page[-1]['pair_id']:04d}.jpg", quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-pass", type=Path,
        help="Carry over pair identifiers and render contact sheets for additional candidates",
    )
    args = parser.parse_args()
    rows = load_rows()
    print(f"Retained subset: {dict(Counter(r['split'] for r in rows))}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        features_by_image = list(pool.map(features, rows))
    grids, phashes, dhashes, crop_names = zip(*features_by_image)
    candidates = cross_candidates(
        rows, np.stack(grids), np.stack(phashes), np.stack(dhashes), crop_names[0]
    )
    additional = candidates
    if args.first_pass:
        first_pass = json.loads(args.first_pass.read_text())["candidates"]
        previous = {
            (pair["left"]["image_path"], pair["right"]["image_path"]): pair
            for pair in first_pass
        }
        additional = [
            pair for pair in candidates
            if (pair["left"]["image_path"], pair["right"]["image_path"]) not in previous
        ]
        for n, pair in enumerate(additional, start=len(first_pass) + 1):
            pair["pair_id"] = n
        candidates = first_pass + additional
    print(
        f"Found {len(candidates)} candidates for manual review ({len(additional)} new)",
        flush=True,
    )
    (OUTPUT / "candidates.json").write_text(json.dumps({
        "pilot_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "screened_count": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "criteria": {
            "full_correlation_at_least": CORRELATION_LIMIT,
            "crop_correlation_at_least": CROP_LIMIT,
            "phash_hamming_at_most": PHASH_LIMIT,
            "dhash_hamming_at_most": DHASH_LIMIT,
            "crop_views": crop_names[0][1:],
            "shared_exact_url_correlation_at_least": 0.65,
        },
        "candidates": candidates,
    }, indent=2) + "\n", encoding="utf-8")
    render_sheets(additional)


if __name__ == "__main__":
    main()

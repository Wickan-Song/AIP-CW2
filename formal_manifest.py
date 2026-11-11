"""Freeze a versioned coursework manifest from reviewed near-duplicate exclusions."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath


def freeze_manifest(pilot_path, audit_path, data_root, output_path):
    pilot_bytes = Path(pilot_path).read_bytes()
    audit_bytes = Path(audit_path).read_bytes()
    manifest = json.loads(pilot_bytes)
    audit = json.loads(audit_bytes)
    pilot_hash = sha256(pilot_bytes).hexdigest()
    if audit["pilot_manifest_sha256"] != pilot_hash:
        raise ValueError("Near-duplicate decisions belong to a different pilot manifest.")
    splits = {"training": "train", "validation": "val", "test": "test"}
    classes = {
        row["class_id"]: (row["host"], row["class_name"])
        for row in manifest["class_roles"]["A"]
    }
    metadata = {}
    with (Path(data_root) / "Metadata.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            class_id = int(row["Index"])
            if class_id in classes:
                split = splits[row["Split"].strip().casefold()]
                image_path = str(PurePosixPath("images", split, row["Name"].strip()))
                host, disease = classes[class_id]
                if (row["Plant"].strip().casefold(), row["Disease"].strip()) != (host, disease):
                    raise ValueError(f"Metadata class mismatch for {image_path}")
                metadata[image_path] = dict(
                    split=split, class_id=class_id, class_name=disease,
                    host=host, image_path=image_path,
                )
    exact_excluded = {row["image_path"] for row in manifest["audit_exclusions"]}
    retained = metadata.keys() - exact_excluded
    decisions = audit["exclusions"]
    removed = {row["image_path"] for row in decisions}
    if len(removed) != len(decisions) or not removed <= retained:
        raise ValueError("Near-duplicate exclusions must be unique, retained source images.")
    for row in decisions:
        source = metadata[row["image_path"]]
        if any(key in row and row[key] != source[key] for key in ("split", "class_id", "class_name", "host")):
            raise ValueError(f"Near-duplicate decision disagrees with metadata: {row['image_path']}")
        kept = row.get("retained_path")
        if kept and kept not in retained - removed:
            raise ValueError(f"Near-duplicate representative is missing: {kept}")
        if kept and metadata[kept]["class_id"] != metadata[row["image_path"]]["class_id"]:
            raise ValueError(f"Conflicting-label group must not keep a representative: {kept}")
        if not row.get("reason"):
            raise ValueError(f"Missing exclusion reason for {row['image_path']}")
        manifest["audit_exclusions"].append({**source, **row})

    for assignment, rows in manifest["samples"].items():
        for row in rows:
            if row["image_path"] not in retained:
                raise ValueError(f"Pilot sample absent from source data: {row['image_path']}")
        filtered = [row for row in rows if row["image_path"] not in removed]
        partition_classes = lambda items: {(row["split"], row["class_id"]) for row in items}
        lost = partition_classes(rows) - partition_classes(filtered)
        if lost:
            raise ValueError(f"Assignment {assignment} loses class coverage: {sorted(lost)}")
        manifest["samples"][assignment] = filtered
        counts = Counter(f"{row['split']}/{row['role']}" for row in filtered)
        print(f"Assignment {assignment}: {len(rows) - len(filtered)} eligible rows excluded; {dict(counts)}")

    manifest["formal_audit"] = {
        "pilot_manifest_sha256": pilot_hash,
        "near_duplicate_decisions_sha256": sha256(audit_bytes).hexdigest(),
        "exact_exclusions": len(exact_excluded),
        "near_duplicate_exclusions": len(removed),
        "retained_by_original_partition": dict(sorted(Counter(
            metadata[image_path]["split"] for image_path in retained - removed
        ).items())),
    }
    output = Path(output_path)
    rendered = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if output.exists() and output.read_bytes() != rendered:
        raise FileExistsError(f"Frozen manifest differs: {output}")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rendered)
    print(f"Frozen manifest: {output}; SHA-256: {sha256(rendered).hexdigest()}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("artifacts/option2_manifest.json"))
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/plantseg"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/formal/manifest.json"))
    args = parser.parse_args()
    freeze_manifest(args.pilot, args.audit, args.data_root, args.output)

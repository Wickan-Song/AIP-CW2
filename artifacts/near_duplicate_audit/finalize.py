"""Convert visually confirmed near-duplicate pairs into exact exclusions."""

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PILOT_MANIFEST = OUTPUT.parent / "option2_manifest.json"
PARTITION_PRIORITY = {"test": 0, "val": 1, "train": 2}


def main():
    candidate_path = OUTPUT / "candidates.json"
    candidates = json.loads(candidate_path.read_text())
    review = json.loads((OUTPUT / "manual_review.json").read_text())
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if candidates["pilot_manifest_sha256"] != hashlib.sha256(PILOT_MANIFEST.read_bytes()).hexdigest():
        raise ValueError("The candidates belong to a different pilot manifest")
    if review["candidates_sha256"] != candidate_sha:
        raise ValueError("The visual review belongs to a different candidate file")
    rejected = set(review["rejected_pair_ids"])
    uncertain = set(review["uncertain_pair_ids"])
    pair_ids = {pair["pair_id"] for pair in candidates["candidates"]}
    if len(pair_ids) != len(candidates["candidates"]) or review["reviewed_pair_count"] != len(pair_ids):
        raise ValueError("Every candidate must have a unique reviewed pair ID")
    if rejected & uncertain or not (rejected | uncertain) <= pair_ids:
        raise ValueError("Visual review lists invalid candidate IDs")

    graph = defaultdict(set)
    rows = {}
    confirmed_pairs = []
    for pair in candidates["candidates"]:
        if pair["pair_id"] in (rejected | uncertain):
            continue
        left = pair["left"]["image_path"]
        right = pair["right"]["image_path"]
        rows[left], rows[right] = pair["left"], pair["right"]
        graph[left].add(right)
        graph[right].add(left)
        confirmed_pairs.append(pair)

    visited = set()
    exclusions = []
    components = []
    for origin in sorted(graph):
        if origin in visited:
            continue
        component = []
        queue = deque([origin])
        visited.add(origin)
        while queue:
            path = queue.popleft()
            component.append(path)
            for neighbor in sorted(graph[path] - visited):
                visited.add(neighbor)
                queue.append(neighbor)
        component.sort(
            key=lambda path: (PARTITION_PRIORITY[rows[path]["split"]], Path(path).name)
        )
        if len({rows[path]["split"] for path in component}) < 2:
            raise ValueError("Expected every component to cross partitions")
        conflicting_labels = len({rows[path]["class_id"] for path in component}) > 1
        retained = None if conflicting_labels else component[0]
        components.append({
            "component_id": len(components) + 1,
            "members": [{key: val for key, val in rows[path].items() if key != "url"} for path in component],
            "pair_ids": [
                pair["pair_id"] for pair in confirmed_pairs
                if pair["left"]["image_path"] in component
            ],
            "conflicting_labels": conflicting_labels,
            "retained_path": retained,
        })
        for path in component:
            if path == retained:
                continue
            reason = (
                "conflicting labels for visually confirmed near-duplicate photographs"
                if conflicting_labels else
                "visually confirmed near-duplicate photograph"
            )
            row = {
                "image_path": path,
                "reason": reason,
                "split": rows[path]["split"],
                "class_id": rows[path]["class_id"],
                "class_name": rows[path]["class_name"],
                "component_id": components[-1]["component_id"],
            }
            if retained is not None:
                row["retained_path"] = retained
            exclusions.append(row)
    exclusions.sort(key=lambda row: row["image_path"])
    dropped = {row["image_path"] for row in exclusions}
    if len(dropped) != len(exclusions):
        raise ValueError("Duplicate exclusion rows")
    if not all(
        pair["left"]["image_path"] in dropped or pair["right"]["image_path"] in dropped
        for pair in confirmed_pairs
    ):
        raise ValueError("A confirmed near-duplicate pair would remain")

    pilot = json.loads(PILOT_MANIFEST.read_text())
    coverage = {}
    for assignment, samples in pilot["samples"].items():
        remaining = [row for row in samples if row["image_path"] not in dropped]
        expected = {(row["split"], row["role"], row["class_id"]) for row in samples}
        observed = {(row["split"], row["role"], row["class_id"]) for row in remaining}
        if expected != observed:
            raise ValueError(f"Exclusions remove complete assignment {assignment} cells: {expected - observed}")
        coverage[assignment] = {
            split: dict(Counter(row["role"] for row in remaining if row["split"] == split))
            for split in PARTITION_PRIORITY
        }
    counts = {
        "screened_photographs": candidates["screened_count"],
        "reviewed_pairs": len(pair_ids),
        "confirmed_pairs": len(confirmed_pairs),
        "rejected_pairs": len(rejected),
        "uncertain_pairs": len(uncertain),
        "confirmed_components": len(components),
        "conflicting_components": sum(comp["conflicting_labels"] for comp in components),
        "excluded_photographs": len(exclusions),
        "excluded_by_partition": dict(Counter(row["split"] for row in exclusions)),
        "retained_photographs": candidates["screened_count"] - len(exclusions),
    }
    decisions = {
        "pilot_manifest_sha256": candidates["pilot_manifest_sha256"],
        "candidates_sha256": candidate_sha,
        "counts": counts,
        "assignment_coverage": coverage,
        "retention_rule": "Exclude every member of a conflicting-label component. For consistent labels keep one image in its original partition, prioritising test then val then train, with filename ties broken lexicographically.",
        "components": components,
        "exclusions": exclusions,
    }
    (OUTPUT / "decisions.json").write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "assignment_coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()

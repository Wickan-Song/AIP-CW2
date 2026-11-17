"""Independent audit 01: frozen manifest vs the CW2 task sheet.

Re-derives the Option 2 class-role mapping from the task-sheet rule, checks
partition/role containment, counts against the published table, exclusion
arithmetic, mask-path containment and the manifest SHA-256 chain.

Run from the repository root:  python3 review/01_manifest_audit.py
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/formal/manifest.json"

EXPECTED_COUNTS = {  # published in README / Chinese PDF / protocol
    "A": {("train", "known"): 1576, ("val", "known"): 266, ("val", "familiar_dev"): 54,
          ("val", "unfamiliar_dev"): 39, ("test", "known"): 490, ("test", "familiar_test"): 137,
          ("test", "unfamiliar_test"): 26},
    "B": {("train", "known"): 1432, ("val", "known"): 246, ("val", "familiar_dev"): 64,
          ("val", "unfamiliar_dev"): 39, ("test", "known"): 452, ("test", "familiar_test"): 132,
          ("test", "unfamiliar_test"): 26},
}
EXPECTED_EXCLUSIONS = {"exact": 201, "near_duplicate": 209}
EXPECTED_MANIFEST_SHA = "bd98c23f968b2b4f6e572a32226fcad20e42639deb6262ddc115a3c0b149be1e"
FAMILIAR_HOSTS = {"apple", "banana", "corn", "soybean", "tomato", "wheat", "zucchini"}

findings = []
def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append(f"{label}: {detail}")

def disease_of(class_name, host):
    prefix = host + " "
    assert class_name.startswith(prefix), class_name
    return class_name[len(prefix):]

def expected_roles_for(assignment, class_roles):
    """Task sheet 4.2: citrus = development unknowns, potato = test unknowns;
    for each other host sort its disease labels alphabetically — A reserves the
    1st (dev) and 2nd (test), B the 2nd (dev) and 3rd (test); rest are known."""
    by_host = {}
    for row in class_roles:
        by_host.setdefault(row["host"], []).append(row)
    expected = {}
    for host, rows in by_host.items():
        rows = sorted(rows, key=lambda r: disease_of(r["class_name"], r["host"]))
        if host == "citrus":
            for r in rows:
                expected[r["class_id"]] = "unfamiliar_dev"
        elif host == "potato":
            for r in rows:
                expected[r["class_id"]] = "unfamiliar_test"
        else:
            first, second = (0, 1) if assignment == "A" else (1, 2)
            for i, r in enumerate(rows):
                if i == first:
                    expected[r["class_id"]] = "familiar_dev"
                elif i == second:
                    expected[r["class_id"]] = "familiar_test"
                else:
                    expected[r["class_id"]] = "known"
    return expected

def main():
    raw = MANIFEST.read_bytes()
    m = json.loads(raw)
    check("manifest sha256 matches protocol/frozen chain",
          hashlib.sha256(raw).hexdigest() == EXPECTED_MANIFEST_SHA,
          hashlib.sha256(raw).hexdigest())

    for assignment in ("A", "B"):
        roles = m["class_roles"][assignment]
        check(f"[{assignment}] 43 host-disease categories", len(roles) == 43, f"n={len(roles)}")
        hosts = {r["host"] for r in roles}
        check(f"[{assignment}] exactly the nine coursework hosts",
              hosts == FAMILIAR_HOSTS | {"citrus", "potato"}, str(sorted(hosts)))

        expected = expected_roles_for(assignment, roles)
        actual = {r["class_id"]: r["role"] for r in roles}
        mismatches = {cid: (actual[cid], expected[cid]) for cid in actual
                      if actual[cid] != expected[cid]}
        check(f"[{assignment}] roles re-derived from alphabetical rule (task sheet 4.2)",
              not mismatches, str(mismatches))

        known = [r for r in roles if r["role"] == "known"]
        check(f"[{assignment}] 25 known classes", len(known) == 25, f"n={len(known)}")
        order = m["known_class_order"][assignment]
        known_sorted = [r["class_name"] for r in sorted(known, key=lambda r: r["class_id"])]
        check(f"[{assignment}] known_class_order equals known classes ordered by class id",
              order == known_sorted)

        samples = m["samples"][assignment]
        target_by_class = {r["class_name"]: t for t, r in enumerate(
            sorted(known, key=lambda r: r["class_id"]))}
        sample_targets = {s["class_name"]: s["target"] for s in samples if s["role"] == "known"}
        check(f"[{assignment}] sample targets equal position in known_class_order",
              sample_targets == target_by_class)
        counts = Counter((s["split"], s["role"]) for s in samples)
        check(f"[{assignment}] split/role counts equal published table",
              dict(counts) == EXPECTED_COUNTS[assignment],
              str({k: (counts.get(k), v) for k, v in EXPECTED_COUNTS[assignment].items()
                   if counts.get(k) != v}))

        allowed = {("train", "known"), ("val", "known"), ("val", "familiar_dev"),
                   ("val", "unfamiliar_dev"), ("test", "known"), ("test", "familiar_test"),
                   ("test", "unfamiliar_test")}
        bad = {k for k in counts if k not in allowed}
        check(f"[{assignment}] no forbidden split/role combination", not bad, str(bad))

        role_by_class = {r["class_id"]: r["role"] for r in roles}
        wrong = [s["image_path"] for s in samples
                 if s["role"] != role_by_class[s["class_id"]]]
        check(f"[{assignment}] every sample role matches its class role", not wrong,
              str(wrong[:3]))

        train_classes = {s["class_name"] for s in samples if s["split"] == "train"}
        known_names = {r["class_name"] for r in roles if r["role"] == "known"}
        check(f"[{assignment}] training uses known classes only", train_classes == known_names)

        val_test_unknown = {s["class_name"] for s in samples
                            if s["split"] == "val" and s["role"] == "familiar_test"}
        test_dev_unknown = {s["class_name"] for s in samples
                            if s["split"] == "test" and s["role"] == "familiar_dev"}
        check(f"[{assignment}] no test-unknown class images in validation", not val_test_unknown)
        check(f"[{assignment}] no dev-unknown class images in test", not test_dev_unknown)

        masks_outside = [s for s in samples if s.get("mask_path") and s["split"] != "train"]
        check(f"[{assignment}] lesion masks recorded for train rows only", not masks_outside)
        train_rows = [s for s in samples if s["split"] == "train"]
        missing_mask = [s for s in train_rows if not s.get("mask_path")]
        check(f"[{assignment}] every known training row carries a mask path", not missing_mask)

        paths = [s["image_path"] for s in samples]
        dupes = [p for p, n in Counter(paths).items() if n > 1]
        check(f"[{assignment}] image paths unique within assignment", not dupes, str(dupes[:3]))
        img_per_split = {sp: {s["image_path"] for s in samples if s["split"] == sp}
                         for sp in ("train", "val", "test")}
        check(f"[{assignment}] no image in two splits (leakage)",
              not (img_per_split["train"] & img_per_split["val"])
              and not (img_per_split["train"] & img_per_split["test"])
              and not (img_per_split["val"] & img_per_split["test"]))

        targets = {s["target"] for s in samples if s["split"] == "train"}
        check(f"[{assignment}] train targets cover 0..24 exactly", targets == set(range(25)),
              str(sorted(targets))[:80])
        unknown_targets = {s["target"] for s in samples if s["role"] != "known"}
        check(f"[{assignment}] non-known rows carry null target", unknown_targets == {None})

        # Every known class needs train support (task sheet: learnable) and
        # validation support (macro-F1 checkpoint selection must be defined).
        train_per_class = Counter(s["target"] for s in samples if s["split"] == "train")
        val_per_class = Counter(s["target"] for s in samples
                                if s["split"] == "val" and s["role"] == "known")
        check(f"[{assignment}] every known class has training support",
              min(train_per_class.values()) >= 1, f"min train={min(train_per_class.values())}")
        check(f"[{assignment}] every known class has validation support",
              min(val_per_class.values()) >= 1, f"min val={min(val_per_class.values())}")

    exc = m["audit_exclusions"]
    reasons = Counter(e["reason"] for e in exc)
    exact = sum(n for r, n in reasons.items() if "RGB pixels" in r and "near" not in r)
    near = sum(n for r, n in reasons.items() if "near-duplicate" in r)
    check("exclusions: 201 exact-pixel + 209 near-duplicate",
          (exact, near) == (EXPECTED_EXCLUSIONS["exact"], EXPECTED_EXCLUSIONS["near_duplicate"]),
          f"exact={exact} near={near} total={len(exc)} reasons={dict(reasons)}")
    retained = m["formal_audit"]["retained_by_original_partition"]
    total_retained = sum(retained.values())
    check("arithmetic: 4,349 original = 3,939 retained + 410 excluded",
          4349 == total_retained + len(exc), f"retained={total_retained} excluded={len(exc)}")
    ratios = {k: round(v / total_retained, 3) for k, v in retained.items()}
    check("partition ratios approximately 70/10/20 (task sheet 2.2)",
          abs(ratios["train"] - 0.70) < 0.05 and abs(ratios["val"] - 0.10) < 0.02
          and abs(ratios["test"] - 0.20) < 0.02, str(ratios))
    excluded_in_retained = {e["image_path"] for e in exc} & {
        s["image_path"] for a in ("A", "B") for s in m["samples"][a]}
    check("excluded images absent from retained samples", not excluded_in_retained,
          str(list(excluded_in_retained)[:3]))

    print(f"\n{len(findings)} finding(s).")
    sys.exit(1 if findings else 0)

if __name__ == "__main__":
    main()

"""Independent audit 03: recompute all frozen test metrics from per-image predictions.

Implements accuracy, macro-F1, unknown-positive AUROC and AUOSCR from the
task-sheet definitions (tie-aware), re-derives the shared/unseen disease-name
groups from the manifest, re-validates prediction rows against the manifest
test sets, and compares every stored number in test_results.json plus the
README/PDF headline table against the recomputation.

Run from the repository root:  python3 review/03_test_recomputation.py
"""

import json
import math
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "artifacts/formal"
SCORES = ["MSP", "MaxLogit", "Negative energy"]
GROUPS = ["familiar_host", "unfamiliar_host", "familiar_shared_name", "familiar_unshared_name"]
SEEDS = [42, 123, 2026]

README_NUMBERS = {  # (assignment, metric) -> (baseline, lesion)
    ("A", "known_accuracy"): (0.7850, 0.7844),
    ("A", "familiar_auoscr"): (0.6245, 0.6316),
    ("B", "known_accuracy"): (0.7471, 0.7478),
    ("B", "familiar_auoscr"): (0.6007, 0.5987),
}
PAIRED_DIFFS = {"A": 0.0071, "B": -0.0020}

findings = []
def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append(f"{label}: {detail}")

def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def disease_of(class_name, host):
    name = norm(class_name)
    prefix = norm(host) + " "
    assert name.startswith(prefix), class_name
    return name[len(prefix):]

def macro_f1(targets, preds, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(targets, preds) if t == c and p == c)
        support_t = sum(1 for t in targets if t == c)
        support_p = sum(1 for p in preds if p == c)
        denom = support_t + support_p
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(f1s) / n_classes, f1s

def auroc_unknown_positive(known_scores, unknown_scores):
    """AUROC with unknown as the positive class and -s as the score:
    P(unknown score < known score) with ties counted one half."""
    wins = ties = 0
    for u in unknown_scores:
        for k in known_scores:
            if u < k:
                wins += 1
            elif u == k:
                ties += 1
    return (wins + 0.5 * ties) / (len(unknown_scores) * len(known_scores))

def auoscr(known_scores, known_correct, unknown_scores):
    """Sweep acceptance thresholds; CCR over known vs FPR over unknown;
    trapezoid-integrate CCR across the full FPR range. Tie-aware."""
    n_k, n_u = len(known_scores), len(unknown_scores)
    rows = sorted(
        [("k", s, c) for s, c in zip(known_scores, known_correct)]
        + [("u", s, None) for s in unknown_scores],
        key=lambda r: -r[1])
    points = [(0.0, 0.0)]  # (FPR, CCR) before anything is accepted
    acc_u = acc_c = 0
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][1] == rows[i][1]:
            kind, _, corr = rows[j]
            if kind == "u":
                acc_u += 1
            elif corr:
                acc_c += 1
            j += 1
        points.append((acc_u / n_u, acc_c / n_k))
        i = j
    if points[-1][0] < 1.0:
        points.append((1.0, sum(known_correct) / n_k))
    area = 0.0
    for (f0, c0), (f1, c1) in zip(points, points[1:]):
        if f1 > f0:
            area += (f1 - f0) * (c0 + c1) / 2
    return area, points

def group_flags(rows, class_roles):
    """Recompute shared/unshared familiar-unknown groups from the disease-name rule."""
    known_hosts = {}
    for r in class_roles:
        if r["role"] == "known":
            known_hosts.setdefault(disease_of(r["class_name"], r["host"]), set()).add(norm(r["host"]))
    flags = {}
    for row in rows:
        if row["role"] != "familiar_test":
            flags[row["image_path"]] = None
        else:
            d = disease_of(row["class_name"], row["host"])
            shared = bool(known_hosts.get(d, set()) - {norm(row["host"])})
            flags[row["image_path"]] = "familiar_shared_name" if shared else "familiar_unshared_name"
    return flags

def main():
    manifest = json.loads((F / "manifest.json").read_text())
    frozen = json.loads((F / "evaluation/frozen_choices.json").read_text())
    stored_all = json.loads((F / "evaluation/test_results.json").read_text())

    n_comparisons = 0
    max_diff = 0.0
    worst = None
    summary_source = {}

    for assignment in ("A", "B"):
        class_order = manifest["known_class_order"][assignment]
        test_rows = [s for s in manifest["samples"][assignment] if s["split"] == "test"]
        shared_flag = group_flags(test_rows, manifest["class_roles"][assignment])
        by_path = {s["image_path"]: s for s in test_rows}
        lr = frozen["assignments"][assignment]["selected_learning_rate"]

        check(f"[{assignment}] prediction files cover exactly the manifest test set",
              {r["image_path"] for r in test_rows} == {
                  p["image_path"] for run in stored_all["assignments"][assignment]["runs"]
                  for p in json.loads((F / "evaluation" / run["prediction_path"]).read_text())})

        per_run = {}
        for run in stored_all["assignments"][assignment]["runs"]:
            key = f"{run['model']}_{assignment}_seed{run['seed']}_lr{lr}_epochs20"
            preds = json.loads((F / "evaluation" / run["prediction_path"]).read_text())
            thresholds = frozen["assignments"][assignment]["runs"][key]["thresholds"]

            # row-level validation against the manifest
            bad_rows = []
            for p in preds:
                s = by_path.get(p["image_path"])
                if s is None or p["role"] != s["role"] or p["class_name"] != s["class_name"]:
                    bad_rows.append(p["image_path"]); continue
                if s["role"] == "known" and p["target"] != s["target"]:
                    bad_rows.append(p["image_path"])
                if s["role"] != "known" and p["target"] is not None:
                    bad_rows.append(p["image_path"])
                if not (0 <= p["predicted_known_index"] < 25):
                    bad_rows.append(p["image_path"])
                for sc, v in p["scores"].items():
                    if not math.isfinite(v):
                        bad_rows.append(p["image_path"])
                for sc, t in thresholds.items():
                    if p["accepted"][sc] != (p["scores"][sc] >= t):
                        bad_rows.append(p["image_path"])
            check(f"[{assignment}] {key}: rows match manifest and frozen thresholds", not bad_rows,
                  str(bad_rows[:3]))

            known = [p for p in preds if p["role"] == "known"]
            targets = [p["target"] for p in known]
            preds_idx = [p["predicted_known_index"] for p in known]
            correct = [t == p for t, p in zip(targets, preds_idx)]
            my = {
                "known_count": len(known),
                "known_correct_count": sum(correct),
                "known_accuracy": sum(correct) / len(known),
                "known_macro_f1": macro_f1(targets, preds_idx, 25)[0],
            }
            groups = {}
            for p in preds:
                g = ("familiar_host" if p["role"] == "familiar_test"
                     else "unfamiliar_host" if p["role"] == "unfamiliar_test" else None)
                if g:
                    groups.setdefault(g, []).append(p)
                if p["role"] == "familiar_test":
                    groups.setdefault(shared_flag[p["image_path"]], []).append(p)

            my_scores = {}
            for sc in SCORES:
                ks = [p["scores"][sc] for p in known]
                t = thresholds[sc]
                acc_k = [s >= t for s in ks]
                my_scores[sc] = {
                    "threshold": t,
                    "known_acceptance": sum(acc_k) / len(ks),
                    "known_correct_acceptance": sum(a and c for a, c in zip(acc_k, correct)) / len(ks),
                    "known_accepted_count": sum(acc_k),
                    "groups": {},
                }
                for g, rows in groups.items():
                    us = [p["scores"][sc] for p in rows]
                    area, points = auoscr(ks, correct, us)
                    my_scores[sc]["groups"][g] = {
                        "unknown_count": len(us),
                        "auoscr": area,
                        "unknown_auroc": auroc_unknown_positive(ks, us),
                        "unknown_false_acceptance": sum(s >= t for s in us) / len(us),
                        "unknown_accepted_count": sum(s >= t for s in us),
                    }
                    stored_curve = run["metrics"]["scores"][sc]["groups"].get(g, {}).get("oscr_curve")
                    if stored_curve is not None and sc == stored_all["primary_score"]:
                        stored_pts = list(zip(stored_curve["unknown_false_acceptance"],
                                              stored_curve["known_correct_acceptance"]))
                        if stored_pts != points:
                            check(f"[{assignment}] {key} {sc} {g}: stored OSCR curve identical", False,
                                  f"first divergence at "
                                  f"{next(i for i, (a, b) in enumerate(zip(stored_pts, points)) if a != b)}")
            # compare everything against stored metrics
            sm = run["metrics"]
            def cmp(path, mine, stored, tol=1e-9):
                nonlocal n_comparisons, max_diff, worst
                if isinstance(stored, float) and stored != stored:  # NaN
                    stored = None
                if mine is None or stored is None:
                    ok = mine == stored
                    d = 0.0
                else:
                    d = abs(mine - stored)
                    ok = d <= tol
                n_comparisons += 1
                if d > max_diff:
                    max_diff, worst = d, f"{key}:{path}"
                if not ok:
                    check(f"[{assignment}] {key} {path} matches stored", False,
                          f"mine={mine} stored={stored}")
                return ok

            cmp("known_count", my["known_count"], sm["known_count"])
            cmp("known_correct_count", my["known_correct_count"], sm["known_correct_count"])
            cmp("known_accuracy", my["known_accuracy"], sm["known_accuracy"], 1e-12)
            cmp("known_macro_f1", my["known_macro_f1"], sm["known_macro_f1"], 1e-12)
            for sc in SCORES:
                ms, ss = my_scores[sc], sm["scores"][sc]
                for f in ("threshold", "known_acceptance", "known_correct_acceptance", "known_accepted_count"):
                    cmp(f"{sc}.{f}", ms[f], ss[f], 1e-12)
                for g in GROUPS:
                    for f in ("unknown_count", "auoscr", "unknown_auroc",
                              "unknown_false_acceptance", "unknown_accepted_count"):
                        cmp(f"{sc}.{g}.{f}", ms["groups"][g][f], ss["groups"][g][f], 1e-12)
            per_run[(run["model"], run["seed"])] = (my, my_scores)
            summary_source[assignment] = summary_source.get(assignment, [])
            summary_source[assignment].append((run["model"], run["seed"], my, my_scores))

        # headline recomputation vs README / PDF
        prim = stored_all["primary_score"]
        def agg(model, field):
            vals = []
            for m, seed, my, ms in summary_source[assignment]:
                if m != model:
                    continue
                if field == "known_accuracy":
                    vals.append(my["known_accuracy"])
                elif field == "known_macro_f1":
                    vals.append(my["known_macro_f1"])
                elif field == "familiar_auoscr":
                    vals.append(ms[prim]["groups"]["familiar_host"]["auoscr"])
            return statistics.mean(vals), statistics.stdev(vals)
        for field, key in (("known_accuracy", "known_accuracy"),
                           ("known_macro_f1", "known_macro_f1"),
                           ("familiar_auoscr", "familiar_auoscr")):
            b_mean, b_sd = agg("baseline", field)
            l_mean, l_sd = agg("lesion", field)
            stored_sum = stored_all["assignments"][assignment]["summary"][prim][
                {"known_accuracy": "known_accuracy",
                 "known_macro_f1": "known_macro_f1",
                 "familiar_auoscr": "familiar_host/auoscr"}[field]]
            check(f"[{assignment}] {field}: recomputed means match stored summary",
                  math.isclose(b_mean, stored_sum["baseline"]["mean"], abs_tol=1e-9)
                  and math.isclose(l_mean, stored_sum["lesion"]["mean"], abs_tol=1e-9),
                  f"baseline {b_mean:.4f}/{stored_sum['baseline']['mean']:.4f} "
                  f"lesion {l_mean:.4f}/{stored_sum['lesion']['mean']:.4f}")
            if field in ("known_accuracy", "familiar_auoscr"):
                rb, rl = README_NUMBERS[(assignment, field)]
                check(f"[{assignment}] {field}: means match README headline ({rb:.4f} → {rl:.4f})",
                      abs(b_mean - rb) < 5e-4 and abs(l_mean - rl) < 5e-4,
                      f"recomputed {b_mean:.4f} → {l_mean:.4f}")
            print(f"    [{assignment}] {field}: baseline {b_mean:.4f}±{b_sd:.4f} "
                  f"lesion {l_mean:.4f}±{l_sd:.4f}")
        paired = statistics.mean(
            ms[prim]["groups"]["familiar_host"]["auoscr"] - 
            next(m2[3][prim]["groups"]["familiar_host"]["auoscr"]
                 for m2 in summary_source[assignment] if m2[0] == "baseline" and m2[1] == seed)
            for m, seed, my, ms in summary_source[assignment] if m == "lesion")
        check(f"[{assignment}] paired AUOSCR diff matches README {PAIRED_DIFFS[assignment]:+.4f}",
              abs(paired - PAIRED_DIFFS[assignment]) < 5e-4, f"recomputed {paired:+.4f}")

    hyp_a = stored_all["assignments"]["A"]["hypothesis"]
    hyp_b = stored_all["assignments"]["B"]["hypothesis"]
    check("stored hypothesis verdict is 'not supported' in both assignments",
          stored_all["supports_hypothesis"] is False
          and not hyp_a["meets_all_criteria"] and not hyp_b["meets_all_criteria"],
          f"A diff={hyp_a['mean_familiar_host_auoscr_difference']:+.4f} "
          f"pos_seeds={hyp_a['positive_seed_differences']}/3; "
          f"B diff={hyp_b['mean_familiar_host_auoscr_difference']:+.4f} "
          f"pos_seeds={hyp_b['positive_seed_differences']}/3")

    print(f"\nnumeric comparisons against stored results: {n_comparisons} "
          f"(max |diff| = {max_diff:.2e} at {worst})")
    print(f"{len(findings)} finding(s).")
    sys.exit(1 if findings else 0)

if __name__ == "__main__":
    main()

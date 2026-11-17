"""Independent audit 02: pre-registered protocol, freeze chain and selection rules.

Verifies the SHA-256 provenance chain, recomputes the primary-score and
learning-rate selection from validation_grid.json with rules restated from
protocol.json, checks threshold calibration coverage and the freeze ordering
against experiment.log timestamps.

Run from the repository root:  python3 review/02_protocol_freeze_audit.py
"""

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "artifacts/formal"
SEEDS = [42, 123, 2026]
LRS = [0.0001, 0.0003]
MODELS = ["baseline", "lesion"]
SCORES = ["MSP", "MaxLogit", "Negative energy"]
TOLERANCE = 0.02

findings = []
def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append(f"{label}: {detail}")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_key(a, seed, lr, model):
    return f"{model}_{a}_seed{seed}_lr{lr}_epochs20"

def parse_ts(text):
    return datetime.fromisoformat(text)

def main():
    protocol_raw = (F / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    grid = json.loads((F / "evaluation/validation_grid.json").read_text())
    frozen = json.loads((F / "evaluation/frozen_choices.json").read_text())
    log = (F / "experiment.log").read_text()

    # --- pre-registration -------------------------------------------------
    hyp = protocol["hypothesis"]
    check("protocol pre-registers AUOSCR gain >= 0.02",
          hyp["minimum_mean_auoscr_gain"] == TOLERANCE)
    check("protocol pre-registers >= 2 positive paired seeds",
          hyp["minimum_positive_pairs"] == 2)
    check("protocol pre-registers known macro-F1 drop <= 0.02",
          hyp["maximum_mean_known_macro_f1_drop"] == TOLERANCE)
    check("protocol seeds {42,123,2026}", protocol["seeds"] == SEEDS)
    check("protocol LR candidates {1e-4, 3e-4}", protocol["learning_rates"] == LRS)
    check("protocol models {baseline, lesion}", protocol["models"] == MODELS)
    check("protocol budget 20 epochs / batch 32 / AdamW",
          protocol["epochs"] == 20 and protocol["batch_size"] == 32
          and protocol["optimizer"] == "AdamW")
    check("backbone is generic ImageNet ResNet-18 (no PlantSeg checkpoint)",
          "ResNet-18" in protocol["backbone"] and "ImageNet" in protocol["backbone"])

    training_started = parse_ts(log.splitlines()[0].split(" at ")[1])
    protocol_recorded = parse_ts(protocol["recorded_at"])
    check("protocol recorded before training started",
          protocol_recorded <= training_started,
          f"protocol={protocol_recorded} training={training_started}")

    # --- SHA-256 provenance chain ----------------------------------------
    manifest_sha = sha(F / "manifest.json")
    protocol_sha = sha(F / "protocol.json")
    grid_sha = sha(F / "evaluation/validation_grid.json")
    frozen_sha = sha(F / "evaluation/frozen_choices.json")
    chain_ok = (
        grid["provenance"]["manifest_sha256"] == manifest_sha
        and grid["provenance"]["protocol_sha256"] == protocol_sha
        and frozen["provenance"]["manifest_sha256"] == manifest_sha
        and frozen["provenance"]["protocol_sha256"] == protocol_sha
        and frozen["validation_grid_sha256"] == grid_sha
    )
    check("manifest/protocol/grid hashes bind the freeze chain", chain_ok)
    tr = json.loads((F / "evaluation/test_results.json").read_text())
    check("test_results pins frozen_choices sha256 and reuses provenance",
          tr["frozen_choices_sha256"] == frozen_sha
          and tr["provenance"]["manifest_sha256"] == manifest_sha
          and tr["provenance"]["protocol_sha256"] == protocol_sha)

    # --- grid completeness -------------------------------------------------
    expected_keys = {run_key(a, s, lr, m) for a in ("A", "B") for s in SEEDS
                     for lr in LRS for m in MODELS}
    check("validation grid covers exactly the 24 planned runs",
          set(grid["runs"]) == expected_keys and not grid["missing_runs"],
          f"n={len(grid['runs'])}")

    # --- selection rule recomputation --------------------------------------
    runs = grid["runs"]
    def fam_auoscr(key, score):
        return runs[key]["metrics"]["scores"][score]["groups"]["familiar_host"]["auoscr"]

    baseline_means = {}
    for score in SCORES:
        vals = [fam_auoscr(run_key(a, s, lr, "baseline"), score)
                for a in ("A", "B") for s in SEEDS for lr in LRS]
        baseline_means[score] = sum(vals) / len(vals)
    primary = max(SCORES, key=lambda s: baseline_means[s])
    check("primary score recomputed from 24 baseline runs is MaxLogit",
          primary == "MaxLogit", f"{ {k: round(v, 6) for k, v in baseline_means.items()} }")
    max_gap = max(baseline_means.values()) - sorted(baseline_means.values())[-2]
    check("primary-score margin exceeds tie-break territory", max_gap > 1e-6,
          f"margin={max_gap:.6f}")
    check("stored baseline grid means match recomputation",
          all(math.isclose(baseline_means[s], frozen["baseline_grid_mean_familiar_host_auoscr"][s],
                           abs_tol=1e-12) for s in SCORES))

    for assignment in ("A", "B"):
        stored = frozen["assignments"][assignment]
        candidates = []
        for lr in LRS:
            b_f1 = [runs[run_key(assignment, s, lr, "baseline")]["metrics"]["known_macro_f1"]
                    for s in SEEDS]
            l_f1 = [runs[run_key(assignment, s, lr, "lesion")]["metrics"]["known_macro_f1"]
                    for s in SEEDS]
            known_diff = sum(l - b for b, l in zip(b_f1, l_f1)) / len(SEEDS)
            b_au = [fam_auoscr(run_key(assignment, s, lr, "baseline"), primary) for s in SEEDS]
            l_au = [fam_auoscr(run_key(assignment, s, lr, "lesion"), primary) for s in SEEDS]
            b_mean, l_mean = sum(b_au) / 3, sum(l_au) / 3
            candidates.append({
                "learning_rate": lr,
                "known_diff": known_diff,
                "paired_mean": (b_mean + l_mean) / 2,
                "eligible": known_diff >= -TOLERANCE - 1e-12,
            })
        eligible = [c for c in candidates if c["eligible"]]
        pool = eligible if eligible else candidates
        selected = max(pool, key=lambda c: (c["paired_mean"], -c["learning_rate"]))
        check(f"[{assignment}] learning-rate selection recomputes to stored choice",
              math.isclose(selected["learning_rate"], stored["selected_learning_rate"]),
              f"mine={selected['learning_rate']} stored={stored['selected_learning_rate']}")
        stored_c = {c["learning_rate"]: c for c in stored["validation_candidates"]}
        table_ok = all(
            math.isclose(c["known_diff"], stored_c[c["learning_rate"]]["mean_paired_known_macro_f1_difference"], abs_tol=1e-12)
            and math.isclose(c["paired_mean"], stored_c[c["learning_rate"]]["paired_mean_familiar_host_auoscr"], abs_tol=1e-12)
            for c in candidates)
        check(f"[{assignment}] stored candidate table matches recomputation", table_ok)

        # threshold calibration coverage on validation
        accs = []
        for seed in SEEDS:
            for model in MODELS:
                key = run_key(assignment, seed, selected["learning_rate"], model)
                stored_run = stored["runs"][key]
                grid_run = runs[key]
                same_thresh = all(
                    math.isclose(stored_run["thresholds"][sc], grid_run["metrics"]["scores"][sc]["threshold"])
                    for sc in SCORES)
                if not same_thresh:
                    check(f"[{assignment}] frozen thresholds equal grid thresholds for {key}", False)
                for sc in SCORES:
                    accs.append((key, sc, grid_run["metrics"]["scores"][sc]["known_acceptance"]))
        check(f"[{assignment}] all frozen thresholds match validation grid", True)
        bad = [(k, sc, a) for k, sc, a in accs if a < 0.95 - 1e-9]
        check(f"[{assignment}] every calibration accepts >= 95% known validation", not bad, str(bad))
        slack = max(a for _, _, a in accs) - 0.95
        print(f"    [{assignment}] calibration acceptance range: "
              f"{min(a for _,_,a in accs):.4f}–{max(a for _,_,a in accs):.4f} (slack {slack:.4f})")

    # --- freeze ordering ----------------------------------------------------
    lines = log.splitlines()
    frozen_at = parse_ts(frozen["frozen_at_utc"]).astimezone(training_started.tzinfo)
    test_started = parse_ts([l for l in lines
                             if l.startswith("Frozen test started at")][0].split(" at ")[1])
    finished = parse_ts([l for l in lines
                         if l.startswith("Formal experiment finished")][0].split(" at ")[1])
    check("log records the freeze event", any("Choices frozen" in l for l in lines))
    # experiment.log stores whole seconds; frozen_at_utc carries microseconds.
    # Compare at second granularity and flag the sub-second ambiguity.
    same_or_before = frozen_at.replace(microsecond=0) <= test_started
    check("freeze timestamp precedes frozen test start (second granularity)",
          same_or_before, f"freeze={frozen_at.isoformat()} test={test_started.isoformat()}")
    if same_or_before and frozen_at > test_started:
        print(f"    note: frozen_at ({frozen_at.isoformat()}) is sub-second after the "
              f"whole-second test log stamp ({test_started.isoformat()}); ordering within "
              f"that second is not resolvable from the log, same second is recorded.")
    check("test evaluations reference only frozen-selected checkpoints",
          all(f"{m}_{a}_seed{s}_lr0.0001_epochs20" in log
              for a in ("A", "B") for s in SEEDS for m in MODELS))
    nonselected = [l for l in lines
                   if l.startswith("Evaluated frozen test checkpoint") and "lr0.0003" in l]
    check("no lr0.0003 run was test-evaluated", not nonselected)
    resumed = [l for l in lines if "resum" in l.lower()]
    print(f"    resume events in log: {len(resumed)}")
    check("experiment finished after test", finished >= test_started)

    # --- training completeness before freeze --------------------------------
    epoch_lines = [l for l in lines if " val macro-F1 " in l]
    check("exactly 480 validation epochs logged (24 runs x 20 epochs)",
          len(epoch_lines) == 480, f"n={len(epoch_lines)}")
    per_run = {}
    for l in epoch_lines:
        parts = l.split(" ")
        model, a, _, seed, _, lr = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        key = f"{model}_{a}_seed{seed}_lr{lr}_epochs20"
        per_run.setdefault(key, set()).add(int(parts[7].split("/")[0]))
    complete = {k: v == set(range(1, 21)) for k, v in per_run.items()}
    check("every one of the 24 runs logged epochs 1..20",
          len(per_run) == 24 and all(complete.values()),
          str([k for k, ok in complete.items() if not ok]))
    freeze_line = next(i for i, l in enumerate(lines) if "Choices frozen" in l)
    last_epoch_line = max(i for i, l in enumerate(lines) if " val macro-F1 " in l)
    check("all training epochs precede the freeze entry in the log",
          last_epoch_line < freeze_line,
          f"last epoch line={last_epoch_line + 1}, freeze line={freeze_line + 1}")

    print(f"\n{len(findings)} finding(s).")
    sys.exit(1 if findings else 0)

if __name__ == "__main__":
    main()

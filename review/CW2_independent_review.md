# CW2 Part 1 — Independent Review

**Reviewer:** Colin (personal branch `colin/review-writing`)
**Date:** 2026-11-11
**Scope:** fresh, full review of the completed implementation / experiments / analysis work, deliberately ignoring the earlier reviews (`docs/CW2_part1_review.md`, `docs/CW2_lesion_pilot_review.md`, `artifacts/review_20261111/`). Reference point: `docs/AIP304 CW2 Task Sheet.pdf` (§2 dataset & protocol, §4.2 split construction, §5.1 investigation, §5.2 metrics).
**Method:** static code review of `formal_manifest.py` / `formal_training.py` / `formal_evaluation.py` plus five reproducible audit scripts in this directory that re-derive every checkable claim from the frozen artifacts. No dataset download or weight reload was needed for this review: every headline number is recomputable from `artifacts/formal/` alone.

## Verdict

**The frozen experimental record is internally consistent and task-sheet conformant. No blocking issue found.** All 125 automated checks pass; 912 numeric comparisons between my independent recomputation and the stored results agree to floating-point noise (max |diff| = 7.8e-16). The negative hypothesis verdict (lesion supervision did not reach the pre-registered +0.02 AUOSCR gain) is correctly derived and honestly reported.

## How to reproduce this review

From the repository root (Python 3.9+, standard library only):

```shell
for s in review/0*_*.py; do python3 "$s"; done
```

Each script writes its full output to `review/results/` and exits non-zero on any failure.

| Script | What it verifies | Result |
|---|---|---|
| `01_manifest_audit.py` | class-role mapping re-derived from the task-sheet rule; partition containment; counts; exclusion arithmetic; mask-path confinement; manifest SHA-256 | 45/45 PASS |
| `02_protocol_freeze_audit.py` | pre-registered hypothesis & constants; SHA-256 chain manifest→protocol→grid→frozen→test; primary-score and LR selection recomputed from the 24-run validation grid; threshold calibration ≥95%; freeze-before-test ordering; 480/480 training epochs logged before freeze | 31/31 PASS |
| `03_test_recomputation.py` | independent reimplementation of accuracy / macro-F1 / unknown-AUROC / AUOSCR (tie-aware) + fixed-threshold counts from the 12 per-image prediction files; shared/unseen disease-name groups re-derived from the manifest; README headline numbers and hypothesis verdict re-derived | 30/30 PASS, 912 comparisons, max diff 7.8e-16 |
| `04_log_notebook_audit.py` | run-log completeness (24 runs × 20 epochs, one device, no NaN, verdict echoed); both notebooks: no error outputs, figures saved, execution counts, no CJK, training flags default off | 19/19 PASS |
| `05_code_conformity_checks.py` | static leakage & protocol checks on source: RGB-only evaluation, masks confined to known training rows, ImageNet-only weights, deterministic eval, seeding/pairing, checkpoint rule, constants | 30/30 PASS |

## Evidence against the task sheet

### §2.1 Dataset and subset

- Nine required hosts, 43 host–disease categories, PlantSeg v7 Zenodo record pinned; original partitions preserved (2676/435/828 ≈ 68/11/21% after cleaning).
- Duplicate handling satisfies "keep identical images within one partition, document exclusions": 131 same-label exact duplicates collapsed to one copy per group, 70 conflicting-label exact duplicates excluded outright; the near-duplicate pass (436 screened pairs → 213 confirmed → 209 further exclusions) is hash-chained into the manifest (`decisions.json` SHA-256 matches `formal_audit.near_duplicate_decisions_sha256`). Every exclusion row carries split/class/reason.
- Lesion masks appear **only** in training rows of the manifest; every known training row has one (checked in 01, re-checked in source in 05).

### §2.2 Training/validation/test protocol

- Fixed manifest recorded before training and bound by SHA-256 into protocol, validation grid, frozen choices and test results (02).
- Augmented copies stay with their original training image (augmentation is on-the-fly); preprocessing uses the pretrained weights' documented ImageNet statistics (explicitly permitted); no class-bearing filenames/metadata enter as inputs (targets come from manifest targets; images are the only input).
- Deterministic validation/test transforms, `model.eval()` + `@torch.inference_mode()`; checkpoint selection by known-validation macro-F1 (strictly-greater keeps the earliest epoch on ties, exactly as documented).
- Reproducibility details recorded: seeds {42, 123, 2026}, software deps in `requirements.txt`, pretrained identifier `IMAGENET1K_V1`, device `mps` in every run and log line, per-epoch timings (mean 19.8 s/epoch, 158.3 min total), checkpoint-selection rule in `protocol.json`.
- No PlantSeg/PlantWild-trained checkpoints anywhere; the only weights enum is ImageNet.

### §4.2 Split construction

- Roles re-derived from scratch in 01 and match exactly: citrus = development unknowns, potato = test unknowns; for the other seven hosts diseases sorted alphabetically with A reserving #1/#2 and B reserving #2/#3; apple black rot / mosaic virus / rust / scab land exactly as the task sheet specifies.
- 25 known classes per assignment; train contains known classes only; validation contains known + development unknowns; test contains known + test unknowns; **no test-unknown image in training/calibration/model-selection paths** (containment + threshold-from-validation-only both verified); dev-unknown classes never appear in test and vice versa.
- Familiar-host vs unfamiliar-host unknowns evaluated separately, with the potato control kept supplementary. Shared vs unseen disease names reported via a documented normalisation rule (NFKC, case-fold, whitespace, host-prefix strip, exact match): test familiar-unknowns split A 18 shared / 119 unshared, B 72 / 60; validation A 0 / 54, B 8 / 56.

### §5.1 Required investigation

- **Hypothesis (item 2):** falsifiable, pre-registered in `protocol.json` at 09:37:01+08:00, four minutes before training started (02): mean AUOSCR gain ≥ 0.02, ≥ 2 of 3 paired seeds positive, known macro-F1 drop ≤ 0.02, in **both** assignments. Used identically for validation and test interpretation.
- **Focused method (item 3):** auxiliary binary lesion prediction head on `layer2` features, BCE + soft-Dice averaged, weight 0.25, padding masked out; a mechanism with a stated failure mode, not a backbone swap.
- **Matched baseline & scoring (item 4):** identical backbone, initialisation, preprocessing, augmentation stream (lesion-head init is RNG-forked so the paired baseline's augmentation stream is unchanged), budget, and HP-search budget. MSP / MaxLogit / negative-energy evaluated on the same classifiers; MaxLogit selected **from baseline validation only** and then applied to both models — recomputed from the grid in 02, including the documented tie-break order and margins (0.0030 above runner-up).
- **Ablations & repeated trials (item 5):** the baseline *is* lesion-supervision-removed; 3 seeds × 2 assignments × 2 model configs = 6 runs per configuration, both mappings frozen before training; each assignment trained independently from generic initialisation (checkpoints pin manifest hash + class order, and refuse mismatches). Seed means, SDs and paired differences reported per assignment.
- Masks used only inside the training procedure; every validation/test prediction RGB-only (05 confirms the evaluation module cannot even see annotation paths).

### §5.2 Metrics

- AUOSCR/AUROC recomputed from per-image scores with an independent tie-aware implementation and match to 1e-15 (03); the stored OSCR curves are point-for-point identical to mine.
- Fixed operating point: thresholds are the `ceil(0.95·n)`-th largest known **validation** score; frozen before test; calibration acceptance landed at 0.9511–0.9512 across all 24 runs (02). Test known-acceptance drift (92.5%/91.1%) is reported, as the task sheet anticipates.
- Macro-F1 uses a fixed 25-class order with a documented zero-denominator rule; unknown AUROC uses unknown-positive with −s(x), both per §5.2.
- Headline numbers: A accuracy 78.50% → 78.44%, familiar AUOSCR 0.6245 → 0.6316 (paired +0.0071, 2/3 seeds); B 74.71% → 74.78%, 0.6007 → 0.5987 (paired −0.0020, 1/3 seeds) — every one of these re-derived from raw predictions in 03 and equal to the README/PDF table. `supports_hypothesis = false` follows from the pre-registered criteria and is echoed in the log.

## Findings (none blocking)

1. **Sub-second freeze ordering is unprovable from the log alone.** `frozen_at_utc` = 13:23:35.629+08:00 vs the whole-second "Frozen test started at 13:23:35+08:00". Within-second order cannot be resolved; the runner's code path (freeze returns before test starts) and the absence of any lr0.0003 test evaluation make a violation implausible. Suggestion for future runs: log millisecond timestamps at freeze/test boundaries.
2. **Thin per-class validation support.** The smallest known-class validation support is 2 images (per assignment), so checkpoint-selection macro-F1 is noisy at the class level. This affects baseline and lesion symmetrically (paired design), so the comparison stands, but per-class validation numbers should not be over-interpreted.
3. **Validation-time shared-name group is empty for A** (0/54) and tiny for B (8/64): the shared-vs-unseen comparison is only meaningful on the test split (A 18/119, B 72/60). Fine for reporting, worth stating explicitly in the report.
4. **Lesion-loss weight (0.25) and head placement are not ablated**, and the optional shuffled-mask control was not run. The task sheet frames the shuffled-mask control as optional ("if additional time is available") and the main ablation (remove lesion supervision) is exactly the baseline, so this is not a compliance failure — but the report should acknowledge it as a limitation of the mechanism attribution.
5. **`known_class_order` is keyed by class-id order rather than an explicit target index** in the manifest; the audit confirms they coincide, and checkpoints re-verify at load time, so there is no residual risk — noted only because a naive future edit could desynchronise them (the load-time guard would catch it).

## Remaining coursework items (out of Part 1 scope, tracked separately)

- Literature review with ≥ 3 closely related methods (task sheet §5.1 item 1) — not yet done.
- Fresh Colab/Kaggle execution check of the notebooks (§6.1) — not yet done; notebooks currently rely on the repo's `formal_*.py` modules and restored assets.
- Single-image inference demonstration cell for the notebook/video — not yet present.
- English report (≤ 5 pages, AI disclosure, contribution statement) and ≤ 3-minute video — not started.
- Optional exploratory extras (lesion-weight ablation, shuffled-mask control) — must be labelled exploratory if run; frozen results must remain untouched.

## Conclusion

Implementation, experiments and analysis are complete, mutually consistent, and faithfully derived under the pre-registered protocol. The negative result (no reliable AUOSCR gain from lesion supervision; known-class performance preserved) is a valid, well-supported outcome under the task sheet and can be written up as-is.

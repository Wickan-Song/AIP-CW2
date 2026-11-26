# CW2 Report — Figure Captions and Results Draft

**Status: NEWLY DRAFTED.** No original English manuscript exists yet; every sentence below is new text derived from the frozen artifacts (`artifacts/formal/`), not a polish of prior prose. All numbers trace to `test_results.json`, `frozen_choices.json`, `validation_grid.json`, `manifest.json`, or `artifacts/formal/diagnostics/` and were independently recomputed during the Part-1 review (912 comparisons, max deviation 7.8e-16).

Figures: `artifacts/formal/figures/report/` (PDF vector + PNG preview, 300 DPI, Okabe-Ito palette). Generator: `report_figures.py`; provenance in `artifacts/formal/figures/report/provenance.json`. Figures 1–8 regenerate from the frozen JSONs alone; Figure 9 additionally reads the restored dataset images.

---

## Figure captions

### Figure 1 — Coursework subset and Option 2 class assignments

**(a)** Known-class training-set frequencies for the 18 host–disease classes that have training images under both assignments, after exact-pixel and near-duplicate audits (blue, Assignment A: 1,576 images; vermillion, Assignment B: 1,432 images). Counts range from 18 to 134 images per class; no class was empty in either assignment. **(b)** Partition and role composition within each assignment's retained subset (Assignment A: 2,588 images; Assignment B: 2,391; both drawn from the 3,939 images retained after the duplicate audits), using the original PlantSeg partitions. Train contains known classes only; validation contains known and development-unknown images (familiar-host and citrus controls); test contains known and test-unknown images (familiar-host and potato controls). Segment labels show image counts; totals above each stack. Class-role rule: within each familiar host, diseases are sorted alphabetically; Assignment A reserves the 1st (development unknown) and 2nd (test unknown), Assignment B the 2nd and 3rd; all remaining classes are known. For apple: black rot (A: dev. unknown, B: known), mosaic virus (A: test unknown, B: dev. unknown), rust (A: known, B: test unknown), scab (A: known, B: known). Citrus serves only as the development-unfamiliar host; potato only as the test-unfamiliar host.

### Figure 2 — Test score distributions under the frozen rejection rule

Maximum-logit (MaxLogit) score distributions on the frozen test split, pooled over the six models per assignment (baseline and lesion-supervised × three seeds), for known images (blue), familiar-host unknown images (vermillion), and unfamiliar-host unknown images (green). Dashed vertical lines are the per-model, per-seed decision thresholds, each calibrated on known validation images to accept at least 95% of them (observed validation acceptance 95.11–95.12%). Assignment A: n = 490 known, 137 familiar-host unknown, 26 unfamiliar-host unknown images. Assignment B: n = 452, 132, 26. The three distributions overlap substantially; the calibrated thresholds fall inside the familiar-host unknown mode, which explains the high false-acceptance rates reported in the text.

### Figure 3 — Paired seed-level comparison of the two training regimes

Each point pairs one seed's baseline (x-axis) with the lesion-supervised model's (y-axis) value; grey drops connect the pair to the identity line, and the black diamond is the seed mean ± SD (n = 3 paired seeds). Left pair of panels: familiar-host AUOSCR (area under the open-set classification-rate curve; higher is better). Assignment A mean paired difference +0.0071 with 2/3 seeds positive; Assignment B −0.0020 with 1/3 positive. Right pair: known-class macro-F1; differences +0.0011 (A) and +0.0014 (B). Differences are descriptive; no significance test was pre-registered.

### Figure 4 — Where false-accepted familiar-host unknowns go

The ten most frequent true-unknown → predicted-known destinations for familiar-host test-unknown images accepted at the frozen MaxLogit thresholds, counted over the three seeds per model (baseline blue, lesion-supervised vermillion). Assignment A: n = 137 familiar-host unknown images (411 seed-level judgments per model); Assignment B: n = 132 (396 per model). The dominant confusions pair visually similar diseases on the same host, e.g. zucchini powdery mildew → zucchini yellow mosaic virus and wheat leaf rust → wheat stripe rust; in Assignment B, apple rust (a test-unknown class) is most often accepted as apple scab.

### Figure 5 — Open-set classification-rate curves

Open-set classification rate (correct-and-accepted known rate) against familiar-host unknown false-acceptance rate, swept over all acceptance thresholds, for each seed (thick line: seed 42; thin: seeds 123, 2026; blue: baseline, vermillion: lesion-supervised). The dotted diagonal is chance-level coupling. Areas under these curves (averaged over seeds) are the AUOSCR values in Figure 3; curves for both models lie close together in both assignments, consistent with the small paired differences.

### Figure 6 — Familiar-host unknowns split by shared versus unseen disease names

Familiar-host AUOSCR for test-unknown classes whose disease name also occurs among known classes on another host (shared) versus those with no such occurrence (unseen), using Unicode NFKC, case-folded, whitespace-normalised names with the host prefix removed and exact matching. Bars are seed means ± SD (n = 3 seeds; dots are individual seeds). Group sizes: A 18 shared / 119 unseen; B 72 / 60. No group shows a consistent advantage for lesion supervision; the shared-name groups are small and their seed variation is correspondingly larger.

### Figure 7 — Known-class test confusion matrices (full, colour-only)

Known-class test confusion for Assignments A and B, pooled over the three seeds per model (A: 490 images × 3 seeds = 1,470 seed-level judgments; B: 452 × 3 = 1,356). Rows are true classes, columns predicted classes, in the fixed 25-class order; colour shows row-normalised recall; grey lines separate plant hosts. Cells are deliberately unnumbered to keep the structure legible; the matrices are strongly diagonal (A: 1,154/1,470 and 1,153/1,470 correct for baseline and lesion-supervised; B: 1,013/1,356 and 1,014/1,356). Exact per-class counts are available in the frozen per-image prediction files; the host-level aggregation with readable numbers is Figure 7b.

### Figure 7b — Host-level aggregation of the known-class confusion

The same pooled counts aggregated over the 25 known classes into their seven familiar hosts, with per-cell numbers (seed-level judgments out of each row total). Most predictions stay on the true host: 91.5%/91.4% (A baseline/lesion), 90.1%/90.6% (B). Of the class-level errors, the majority are within-host disease confusions (A: 191/316 and 190/317; B: 209/343 and 214/342), with the largest single cell in every class-level matrix being tomato yellow leaf curl virus → tomato mosaic virus (13–17 pooled). The two training regimes produce nearly identical host-level patterns.

### Figure 8 — Validation learning curves at the selected learning rate

Known-validation macro-F1 per epoch (20 epochs) at the selected learning rate (1e-4). Lines are seed means (n = 3 paired seeds); bands span the seed minimum and maximum; dots mark each seed's selected checkpoint (highest validation macro-F1, earliest epoch on ties). Curves are parsed from the recorded training log; both arms share the initialisation, augmentation stream and training budget.

### Figure 9 — Test prediction gallery under the frozen rejection rule

Test-set examples under the frozen MaxLogit rule, ranked by median score over the three lesion-supervised seeds. Rows: correctly accepted known images (green), correctly rejected familiar-host unknowns (blue), and false-accepted familiar-host unknowns (vermillion); titles give the true class, the decision, and the median score against the frozen threshold. In Assignment A the highest-scoring false accept is an apple mosaic virus image accepted as soybean mosaic (median MaxLogit 9.6), an error consistent with the shared-disease-name grouping of Figure 6. In Assignment B, apple rust appears both among the lowest-scoring correct rejects (2.1) and among false accepts as apple scab (7.4–7.6), so acceptance is image-specific rather than class-uniform.

---

## Results (draft, newly written)

We tested whether adding a binary lesion-mask segmentation loss during training improves rejection of held-out disease classes on familiar plant hosts, relative to an otherwise matched classifier trained without lesion masks, while preserving known-class recognition when inference uses RGB images alone. Both arms shared an ImageNet-pretrained ResNet-18 backbone, preprocessing, augmentation stream, 20-epoch budget, and hyperparameter-search budget; each assignment (A, B) trained three paired seeds per arm from generic initialisation. Model selection used known-validation macro-F1; the rejection score (MaxLogit, selected from baseline validation results over MSP, MaxLogit, and negative energy) and per-model thresholds (calibrated to accept ≥95% of known validation images) were frozen for both assignments before any test inference. Validation macro-F1 rose over the 20-epoch budget in every run and the selected checkpoints varied by seed (Figure 8).

Lesion supervision left known-class recognition essentially unchanged. Known-test accuracy was 78.50% ± 1.23 (baseline) versus 78.44% ± 0.42 (lesion-supervised) in Assignment A and 74.71% ± 0.92 versus 74.78% ± 0.80 in Assignment B; known-class macro-F1 differed by +0.0011 and +0.0014 respectively, well inside the pre-registered tolerance of a 0.02 mean decrease (Figure 3, right panels).

The pre-registered primary outcome was not met. Mean familiar-host AUOSCR improved by +0.0071 in Assignment A (2/3 seeds positive) and decreased by −0.0020 in Assignment B (1/3 positive); neither assignment reached the pre-registered criterion of a ≥0.02 mean gain with at least 2/3 positive paired seeds (Figure 3, left panels). Under the hypothesis-testing rules fixed before training, the evidence does not support the claim that lesion supervision improves rejection of unfamiliar diseases on familiar hosts.

At the frozen operating points, most familiar-host unknown images were still accepted as known classes: false-acceptance rates averaged 82.48% (baseline) and 82.97% (lesion-supervised) in Assignment A, and 75.25% and 76.52% in Assignment B, while test known-acceptance stayed near 91–93%. Figure 2 shows why: the MaxLogit distributions of known and familiar-host unknown images overlap heavily, and the validation-calibrated thresholds fall within the unknown mode. Figure 4 shows that the accepted unknowns are not random: the most frequent confusions pair visually similar diseases on the same host (e.g. zucchini powdery mildew → yellow mosaic virus; wheat leaf rust → stripe rust), and apple rust is most often accepted as apple scab. In validation, corn northern-leaf-blight images were accepted in 24 of 27 seed-level judgments by lesion-supervised models, 18 of them as gray leaf spot (diagnostics log). Known-class errors followed the same within-host pattern: 90–92% of seed-level judgments stayed on the true plant host, and 60–63% of the class-level errors were within-host disease confusions, the largest being tomato yellow leaf curl virus → tomato mosaic virus (Figures 7 and 7b). Per-image examples show the failure is image-specific rather than class-uniform: in Assignment A the top-scoring false accept is an apple mosaic virus image accepted as soybean mosaic (median MaxLogit 9.6, a shared disease name), while in Assignment B apple rust images span both the lowest correct reject (2.1) and false accepts as apple scab (7.4–7.6; Figure 9). Splitting familiar-host unknowns by disease name shared with a known class did not reveal a consistent pattern favouring either arm (Figure 6), and the unfamiliar-host controls (citrus, potato) are reported separately as the protocol requires.

Together, the lesion-supervised models recognised known diseases as well as the matched baselines but did not reject unfamiliar familiar-host diseases more reliably. The lesion branch did highlight visible symptoms in many images, including untrained diseases, but symptom localisation did not translate into more selective open-set scores; the current evidence does not attribute the shortfall to the backbone or optimiser, and a lesion-loss-weight ablation and a shuffled-mask control remain untested.

---

## Change notes and verification summary

**What was produced.** `report_figures.py` (new; twelve figure files across ten numbered figures, including the host-level Figure 7b, + provenance manifest; Figures 1–8 regenerate from frozen JSONs only, Figure 9 also uses restored images) and this draft. No existing file was modified; outputs went to the new directory `artifacts/formal/figures/report/`.

**Why.** The figures follow the coursework task-sheet §6.3 evidence list (class frequencies, score distributions, OSCR curves, failure analysis with error bars/seed points) with a colour-blind-safe Okabe-Ito scheme, consistent model colours across all figures, seed-level raw points alongside means ± SD, and vector PDF export. Captions state panel contents, group sizes, thresholds, the frozen scoring rule, and that differences are descriptive.

**Scientific meaning.** Unchanged by construction: every plotted value is read directly from the frozen evaluation artifacts; no re-aggregation, re-thresholding, or re-selection was performed.

**Consistency checks.** All caption numbers re-verified against `test_results.json` / `frozen_choices.json` / `manifest.json` / `experiment.log` in this session: accuracy/F1/AUOSCR means and SDs, paired differences and positive-seed counts, group sizes 490/137/26 and 452/132/26, shared/unseen 18/119 and 72/60, calibration acceptance 95.11–95.12%, false-acceptance means, destination counts, corn NLB 24/27 → 18 gray leaf spot; pooled confusion diagonals (A 1,154/1,470 and 1,153/1,470; B 1,013/1,356 and 1,014/1,356), largest class-level off-diagonal cells (tomato yellow leaf curl virus → tomato mosaic virus, 13–17 pooled; no off-diagonal cell reaches 20 in Assignment A), and the host-level aggregation (stay-on-true-host 91.5/91.4% in A, 90.1/90.6% in B; within-host share of class-level errors 191/316, 190/317, 209/343, 214/342 — every Figure 7b cell number cross-checked against the recomputed 7×7 matrices); Figure 9 gallery rows re-derived from the per-image predictions (A top false accept: apple mosaic virus → soybean mosaic, median 9.6; B apple rust: reject at 2.1 and false accepts at 7.4–7.6). The "18 shared classes" in Figure 1a is the intersection of both assignments' known-training sets and is stated as such. Training curves (Figure 8) are parsed from `experiment.log` at the selected learning rate; selected epochs are read from `validation_grid.json`.

**Not done / open.** No statistical significance tests exist (none were pre-registered); captions avoid them. The Results draft above is provisional text for the 5-page report and has not been merged into any manuscript. Per your instruction, the scientific-visualization skill is not cited; its contribution can instead be mentioned in the report's AI-use disclosure if you later prefer.

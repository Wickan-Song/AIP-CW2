# AIP304 CW2 Lesion Pilot Review

Reviewed on 11 November 2026. Scope: the assignment-A, seed-42 matched baseline/lesion pilot in `Predict.ipynb`, checked against task-sheet Sections 2, 4.2, 5 and 6.

## Verdict

No blocking implementation or metric error was found in the reviewed pilot. Fresh checkpoint inference reproduced the saved results. The pilot provides preliminary validation evidence; it does not establish the full research hypothesis or constitute a complete coursework submission.

This review did not repeat either 20-epoch training run. It re-executed the data audit, checked initialization and augmentation pairing, tested the auxiliary loss and model forward paths, reloaded both saved models, and independently recomputed validation metrics. No test prediction was performed. Reading test RGB images for the label-independent duplicate audit is distinct from evaluating a model on the test set.

## Checks completed

- All 4,349 selected RGB files decoded. The audit again produced 73 cross-partition duplicate groups, 34 conflicting-label groups, 201 exclusions and 4,148 retained images. The frozen manifest remained byte-for-byte unchanged, with SHA-256 `cc9a775edf59da9b52e2c3a036b7fb0045dd0f6b5d33fcbf2be125b1a760b594`.
- A/B class assignments and original partitions were reconstructed from the downloaded metadata. Each assignment has 25 known classes. Training contains only eligible known classes; saved validation/test records contain no mask paths.
- Baseline and lesion classifier initial state tensors matched exactly. The first three training batches (96 images), including sampled order, RGB augmentation tensors and labels, matched exactly after resetting the paired seed. Static inspection found no extra random draw in the lesion forward/loss path; auxiliary-head initialization is isolated from the CPU RNG stream.
- Saved checkpoint settings match for pretrained source, seed, assignment, manifest, class order, batch size, optimizer, learning rate, weight decay and scheduler. All 20 recorded learning-rate values also match.
- The lesion-only loss produced a nonzero gradient in the shared layer-2 convolution. Its gradient did not reach the classification head when classification loss was omitted. Synthetic padding tests gave zero padded-pixel gradients and unchanged loss when padded logits changed.
- The two classification forward paths agreed in evaluation mode. Validation inference used RGB tensors only, evaluation mode and disabled gradients on the same 374 images (278 known, 54 familiar-host unknown, 42 citrus unknown).
- Both checkpoints select the earliest maximum known-validation macro-F1 from their 20-epoch histories: baseline epoch 15, lesion epoch 17.
- Independent class-by-class F1 and pairwise-ranking AUOSCR/AUROC reproduced the saved results for both models, all three scores, and both unknown groups. Synthetic tests covered tied scores, all incorrect known predictions and perfect separation. Known-only threshold calibration accepted 265/278 known images for each score and model.
- Notebook code compiled, stored outputs contained no errors, and notebook text contained no Chinese characters.
- The notebook, original manifest, both checkpoints and original pilot-results JSON remained unchanged during the review. Their SHA-256 hashes are recorded in the evidence JSON.

## Reproduced results

| Model | Selected epoch | Known correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 15 | 219/278 | 78.7770% | 0.76613552 |
| Lesion supervision | 17 | 227/278 | 81.6547% | 0.78253098 |

| Score | Familiar-host baseline AUOSCR | Familiar-host lesion AUOSCR | Paired difference |
|---|---:|---:|---:|
| MSP | 0.59372502 | 0.60911271 | +0.01538769 |
| Maximum logit | 0.60778044 | 0.62743139 | +0.01965095 |
| Negative energy | 0.60991207 | 0.63156142 | +0.02164935 |

Negative-energy unknown AUROC increases from 0.70883293 to 0.72049027. At 95.3237% known acceptance, familiar-host unknown false acceptance is 47/54 (87.0370%) for the baseline and 43/54 (79.6296%) for the lesion model. This remains high. The lesion model rejects only 11/54 of these unknown images at this operating point.

## Interpretation and work still required

1. **The full hypothesis remains unresolved.** A single A/seed-42 validation pair cannot establish consistency across two assignments and three seeds. The approximately 0.02165 negative-energy AUOSCR increase is preliminary. AUOSCR combines classification and rejection, so it cannot be described as a pure rejection improvement.
2. **Formal selection is unfinished.** Complete the paired seed/configuration comparisons, select the common primary score using baseline validation results only, and freeze both assignments' configurations and known-validation thresholds before inspecting either assignment's test results. Retain all three required score comparisons and report means, standard deviations and paired differences separately for A and B.
3. **Near-duplicate leakage remains unresolved.** The current audit covers exact decoded RGB duplicates. Investigate near duplicates before formal runs or disclose the unresolved risk explicitly, as allowed by the task sheet. If exclusions change, version the manifest and retrain all affected matched comparisons.
4. **Shared disease-label analysis is absent.** Record whether each familiar-host unknown shares its disease label with a known class on another host; document normalization and report subgroup results where sample sizes permit.
5. **Research and submission requirements remain.** Add the current comparison of at least three related methods, evidence-based positioning of the proposed adaptation, fuller hypothesis interpretation, final error analysis and RGB-only inference demo. Verify a clean Colab/Kaggle run with dependencies, dataset/checkpoint access and demonstration/full modes. Complete the five-page report, three-minute face-visible video and AI disclosure.
6. **Provenance limits.** The current notebook states the hypothesis and tolerance, but this review alone cannot independently certify when that text was first written. Retain the earlier records and freeze a timestamped protocol for the formal experiment.

No training code or reported score was changed by this review.

## Reproduction and evidence

- Notebook: `../Predict.ipynb`
- Audit script: `../artifacts/review_20261111/check_pilot.py`
- Machine-readable review: `../artifacts/review_20261111/review_results.json`
- Recomputed logits: `../artifacts/review_20261111/validation_logits.npz`
- Fresh data-audit output: `../artifacts/review_20261111/data_audit.txt`

Run the audit script with the project's `.venv/bin/python`. It performs no model training and verifies that the reviewed source artifacts are not modified. The logits follow assignment-A validation-row order in the frozen manifest.

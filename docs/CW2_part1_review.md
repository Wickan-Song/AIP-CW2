# AIP304 CW2 Part 1 Review

Reviewed on 10 November 2026 against task-sheet sections 2, 4.2, 5 and 6.

## Conclusion and scope

The current data preparation and assignment-A baseline satisfy the requirements applicable to this stage. This remains a partial Option 2 implementation. This review re-executed data preparation and both assignment previews, tested the assignment-B loader, and independently evaluated the saved assignment-A checkpoint. Original 20-epoch training outputs remain in the notebook.

## Verified items

| Item | Evidence |
|---|---|
| Release and subset | PlantSeg v7 archive checksum passed. Nine hosts, 4,349 source images, 43 original classes. |
| Original partitions | Each retained sample matches its source metadata partition. Exclusions remove samples without moving retained images. |
| Duplicates | EXIF-corrected RGB pixels and dimensions are hashed. There are 73 cross-partition groups and 34 conflicting-label groups. Conflicting-label groups are excluded. Consistent-label groups retain one image by test, validation, training priority, then filename. |
| Exclusions | 201 exclusions are documented. 4,148 images remain. Each retained pixel-hash group has at most one image. |
| Class roles | A and B use the prescribed alphabetical reservations and citrus/potato controls. Each has 25 known classes. |
| Eligibility and targets | Training uses known classes in the original training partition. Each validation/test sample has a permitted role. Known targets match the saved class order. Unknown targets are unset. |
| Training masks | All 1,980 distinct images in the union of A/B permitted training sets have readable, dimensionally aligned masks after EXIF correction. Binary foreground remains after resizing to 224 pixels. |
| Model inputs | The baseline uses RGB pixels and classification targets. Filenames and host metadata serve indexing/auditing only. |
| Preprocessing | Aspect ratio preserved, mean-colour padding, ImageNet normalization. Random transformations apply to training; validation transformations are deterministic. |
| Evaluation | Evaluation mode and disabled gradients. Macro-F1 averages class-wise F1 in a fixed order. Highest known-validation Macro-F1 selects the checkpoint, with the earliest epoch retained on ties. |
| Checkpoint | Saved manifest hash and class order match. Training history has 20 epochs and selects epoch 15. |
| Independent metrics | Reloaded weights classify 219 of 278 known validation images correctly. Accuracy 0.7877697842; independently computed Macro-F1 0.7661355213. |
| English | Notebook source and saved textual outputs contain no Chinese characters. |

Manifest SHA-256 is cc9a775edf59da9b52e2c3a036b7fb0045dd0f6b5d33fcbf2be125b1a760b594.

| Assignment | Known training | Validation known / familiar unknown / unfamiliar unknown | Test known / familiar unknown / unfamiliar unknown |
|---|---:|---:|---:|
| A | 1,666 | 278 / 54 / 42 | 495 / 145 / 27 |
| B | 1,516 | 257 / 70 / 42 | 460 / 132 / 27 |

## Findings addressed

1. **Assignment-B preview failure.** The preview explicitly selected apple rust. It is a test-unknown class in B, so selecting it from B's training rows raised StopIteration. The preview now selects the first eligible known training image. B's preview and a four-image loader batch passed the regression check. The B preview was also visually inspected.
2. **Incomplete displayed audit.** The notebook now displays counts for each of the 43 classes across original partitions after exclusions, plus zero unreadable RGB files after all 4,349 images successfully decode.
3. **Metric conventions.** The English explanation now states the fixed class order, validation support and zero-denominator handling.

These changes concern audit presentation and preview selection. Reloading the existing checkpoint reproduced its recorded validation result.

## Remaining work

- Near-duplicate leakage remains unresolved and must be investigated or reported as a limitation.
- Dependency installation and cloud dataset setup need a fresh Colab/Kaggle execution check. Local execution and a portable dataset-root setting have been verified.
- Complete the mask-supervised model, matched ablations, both class assignments and three seeds per configuration. Current training covers A, seed 42, one learning-rate candidate.
- Add MSP, maximum-logit and negative-energy scores. Select the primary score from baseline validation results and use it for both models.
- Select configurations by familiar-host validation AUOSCR subject to the predeclared known-class tolerance. Calibrate each run's threshold on known validation images and freeze both assignments' choices before test evaluation.
- Complete related-method comparisons, the predeclared protocol, shared-disease-label grouping, open-set plots, error analysis, inference demonstration and disclosures.
- Before training the lesion branch, add a valid-region mask for excluding padding from segmentation loss. Match initial shared weights and augmentation randomness across paired runs.

## Files reviewed

- [Task sheet](AIP304%20CW2%20Task%20Sheet.pdf)
- [Notebook](../Predict.ipynb)
- [Manifest](../artifacts/option2_manifest.json)
- [Checkpoint](../artifacts/baseline_A_seed42_lr0.0003_epochs20.pt)

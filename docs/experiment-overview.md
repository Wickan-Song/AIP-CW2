# Experiment overview and results

AIP304 Coursework 2, Option 2. This overview explains the experiment and summarises the frozen results. See the [full report](../report/main.pdf) for the formal analysis and figures.

## Research question

Can lesion-mask supervision during training help a model reject unseen host–disease combinations at inference, when it receives only an RGB image, while preserving known-disease classification? The primary evaluation considers unseen diseases on familiar hosts. An unfamiliar host may be easier to detect for reasons unrelated to disease, so it is evaluated separately.

The [protocol](../artifacts/formal/protocol.json) required each assignment to meet all three conditions: a mean AUOSCR gain of at least 0.02, improvement in at least two of three paired seeds, and a known-class macro-F1 drop no greater than 0.02.

## Dataset and class assignments

The coursework subset contains 4,349 PlantSeg v7 images across 43 classes. Exact-duplicate exclusions removed 201 images. Review of 436 near-duplicate candidates confirmed 213 pairs and led to 209 further exclusions. The resulting 3,939 images retain their original partitions before assignment-specific class filtering.

For each familiar host, disease names are sorted alphabetically. Assignment A holds out the first disease for validation and the second for test. Assignment B holds out the second for validation and the third for test. Citrus and potato are the unfamiliar-host validation and test controls, respectively.

| Apple disease | Assignment A | Assignment B |
|---|---|---|
| Black rot | Validation unknown | Known |
| Mosaic virus | Test unknown | Validation unknown |
| Rust | Known | Test unknown |
| Scab | Known | Known |

An apple rust image from the original training partition can be used in A. In B, rust is a test unknown, so a training-partition rust image is excluded. Only original test-partition rust images can be used to test rejection in B.

| Assignment | Known training | Validation: known / familiar unknown / citrus | Test: known / familiar unknown / potato |
|---|---:|---:|---:|
| A | 1,576 | 266 / 54 / 39 | 490 / 137 / 26 |
| B | 1,432 | 246 / 64 / 39 | 452 / 132 / 26 |

## Matched training and model selection

Both arms use ImageNet-pretrained ResNet-18 with a 25-class classification head. Input is resized to a 224-pixel long edge, padded, and normalised. During training, images may be flipped or receive light colour augmentation. Masks receive the corresponding geometric transformations, and padded pixels are excluded from the auxiliary loss. The lesion model adds a binary segmentation loss with weight 0.25 during training. Evaluation uses only RGB inputs.

The [pilot notebook](../Predict.ipynb) checked the pipeline using assignment A and seed 42 before the near-duplicate audit. Validation accuracy was 78.78% for the baseline and 81.65% for the lesion arm. Familiar-host validation AUOSCR under MaxLogit was 0.6078 and 0.6274, respectively. These are pilot validation results.

The formal grid has 24 runs: two assignments, two arms, three seeds (42, 123, 2026), and two candidate learning rates (0.0001 and 0.0003). Each run used 20 epochs, batch size 32, and AdamW. Known-validation macro-F1 selected the checkpoint. Baseline validation selected MaxLogit from maximum softmax probability, MaxLogit, and negative energy. Both assignments selected 0.0001. The threshold for each checkpoint targeted at least 95% acceptance of known validation examples. These choices were frozen before testing the 12 selected checkpoints.

## Frozen test results

Means and sample standard deviations across three seeds under the frozen MaxLogit rule:

| Metric | A baseline | A lesion | B baseline | B lesion |
|---|---:|---:|---:|---:|
| Known accuracy | 78.50% ± 1.23% | 78.44% ± 0.42% | 74.71% ± 0.92% | 74.78% ± 0.80% |
| Known macro-F1 | 0.7578 ± 0.0203 | 0.7589 ± 0.0095 | 0.7017 ± 0.0099 | 0.7031 ± 0.0065 |
| Familiar-host AUOSCR | 0.6245 ± 0.0115 | 0.6316 ± 0.0048 | 0.6007 ± 0.0119 | 0.5987 ± 0.0141 |
| Familiar-host unknown AUROC | 0.7333 ± 0.0081 | 0.7425 ± 0.0032 | 0.7367 ± 0.0109 | 0.7345 ± 0.0122 |
| Familiar-host false acceptance | 82.48% ± 4.44% | 82.97% ± 5.28% | 75.25% ± 5.16% | 76.52% ± 4.22% |
| Potato control AUOSCR | 0.6385 ± 0.0231 | 0.6441 ± 0.0248 | 0.6386 ± 0.0069 | 0.6426 ± 0.0074 |
| Known acceptance | 92.52% ± 0.96% | 92.72% ± 0.72% | 91.08% ± 0.34% | 92.18% ± 0.78% |
| Known correct and accepted | 75.65% ± 1.39% | 75.71% ± 0.41% | 71.53% ± 0.51% | 72.05% ± 1.05% |

Familiar-host AUOSCR changed by +0.0071 ± 0.0120 in A (two seeds improved) and −0.0020 ± 0.0064 in B (one improved). Both arms met the known-class tolerance in each assignment. Neither assignment reached the AUOSCR gain target. AUOSCR accounts for known classification and unknown rejection across thresholds. False acceptance measures the fraction of unknowns that passed the frozen threshold.

## Error analysis and limitations

The 12 frozen checkpoints reproduced the validation acceptance counts. In some images, predicted lesion activations aligned with visible symptoms, including soybean frog eye leaf spot. Other examples showed activations outside visible lesions or strong activations for unknown diseases. No reference validation masks were used for these qualitative maps.

In B, the lesion models accepted 24 of 27 seed-level judgments for validation images of northern corn leaf blight, 18 as corn gray leaf spot. The same image may contribute one judgment per seed. The [validation analysis](../artifacts/formal/diagnostics/validation_findings.md) links to the original overlays. At the frozen thresholds, validation familiar-host false acceptance was 136/162 versus 133/162 in A, and 140/192 versus 143/192 in B (baseline versus lesion).

The evidence supports preserving known-class performance under this protocol, while the rejection benefit was inconsistent. The analysis cannot establish which visual cues caused the misclassifications or attribute the result to the backbone or optimiser. The potato control covers only one unfamiliar host.

See the [experiment handoff](experiment-handoff.md), [formal notebook](../Formal_Experiment.ipynb), and [frozen results](../artifacts/formal/evaluation/test_results.json).

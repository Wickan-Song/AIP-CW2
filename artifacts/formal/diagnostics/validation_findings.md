# Validation diagnostics for lesion prediction and rejection

The analysis uses the 12 frozen checkpoints from assignments A and B at the selected learning rate of 0.0001. Each model processes validation RGB images with the original preprocessing and frozen MaxLogit threshold. Acceptance counts match the saved validation grid for every checkpoint and group.

## Lesion predictions

The overlays average the pixelwise sigmoid outputs from the three lesion models for each assignment. They show only the unpadded image region, using the same colour scale. The known examples span the 10th, 40th, 70th and 95th percentiles of image-level activation. No reference masks were used, so this is a visual inspection rather than a measurement of segmentation accuracy.

The maps highlight visible spots in a soybean frog eye leaf spot example and streaks in a wheat stripe rust example. In some tomato leaf mold examples, the response covers parts of the background as well as the leaf. Several unfamiliar diseases also show strong responses on visible symptoms. The median image-level 95th percentile pixel score is 0.695 for known images and 0.683 for familiar-host unknown images in A. It is 0.723 and 0.744 respectively in B. These summaries describe activation strength, not localisation accuracy. The diagnostic maps are outputs of the auxiliary branch. The classifier's MaxLogit rejection rule uses its class logits.

[Known-class lesion overlays](validation_known_overlays.png)

## Familiar-host unknown diseases

Each validation image contributes one decision per seed and method. The decisions for the same image are paired. The counts below are descriptive and the three decisions for an image are not independent observations.

| Assignment | Images | Baseline false accepts | Lesion false accepts | Baseline only | Lesion only | Images accepted by both in all three seeds |
|---|---:|---:|---:|---:|---:|---:|
| A | 54 | 136/162 (84.0%) | 133/162 (82.1%) | 5 | 2 | 37 |
| B | 64 | 140/192 (72.9%) | 143/192 (74.5%) | 5 | 8 | 32 |

In assignment B, northern corn leaf blight had 24 false accepts across 27 decisions by the lesion models. Eighteen were predicted as corn gray leaf spot. The example overlay highlights elongated marks on the leaf, and all six models accept that image. This is consistent with confusion between visually similar lesions on a familiar host. Other examples cross host boundaries, such as apple black rot assigned to tomato late blight. These observations suggest possible cues and do not establish which features caused the class predictions.

[Recurring false accepts with lesion overlays](validation_unknown_overlays.png)

The validation evidence does not point to a failed classifier backbone or optimiser. It shows some symptom localisation and repeated acceptance of familiar-host unknowns under the frozen rule. The observed changes in false acceptance differ by assignment. Keep the formal comparison as the coursework result. A later exploratory experiment could test one change to the lesion loss weight with matched conditions on validation, if there is a specific follow-up question about that weight.

Reproduce from the project root with the project Python environment using `python artifacts/formal/diagnostics/inspect_validation.py`. The per-image scores, accepted decisions, and group counts are in [validation_diagnostics.json](validation_diagnostics.json).

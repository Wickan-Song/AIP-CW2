# AIP304 CW2 experiment handoff

This English handoff records the pilot, formal experiment, and verification history for Coursework 2, Option 2. The [experiment overview](experiment-overview.md) explains the A/B assignment with an apple example. See the [root README](../README.md) for setup.

## Entry points

- [Predict.ipynb](../Predict.ipynb) records the pilot. [Formal_Experiment.ipynb](../Formal_Experiment.ipynb) contains the formal comparison and saved outputs.
- [Formal manifest](../artifacts/formal/manifest.json), [pilot manifest](../artifacts/option2_manifest.json), and [formal protocol](../artifacts/formal/protocol.json) define roles, exclusions, and selection rules.
- [The report](../report/main.pdf) is the current five-page English report. The [task sheet](<AIP304 CW2 Task Sheet.pdf>) defines the submission requirements.
- The original PlantSeg v7 archive MD5 is `9358a66dff88cdd15c4fe009763c40a3`. Use the release to restore the data and checkpoints. `PLANTSEG_ROOT` can point the experiment code to the extracted dataset if it is stored elsewhere.
- The local run used Apple Silicon MPS, PyTorch 2.8.0, torchvision 0.23.0, Pillow 11.3.0, NumPy 2.0.2, and Matplotlib 3.9.4. Select the project environment as the notebook kernel and open the notebook from the repository root.

## Pilot and exact duplicates

The initial coursework subset contained 4,349 images and 43 classes across nine hosts. Metadata connects host, disease, class ID, source filename, and partition to images and masks. After EXIF orientation correction, the exact-duplicate audit used dimensions and an RGB pixel hash. It found 73 cross-partition duplicate groups, including 34 with conflicting labels. Conflicting groups were removed. Otherwise one copy was kept, preferring test, then validation, then train, with filenames breaking ties. Removing 201 images left 4,148 for the pilot and later near-duplicate screening. The pilot manifest SHA-256 was `cc9a775edf59da9b52e2c3a036b7fb0045dd0f6b5d33fcbf2be125b1a760b594`.

| Pilot assignment | Known training | Validation: known / familiar unknown / citrus | Test: known / familiar unknown / potato |
|---|---:|---:|---:|
| A | 1,666 | 278 / 54 / 42 | 495 / 145 / 27 |
| B | 1,516 | 257 / 70 / 42 | 460 / 132 / 27 |

The A/42 baseline used 20 epochs, batch size 32, AdamW, an initial learning rate of 0.0003, weight decay of 0.0001, and cosine decay to 0.000001. On 278 known validation images, epoch 1 achieved macro-F1 0.6114 and accuracy 66.91% in 31.2 seconds. Checkpoint selection by macro-F1 chose epoch 15, achieving macro-F1 0.7661 and accuracy 78.78%. The 20 epochs took about 8.4 minutes on MPS. The checkpoint reproduced the validation scores when reloaded.

The [initial review](CW2_part1_review.md) independently verified 219/278 correct decisions, the manifest, and the class roles. The data preview was updated to select a known training image for the active assignment, since apple rust is unknown in B. The 1,980 images in the union of the two usable training sets retained foreground and correctly aligned mask dimensions after orientation correction and resizing.

## Pilot lesion comparison

The matched A/42 lesion intervention added a binary head at ResNet-18 `layer2`. Its loss is the mean of binary cross-entropy and soft Dice multiplied by 0.25, added to the classification cross-entropy. Images and masks receive matched geometry. Padding does not contribute to the lesion loss. The model receives RGB alone at evaluation. The review confirmed matched initial classifier weights, early-batch augmentation and order, shared-feature gradients from the auxiliary loss, and zero auxiliary gradient on padding.

| Pilot validation result | Baseline | Lesion | Change |
|---|---:|---:|---:|
| Chosen epoch | 15 | 17 | |
| Known macro-F1 | 0.7661 | 0.7825 | +0.0164 |
| Known accuracy | 78.78% | 81.65% | |
| Familiar-host AUOSCR, MSP | 0.5937 | 0.6091 | +0.0154 |
| Familiar-host AUOSCR, MaxLogit | 0.6078 | 0.6274 | +0.0197 |
| Familiar-host AUOSCR, negative energy | 0.6099 | 0.6316 | +0.0216 |

At approximately 95.32% known validation acceptance with negative energy, the familiar-host false accepts changed from 47/54 (87.04%) to 43/54 (79.63%) and unknown AUROC changed from 0.7088 to 0.7205. The [pilot review](CW2_lesion_pilot_review.md) reran inference on 374 validation images and independently recomputed the scores, including ties and cases where all known predictions were wrong. It did not retrain or use test images. These validation results from one assignment and seed served as development evidence.

## Near duplicates and formal training

The [near-duplicate audit](../artifacts/near_duplicate_audit/README.md) combined luminance correlation, hashes, crop and mirror variants, and URL matches. Manual review confirmed 213 of 436 candidate pairs. There were 145 connected groups, 12 with label conflicts. The audit excluded 209 images (167 train, 23 validation, 19 test), leaving 3,939. Heavy editing can evade screening. The original partitions and A/B class roles were retained. The formal manifest SHA-256 is `bd98c23f968b2b4f6e572a32226fcad20e42639deb6262ddc115a3c0b149be1e`.

The formal protocol was recorded after the pilot and before formal training. The matched grid used A and B, seeds 42, 123, and 2026, two learning rates (0.0001 and 0.0003), and two arms. All 24 runs started from ImageNet weights, used 20 epochs, and saved resume states every five epochs. Checkpoints were selected by known-validation macro-F1. The run script resumes training and then performs validation, freezes the options, and evaluates the test set. See the [training log](../artifacts/formal/experiment.log). The first formal A/42 baseline at 0.0001 selected epoch 19 and reproduced 81.95% accuracy and 0.8029 macro-F1 on 266 known validation images when reloaded.

Baseline validation selected MaxLogit as the primary score, ahead of maximum softmax probability and negative energy. Both arms shared the chosen rate of 0.0001 within each assignment. Each model threshold was calibrated on known validation examples before test inference. Views with other scores remain in the formal results.

## Frozen evaluation and diagnostics

| Assignment | Familiar-host AUOSCR, baseline → lesion | Paired mean change | Improved seeds | Known macro-F1, baseline → lesion |
|---|---:|---:|---:|---:|
| A | 0.6245 → 0.6316 | +0.0071 | 2/3 | 0.7578 → 0.7589 |
| B | 0.6007 → 0.5987 | −0.0020 | 1/3 | 0.7017 → 0.7031 |

Neither assignment reached the predefined +0.02 AUOSCR gain. Both met the known-class macro-F1 tolerance. Under the frozen thresholds, familiar-host unknown false acceptance was 82.48% versus 82.97% in A and 75.25% versus 76.52% in B (baseline versus lesion). All 24 formal checkpoint hashes, the frozen settings, and the test manifest were checked in the formal review. Known accuracy and macro-F1, open-set scores for both unknown groups, and threshold counts were independently recomputed from the 12 selected models' saved predictions. See the [frozen test results](../artifacts/formal/evaluation/test_results.json).

The [validation diagnostics](../artifacts/formal/diagnostics/validation_findings.md) reproduce the acceptance counts from the 12 frozen checkpoints. The maps average auxiliary pixel scores across three seeds and exclude padding. They are qualitative because the analysis did not use reference segmentation masks. Some maps highlight visible soybean spots or wheat rust, while others respond to the background or unseen diseases. In B, 24 of 27 lesion-model judgments for northern corn leaf blight were false accepts, including 18 predicted as gray leaf spot. Similar-looking symptoms are one possible explanation. The maps cannot establish which features drove classification.

## Figures, report, and submission state

- [Report figure generation](../report_figures.py) and [notebook figure generation](../formal_figures.py) use the [figure audit](../figure_audit.py) to check bounds, font sizes, ticks, and legends. Report and notebook readability floors are 6 pt and 7 pt, respectively. Geometry and label layout were updated while data and experiments were retained. The report figure directory includes final-size previews at 96 and 150 dpi and [provenance](../artifacts/formal/figures/report/provenance.json).
- The [five-page report project](../report/README.md) uses a compact host-level confusion view and the top-six false-acceptance destinations. The notebook contains the full class-level matrices and prediction gallery. The partition counts are in the report text and the notebook.
- The task sheet asks for a notebook runnable in a fresh Colab or Kaggle environment, a report of up to five pages, and a video of up to three minutes with the presenter visible and a working inference demonstration. A fresh cloud run remains unverified here. The local CPU inference command is `python verify_package.py --demo` after asset restoration and dependency installation.
- For handoff, keep the original dataset archive, the pilot and formal checkpoints, the evaluation artifacts, the recorded protocol, and the dependency file available. Keep exploratory experiments separate from the frozen results.

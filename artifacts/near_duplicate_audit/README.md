# Cross-partition near-duplicate audit

The audit screened the 4,148 images remaining after exact RGB deduplication on the nine PlantSeg v7 hosts used in the coursework. It covered train–validation, train–test and validation–test pairs. The screen used the EXIF-corrected photos and original partition metadata. It used no annotations, classifier outputs or performance metrics.

Each image was reduced to a 32 × 32 luminance grid for normalized correlation and a 63-bit low-frequency DCT perceptual hash. A 64-bit horizontal difference hash provided a second signature. Candidate generation included horizontal mirroring, along with 90%, 80% and 65% crop views anchored at nine positions. A shared exact source URL and luminance correlation of at least 0.65 provided an additional signal. The expanded screen accepted candidate matches with full-image correlation of at least 0.87, crop correlation of at least 0.86, perceptual-hash distance of at most 14 bits, or difference-hash distance of at most 10 bits. These thresholds generated a review queue. They did not decide which images to exclude.

The reviewers confirmed 213 of the 436 candidate pairs by checking for distinctive matching lesion, vein, plant, background and overlay details. The 145 connected components include 12 with conflicting class labels. All members of those 12 components were excluded. Consistent-label components kept one member in its original partition using test, then validation, then training priority. Filename order broke ties. The decisions exclude 209 images, leaving 3,939 original images before A/B role filtering. Every assignment, partition and class combination with support in the pilot retains support.

| Assignment | Training known | Validation known / familiar unknown / citrus | Test known / familiar unknown / potato |
|---|---:|---:|---:|
| A | 1,576 | 266 / 54 / 39 | 490 / 137 / 26 |
| B | 1,432 | 246 / 64 / 39 | 452 / 132 / 26 |

The screen can miss stronger transformations such as severe crops, rotation, or a photograph embedded in a collage. Same-partition near duplicates were outside the screening scope. Components can contain same-partition members when those members connect through a confirmed cross-partition match.

From the project root, reproduce the candidate list and apply the recorded manual review.

```bash
.venv/bin/python artifacts/near_duplicate_audit/screen.py --first-pass artifacts/near_duplicate_audit/first_pass_candidates.json
.venv/bin/python artifacts/near_duplicate_audit/finalize.py
```

The first-pass candidate file preserves pair IDs used during the visual review. The candidate file and contact sheets contain the visual review queue. The manual review references the exact candidate file by SHA-256. The decisions file lists each confirmed component and the exact dataset-relative paths to exclude. The earlier manifest and checkpoints document the pilot.

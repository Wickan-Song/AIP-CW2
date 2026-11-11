"""Independent audit 05: static code-conformity checks on the pipeline source.

Verifies, by reading the shipped source, the leakage and protocol properties
the task sheet demands: RGB-only evaluation, masks confined to known-class
training, generic pretrained weights, deterministic evaluation, documented
zero-denominator rule, pre-registered constants and paired seeding.

Run from the repository root:  python3 review/05_code_conformity_checks.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

findings = []
def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append(f"{label}: {detail}")

def segment(source, start, end):
    i = source.index(start)
    j = source.index(end, i)
    return source[i:j]

def main():
    train_src = (ROOT / "formal_training.py").read_text()
    eval_src = (ROOT / "formal_evaluation.py").read_text()
    manifest_src = (ROOT / "formal_manifest.py").read_text()

    # --- pretrained weights / no PlantSeg overlap ---------------------------
    check("training uses torchvision ImageNet ResNet-18 weights identifier",
          "WEIGHTS = ResNet18_Weights.IMAGENET1K_V1" in train_src)
    # "PlantSeg" legitimately names the dataset root/manifest; a PlantSeg-trained
    # checkpoint would have to be loaded via torch.load/load_state_dict.
    weight_loads = re.findall(r"torch\.load\([^)]*\)|load_state_dict\([^)]*\)",
                              train_src + eval_src)
    check("no weight loading from PlantSeg-named sources; 'PlantSeg' only labels the dataset",
          all("plantseg" not in w.lower() for w in weight_loads)
          and train_src.count("ResNet18_Weights.") == 1)

    # --- masks confined to known-class training -----------------------------
    rgb_getitem = segment(train_src, "class RGBDataset(Dataset):", "class LesionTrainingDataset")
    check("RGBDataset never reads mask paths", "mask" not in rgb_getitem)
    lesion_cls = segment(train_src, "class LesionTrainingDataset", "class LesionSupervisedResNet18")
    check("LesionTrainingDataset refuses non-train/non-known rows",
          'raise ValueError("Lesion supervision requires known-class training rows.")' in lesion_cls)
    check("mask mirrors with the image (synchronous augmentation)",
          "ImageOps.mirror(image)" in train_src
          and train_src.index("ImageOps.mirror(image)") < train_src.index("ImageOps.mirror(mask)"))
    check("padding region excluded from lesion loss (valid_region)",
          "valid_region" in train_src and "(pixel_loss * valid_region)" in train_src)
    check("mask binarised as non-zero foreground, nearest-neighbour resize",
          "255 if value > 0 else 0" in train_src and "Resampling.NEAREST" in train_src)
    check("evaluation source never reads lesion masks/annotations (RGB-only paths)",
          "mask_path" not in eval_src and "annotations" not in eval_src
          and ".png" not in eval_src.lower())

    # --- deterministic evaluation -------------------------------------------
    check("validation uses inference mode and eval()",
          "@torch.inference_mode()" in train_src and "model.eval()" in train_src)
    check("validation loader built with training=False (deterministic transform)",
          re.search(r"RGBDataset\(val_rows, data_root, cache=cache\)", train_src) is not None)
    check("EXIF orientation applied before resize (no orientation label leakage)",
          "ImageOps.exif_transpose" in train_src)
    check("normalisation uses the pretrained weights' documented statistics",
          "WEIGHTS.transforms().mean" in train_src and "WEIGHTS.transforms().std" in train_src)

    # --- protocol constants in evaluation ------------------------------------
    for constant, value in (("KNOWN_ACCEPTANCE", "0.95"), ("KNOWN_F1_TOLERANCE", "0.02"),
                            ("AUOSCR_TARGET", "0.02"), ("MIN_POSITIVE_SEEDS", "2")):
        check(f"evaluation constant {constant} = {value} matches protocol",
              re.search(rf"{constant}\s*=\s*{value}", eval_src) is not None)
    check("score order MSP, MaxLogit, Negative energy (documented tie-break)",
              'SCORE_NAMES = ("MSP", "MaxLogit", "Negative energy")' in eval_src
              or re.search(r'SCORE_NAMES\s*=\s*\[\s*"MSP",\s*"MaxLogit",\s*"Negative energy"\s*\]', eval_src))
    check("test evaluation refuses to run without frozen thresholds",
          re.search(r'split == "test" and thresholds is None', eval_src) is not None)
    check("acceptance rule is score >= threshold",
          ">= threshold" in eval_src)
    check("calibration takes the ceil(0.95 n)-th largest known score",
              "-math.ceil(KNOWN_ACCEPTANCE * len(known_scores))" in eval_src)

    # --- paired training / checkpoint rule -----------------------------------
    check("per-run seeding of python, torch and loader generator",
          "random.seed(spec.seed)" in train_src and "torch.manual_seed(spec.seed)" in train_src
          and "torch.Generator().manual_seed(spec.seed)" in train_src)
    check("lesion head initialisation forked off the paired RNG stream",
          "torch.random.fork_rng" in train_src)
    check("checkpoint rule keeps the earliest best epoch (strictly greater)",
          "val_f1 > best_result" in train_src)
    check("training loop pairs identical loss weighting (0.25 lesion loss)",
          "LESION_WEIGHT = 0.25" in train_src and "LESION_WEIGHT * auxiliary" in train_src)
    check("resume path restores RNG, optimizer and scheduler state",
          "restore_rng(" in train_src and "optimizer.load_state_dict" in train_src
          and "scheduler.load_state_dict" in train_src)
    check("nonfinite losses abort the run",
          "FloatingPointError" in train_src)
    check("checkpoints pin manifest sha and class order (anti-mixing guard)",
          "manifest_sha256" in train_src and "known_class_order" in train_src)

    # --- manifest generation --------------------------------------------------
    check("manifest generator refuses metadata/class mismatch",
          "Metadata class mismatch" in manifest_src)
    check("manifest generator verifies near-duplicate decision hashes",
          "decisions" in manifest_src and "sha256" in manifest_src.lower())

    # --- notebook wiring -------------------------------------------------------
    formal_nb = (ROOT / "Formal_Experiment.ipynb").read_text()
    check("formal notebook drives the same module functions (no divergent copy)",
          "formal_training" in formal_nb and "formal_evaluation" in formal_nb)

    print(f"\n{len(findings)} finding(s).")
    sys.exit(1 if findings else 0)

if __name__ == "__main__":
    main()

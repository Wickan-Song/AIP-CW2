"""Validate matched runs and evaluate frozen choices for AIP304 CW2 Option 2."""

import argparse
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from formal_training import (
    RGBDataset,
    RunSpec,
    build_model,
    checkpoint_path,
    load_manifest,
    read_checkpoint,
    resolve_data_root,
)

ASSIGNMENTS = ("A", "B")
SEEDS = (42, 123, 2026)
LEARNING_RATES = (1e-4, 3e-4)
MODELS = ("baseline", "lesion")
SCORE_NAMES = ("MSP", "MaxLogit", "Negative energy")
KNOWN_ACCEPTANCE = 0.95
KNOWN_F1_TOLERANCE = 0.02
AUOSCR_TARGET = 0.02
MIN_POSITIVE_SEEDS = 2

DISEASE_NAME_RULE = (
    "Use the PlantSeg metadata class names and plant names recorded in the manifest. "
    "Apply Unicode NFKC, case folding and whitespace normalization, then remove the "
    "leading host name. Compare exact remaining names across different hosts. "
    "No synonyms or disease families are merged."
)


def _sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    """Publish a complete artifact even if evaluation is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8")
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(device):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _specs():
    return [
        RunSpec(assignment, seed, lr, model)
        for assignment in ASSIGNMENTS
        for lr in LEARNING_RATES
        for seed in SEEDS
        for model in MODELS
    ]


def _run_key(checkpoint_dir, spec):
    return checkpoint_path(checkpoint_dir, spec).stem


def _provenance(manifest_hash, protocol_path):
    protocol_hash = None
    if protocol_path is not None:
        protocol_bytes = Path(protocol_path).read_bytes()
        protocol = json.loads(protocol_bytes)
        expected_grid = {
            "assignments": list(ASSIGNMENTS),
            "seeds": list(SEEDS),
            "learning_rates": list(LEARNING_RATES),
            "models": list(MODELS),
        }
        expected_hypothesis = {
            "minimum_mean_auoscr_gain": AUOSCR_TARGET,
            "minimum_positive_pairs": MIN_POSITIVE_SEEDS,
            "maximum_mean_known_macro_f1_drop": KNOWN_F1_TOLERANCE,
        }
        if any(protocol.get(k) != v for k, v in expected_grid.items()) or any(
            protocol.get("hypothesis", {}).get(k) != v for k, v in expected_hypothesis.items()
        ):
            raise ValueError("Protocol settings differ from the implemented formal evaluation.")
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    return {
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
    }


def _verify_provenance(payload, expected, path):
    if payload.get("provenance") != expected:
        raise ValueError(f"{path} was generated with a different manifest or protocol. Use its original inputs or a separate output directory.")


def _normalize_name(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _disease_name(row):
    name = _normalize_name(row["class_name"])
    prefix = _normalize_name(row["host"]) + " "
    if not name.startswith(prefix):
        raise ValueError(f"Metadata class name {row['class_name']!r} has no host prefix.")
    return name[len(prefix):]


def _known_disease_hosts(class_roles):
    hosts = {}
    for row in class_roles:
        if row["role"] == "known":
            hosts.setdefault(_disease_name(row), set()).add(_normalize_name(row["host"]))
    return hosts


def _disease_groups(rows, class_roles, split):
    familiar_role = "familiar_dev" if split == "val" else "familiar_test"
    unfamiliar_role = "unfamiliar_dev" if split == "val" else "unfamiliar_test"
    roles = np.asarray([row["role"] for row in rows])
    known_hosts = _known_disease_hosts(class_roles)
    shared = np.asarray([
        row["role"] == familiar_role and bool(
            known_hosts.get(_disease_name(row), set()) - {_normalize_name(row["host"])}
        )
        for row in rows
    ])
    return {
        "familiar_host": roles == familiar_role,
        "unfamiliar_host": roles == unfamiliar_role,
        "familiar_shared_name": shared,
        "familiar_unshared_name": (roles == familiar_role) & ~shared,
    }


def _familiar_disease_matches(class_roles, split):
    familiar_role = "familiar_dev" if split == "val" else "familiar_test"
    known_hosts = _known_disease_hosts(class_roles)
    return {
        row["class_name"]: {
            "normalized_disease_name": _disease_name(row),
            "other_known_hosts": sorted(
                known_hosts.get(_disease_name(row), set()) - {_normalize_name(row["host"])}
            ),
        }
        for row in class_roles
        if row["role"] == familiar_role
    }


def _areas(known_scores, known_correct, unknown_scores):
    """Group tied scores before computing OSCR and unknown-positive AUROC."""
    n_known, n_unknown = len(known_scores), len(unknown_scores)
    values = np.concatenate((known_scores, unknown_scores))
    known = np.arange(len(values)) < n_known
    correct = np.concatenate((known_correct, np.zeros(n_unknown, dtype=bool)))
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    endpoints = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    false_accept = np.r_[0, (~known[order]).cumsum()[endpoints]] / n_unknown
    known_accept = np.r_[0, known[order].cumsum()[endpoints]] / n_known
    correct_accept = np.r_[0, correct[order].cumsum()[endpoints]] / n_known
    widths = np.diff(false_accept)
    auoscr = widths @ ((correct_accept[:-1] + correct_accept[1:]) / 2)
    auroc = widths @ ((known_accept[:-1] + known_accept[1:]) / 2)
    return float(auoscr), float(auroc), {
        "unknown_false_acceptance": false_accept.tolist(),
        "known_correct_acceptance": correct_accept.tolist(),
    }


def _score_vectors(logits):
    return {
        "MSP": logits.softmax(dim=1).amax(dim=1),
        "MaxLogit": logits.amax(dim=1),
        "Negative energy": logits.logsumexp(dim=1),
    }


def measure_logits(logits, targets, groups, *, thresholds=None, keep_curves=False):
    """Score fixed RGB predictions with known targets and unknown-group masks.

    Use known-only validation calibration when thresholds are absent. The test
    workflow supplies all three frozen thresholds explicitly.
    """
    logits = torch.as_tensor(logits).detach().cpu()
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.ndim != 1 or len(targets) != len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit vector per evaluation row.")
    known = targets >= 0
    if not known.any() or np.any(targets[known] >= logits.shape[1]):
        raise ValueError("Known evaluation targets do not match the classifier.")
    for mask in groups.values():
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != known.shape or np.any(mask & known):
            raise ValueError("Unknown groups must use Boolean masks aligned with rows and exclude known targets.")
    if thresholds is not None and (
        set(thresholds) != set(SCORE_NAMES) or not all(math.isfinite(v) for v in thresholds.values())
    ):
        raise ValueError("Evaluation requires one finite frozen threshold per scoring rule.")

    predictions = logits.argmax(dim=1).numpy()
    known_correct = predictions[known] == targets[known]
    n_classes = logits.shape[1]
    true_support = np.bincount(targets[known], minlength=n_classes)
    pred_support = np.bincount(predictions[known], minlength=n_classes)
    true_positives = np.bincount(targets[known][known_correct], minlength=n_classes)
    denominator = true_support + pred_support
    class_f1 = np.divide(2 * true_positives, denominator, out=np.zeros(n_classes), where=denominator > 0)
    result = {
        "known_count": int(known.sum()),
        "known_correct_count": int(known_correct.sum()),
        "known_accuracy": float(known_correct.mean()),
        "known_macro_f1": float(class_f1.mean()),
        "scores": {},
    }
    for name, values in _score_vectors(logits).items():
        vector = values.numpy()
        known_scores = vector[known]
        threshold = (
            float(np.sort(known_scores)[-math.ceil(KNOWN_ACCEPTANCE * len(known_scores))])
            if thresholds is None else float(thresholds[name])
        )
        accepted_known = known_scores >= threshold
        per_group = {}
        for group, mask in groups.items():
            unknown_scores = vector[mask]
            if not len(unknown_scores):
                per_group[group] = {
                    "unknown_count": 0,
                    "auoscr": None,
                    "unknown_auroc": None,
                    "unknown_false_acceptance": None,
                    "unknown_accepted_count": 0,
                }
                continue
            auoscr, auroc, oscr = _areas(known_scores, known_correct, unknown_scores)
            accepted = unknown_scores >= threshold
            per_group[group] = {
                "unknown_count": len(unknown_scores),
                "auoscr": auoscr,
                "unknown_auroc": auroc,
                "unknown_false_acceptance": float(accepted.mean()),
                "unknown_accepted_count": int(accepted.sum()),
            }
            if keep_curves:
                per_group[group]["oscr_curve"] = oscr
        result["scores"][name] = {
            "threshold": threshold,
            "known_acceptance": float(accepted_known.mean()),
            "known_correct_acceptance": float((accepted_known & known_correct).mean()),
            "known_accepted_count": int(accepted_known.sum()),
            "groups": per_group,
        }
    return result


def _evaluate_run(spec, manifest, manifest_hash, data_root, checkpoint, device, split, *, thresholds=None):
    if split == "test" and thresholds is None:
        raise ValueError("Test evaluation requires frozen thresholds.")
    assignment = spec.assignment
    class_order = manifest["known_class_order"][assignment]
    rows = [row for row in manifest["samples"][assignment] if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest has no {split} rows for assignment {assignment}.")
    saved = read_checkpoint(checkpoint, spec, manifest_hash, class_order)
    model = build_model(len(class_order), spec.model, pretrained=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(RGBDataset(rows, data_root), batch_size=32, shuffle=False, num_workers=0)
    logits = []
    targets = []
    with torch.inference_mode():
        for images, labels in loader:
            logits.append(model(images.to(device)).cpu())
            targets.append(labels.cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets).numpy()
    groups = _disease_groups(rows, manifest["class_roles"][assignment], split)
    result = measure_logits(logits, targets, groups, thresholds=thresholds, keep_curves=split == "test")
    return result, rows, logits, saved["selected_epoch"]


def validate_grid(manifest_path, data_root, checkpoint_dir, out_dir, *, device=None, protocol_path=None, require_complete=True):
    """Score completed checkpoints on validation RGB images and save their calibration."""
    manifest, manifest_hash = load_manifest(manifest_path)
    provenance = _provenance(manifest_hash, protocol_path)
    data_root = resolve_data_root(data_root)
    device = _device(device)
    path = Path(out_dir) / "validation_grid.json"
    if path.exists():
        payload = _read_json(path)
        _verify_provenance(payload, provenance, path)
    else:
        payload = {
            "provenance": provenance,
            "split": "val",
            "disease_name_rule": DISEASE_NAME_RULE,
            "familiar_disease_matches": {
                assignment: _familiar_disease_matches(manifest["class_roles"][assignment], "val")
                for assignment in ASSIGNMENTS
            },
            "runs": {},
        }
    missing = []
    for spec in _specs():
        checkpoint = checkpoint_path(checkpoint_dir, spec)
        key = checkpoint.stem
        if not checkpoint.is_file():
            missing.append(key)
            payload["runs"].pop(key, None)
            continue
        digest = _sha256(checkpoint)
        previous = payload["runs"].get(key)
        if previous is not None and previous["checkpoint_sha256"] == digest:
            continue
        result, _, _, epoch = _evaluate_run(spec, manifest, manifest_hash, data_root, checkpoint, device, "val")
        if _sha256(checkpoint) != digest:
            raise RuntimeError(f"{checkpoint} changed during validation. Run validation again.")
        payload["runs"][key] = {
            "checkpoint_sha256": digest,
            "evaluation_device": str(device),
            "selected_epoch": epoch,
            "metrics": result,
        }
        payload["missing_runs"] = [k for k in (_run_key(checkpoint_dir, s) for s in _specs()) if k not in payload["runs"]]
        _write_json(path, payload)
        print(f"Validated {key}", flush=True)
    payload["missing_runs"] = missing
    _write_json(path, payload)
    if missing and require_complete:
        raise ValueError(f"{len(missing)} of {len(_specs())} checkpoints are missing. See {path}.")
    return payload


def _at_least(value, cutoff):
    return value >= cutoff or math.isclose(value, cutoff, abs_tol=1e-12, rel_tol=0)


def _choices(validation, checkpoint_dir):
    records = validation["runs"]

    def record(assignment, seed, lr, model):
        return records[_run_key(checkpoint_dir, RunSpec(assignment, seed, lr, model))]

    def metrics(assignment, seed, lr, model):
        return record(assignment, seed, lr, model)["metrics"]

    def area(result, score):
        return result["scores"][score]["groups"]["familiar_host"]["auoscr"]

    baseline_means = {
        score: float(np.mean([
            area(metrics(assignment, seed, lr, "baseline"), score)
            for assignment in ASSIGNMENTS for seed in SEEDS for lr in LEARNING_RATES
        ]))
        for score in SCORE_NAMES
    }
    primary = max(SCORE_NAMES, key=baseline_means.get)
    assignments = {}
    for assignment in ASSIGNMENTS:
        candidates = []
        for lr in LEARNING_RATES:
            baseline = [metrics(assignment, seed, lr, "baseline") for seed in SEEDS]
            lesion = [metrics(assignment, seed, lr, "lesion") for seed in SEEDS]
            known_difference = float(np.mean([
                l["known_macro_f1"] - b["known_macro_f1"] for b, l in zip(baseline, lesion)
            ]))
            baseline_mean = float(np.mean([area(r, primary) for r in baseline]))
            lesion_mean = float(np.mean([area(r, primary) for r in lesion]))
            candidates.append({
                "learning_rate": lr,
                "mean_paired_known_macro_f1_difference": known_difference,
                "baseline_mean_familiar_host_auoscr": baseline_mean,
                "lesion_mean_familiar_host_auoscr": lesion_mean,
                "paired_mean_familiar_host_auoscr": (baseline_mean + lesion_mean) / 2,
                "within_known_macro_f1_tolerance": _at_least(known_difference, -KNOWN_F1_TOLERANCE),
            })
        eligible = [c for c in candidates if c["within_known_macro_f1_tolerance"]]
        if eligible:
            selected = max(eligible, key=lambda c: (c["paired_mean_familiar_host_auoscr"], -c["learning_rate"]))
        else:
            selected = max(candidates, key=lambda c: (
                c["mean_paired_known_macro_f1_difference"],
                c["paired_mean_familiar_host_auoscr"],
                -c["learning_rate"],
            ))
        lr = selected["learning_rate"]
        runs = {}
        for seed in SEEDS:
            for model in MODELS:
                saved = record(assignment, seed, lr, model)
                key = _run_key(checkpoint_dir, RunSpec(assignment, seed, lr, model))
                runs[key] = {
                    "seed": seed,
                    "model": model,
                    "checkpoint_sha256": saved["checkpoint_sha256"],
                    "thresholds": {score: saved["metrics"]["scores"][score]["threshold"] for score in SCORE_NAMES},
                }
        assignments[assignment] = {
            "selected_learning_rate": lr,
            "validation_tolerance_satisfied": bool(eligible),
            "validation_candidates": candidates,
            "runs": runs,
        }
    return {
        "primary_score": primary,
        "baseline_grid_mean_familiar_host_auoscr": baseline_means,
        "known_macro_f1_tolerance": KNOWN_F1_TOLERANCE,
        "known_calibration_target": KNOWN_ACCEPTANCE,
        "selection_rule": (
            "Choose one score from baseline familiar-host validation AUOSCR averaged across "
            "assignments, seeds and both learning rates. Break score ties in the order "
            "MSP, MaxLogit, Negative energy. For each assignment, among learning rates whose "
            "mean paired known macro-F1 difference is at least -0.02, select the learning rate "
            "that maximizes the equal-weight mean of both models' familiar-host validation AUOSCR. "
            "Break ties by smaller learning rate. If no rate is eligible, select the best known "
            "macro-F1 difference, then the best equal-weight mean AUOSCR, then the smaller rate, "
            "and report that the tolerance was missed during selection."
        ),
        "assignments": assignments,
    }


def freeze_choices(manifest_path, checkpoint_dir, out_dir, *, protocol_path=None):
    """Freeze the primary score, shared learning rates and validation-only thresholds."""
    _, manifest_hash = load_manifest(manifest_path)
    provenance = _provenance(manifest_hash, protocol_path)
    out_dir = Path(out_dir)
    validation_path = out_dir / "validation_grid.json"
    validation = _read_json(validation_path)
    _verify_provenance(validation, provenance, validation_path)
    if validation.get("split") != "val":
        raise ValueError("Configuration selection requires validation records.")
    expected = {_run_key(checkpoint_dir, spec) for spec in _specs()}
    if set(validation["runs"]) != expected or validation.get("missing_runs"):
        raise ValueError("Validate all 24 matched runs before freezing choices.")
    for spec in _specs():
        checkpoint = checkpoint_path(checkpoint_dir, spec)
        if _sha256(checkpoint) != validation["runs"][checkpoint.stem]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after validation: {checkpoint}")
    payload = {
        "provenance": provenance,
        "validation_grid_sha256": _sha256(validation_path),
        **_choices(validation, checkpoint_dir),
    }
    path = out_dir / "frozen_choices.json"
    if path.exists():
        existing = _read_json(path)
        if {k: v for k, v in existing.items() if k != "frozen_at_utc"} != payload:
            raise ValueError(f"{path} already contains different frozen choices.")
        return existing
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, payload)
    return payload


def _estimate(values):
    if not values or any(v is None for v in values):
        return {"mean": None, "sd": None}
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def _paired_summary(runs, score):
    def extract(result):
        metric = result["scores"][score]
        values = {
            "known_macro_f1": result["known_macro_f1"],
            "known_accuracy": result["known_accuracy"],
            "known_acceptance": metric["known_acceptance"],
            "known_correct_acceptance": metric["known_correct_acceptance"],
        }
        for group, group_result in metric["groups"].items():
            for name in ("auoscr", "unknown_auroc", "unknown_false_acceptance"):
                values[f"{group}/{name}"] = group_result[name]
        return values

    baseline = {run["seed"]: extract(run["metrics"]) for run in runs if run["model"] == "baseline"}
    lesion = {run["seed"]: extract(run["metrics"]) for run in runs if run["model"] == "lesion"}
    summary = {}
    for name in baseline[SEEDS[0]]:
        a = [baseline[seed][name] for seed in SEEDS]
        b = [lesion[seed][name] for seed in SEEDS]
        differences = [y - x if x is not None and y is not None else None for x, y in zip(a, b)]
        summary[name] = {
            "baseline": _estimate(a),
            "lesion": _estimate(b),
            "lesion_minus_baseline": _estimate(differences),
            "paired_seed_differences": dict(zip(map(str, SEEDS), differences)),
        }
    return summary


def _hypothesis(summary):
    area = summary["familiar_host/auoscr"]
    difference = area["lesion_minus_baseline"]["mean"]
    positives = sum(v > 0 for v in area["paired_seed_differences"].values())
    f1 = summary["known_macro_f1"]["lesion_minus_baseline"]["mean"]
    criteria = {
        "mean_auoscr_difference_at_least_0.02": _at_least(difference, AUOSCR_TARGET),
        "at_least_two_positive_seed_differences": positives >= MIN_POSITIVE_SEEDS,
        "mean_known_macro_f1_difference_at_least_minus_0.02": _at_least(f1, -KNOWN_F1_TOLERANCE),
    }
    return {
        "mean_familiar_host_auoscr_difference": difference,
        "positive_seed_differences": positives,
        "mean_known_macro_f1_difference": f1,
        "criteria": criteria,
        "meets_all_criteria": all(criteria.values()),
    }


def _save_predictions(path, rows, logits, thresholds):
    scores = {name: values.tolist() for name, values in _score_vectors(logits).items()}
    predictions = logits.argmax(dim=1).tolist()
    items = []
    for i, row in enumerate(rows):
        items.append({
            "image_path": row["image_path"],
            "class_name": row["class_name"],
            "role": row["role"],
            "target": row["target"],
            "predicted_known_index": predictions[i],
            "scores": {score: values[i] for score, values in scores.items()},
            "accepted": {score: values[i] >= thresholds[score] for score, values in scores.items()},
        })
    _write_json(path, items)


def evaluate_test(manifest_path, data_root, checkpoint_dir, out_dir, *, device=None, protocol_path=None):
    """Explicitly evaluate only checkpoints and thresholds selected on validation."""
    manifest, manifest_hash = load_manifest(manifest_path)
    provenance = _provenance(manifest_hash, protocol_path)
    out_dir = Path(out_dir)
    frozen_path = out_dir / "frozen_choices.json"
    validation_path = out_dir / "validation_grid.json"
    frozen = _read_json(frozen_path)
    _verify_provenance(frozen, provenance, frozen_path)
    if frozen["validation_grid_sha256"] != _sha256(validation_path):
        raise ValueError("Validation artifact changed after choices were frozen.")
    expected_choices = _choices(_read_json(validation_path), checkpoint_dir)
    if any(frozen[key] != value for key, value in expected_choices.items()):
        raise ValueError("Frozen choices do not match the recorded validation results.")
    tasks = []
    for assignment, choice in frozen["assignments"].items():
        for run in choice["runs"].values():
            spec = RunSpec(assignment, run["seed"], choice["selected_learning_rate"], run["model"])
            checkpoint = checkpoint_path(checkpoint_dir, spec)
            if _sha256(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after choices were frozen: {checkpoint}")
            tasks.append((spec, checkpoint, run))
    data_root = resolve_data_root(data_root)
    device = _device(device)
    primary = frozen["primary_score"]
    results = {
        "provenance": provenance,
        "frozen_choices_sha256": _sha256(frozen_path),
        "split": "test",
        "primary_score": primary,
        "disease_name_rule": DISEASE_NAME_RULE,
        "evaluation_device": str(device),
        "assignments": {
            a: {
                "selected_learning_rate": frozen["assignments"][a]["selected_learning_rate"],
                "validation_tolerance_satisfied": frozen["assignments"][a]["validation_tolerance_satisfied"],
                "known_class_order": manifest["known_class_order"][a],
                "familiar_disease_matches": _familiar_disease_matches(manifest["class_roles"][a], "test"),
                "runs": [],
            }
            for a in ASSIGNMENTS
        },
    }
    for spec, checkpoint, run in tasks:
        metrics, rows, logits, epoch = _evaluate_run(
            spec, manifest, manifest_hash, data_root, checkpoint, device, "test", thresholds=run["thresholds"],
        )
        if _sha256(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError(f"{checkpoint} changed during test evaluation.")
        prediction_path = Path("test_predictions") / f"{checkpoint.stem}.json"
        _save_predictions(out_dir / prediction_path, rows, logits, run["thresholds"])
        results["assignments"][spec.assignment]["runs"].append({
            "seed": run["seed"],
            "model": run["model"],
            "selected_epoch": epoch,
            "metrics": metrics,
            "prediction_path": prediction_path.as_posix(),
        })
        print(f"Evaluated frozen test checkpoint {checkpoint.stem}", flush=True)
    for assignment in results["assignments"].values():
        assignment["summary"] = {score: _paired_summary(assignment["runs"], score) for score in SCORE_NAMES}
        assignment["hypothesis"] = _hypothesis(assignment["summary"][primary])
    results["supports_hypothesis"] = all(a["hypothesis"]["meets_all_criteria"] for a in results["assignments"].values())
    _write_json(out_dir / "test_results.json", results)
    return results


def summary_rows(results, *, score=None):
    """Return one row per assignment and metric for tables or CSV export."""
    score = score or results["primary_score"]
    return [
        {
            "assignment": assignment,
            "score": score,
            "metric": metric,
            "baseline_mean": values["baseline"]["mean"],
            "baseline_sd": values["baseline"]["sd"],
            "lesion_mean": values["lesion"]["mean"],
            "lesion_sd": values["lesion"]["sd"],
            "paired_difference_mean": values["lesion_minus_baseline"]["mean"],
            "paired_difference_sd": values["lesion_minus_baseline"]["sd"],
            **{f"difference_seed_{seed}": value for seed, value in values["paired_seed_differences"].items()},
        }
        for assignment, data in results["assignments"].items()
        for metric, values in data["summary"][score].items()
    ]


def plot_oscr(results, assignment, *, score=None):
    """Plot each frozen seed's OSCR curve with model mean AUOSCR in the legend."""
    import matplotlib.pyplot as plt

    score = score or results["primary_score"]
    data = results["assignments"][assignment]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, layout="constrained")
    for ax, group, title in zip(axes, ("familiar_host", "unfamiliar_host"), ("Familiar hosts", "Unfamiliar host control")):
        for model, color in zip(MODELS, ("#396a9e", "#b94e27")):
            mean = data["summary"][score][f"{group}/auoscr"][model]["mean"]
            model_runs = [run for run in data["runs"] if run["model"] == model]
            for i, run in enumerate(model_runs):
                item = run["metrics"]["scores"][score]["groups"][group]
                curve = item["oscr_curve"]
                ax.plot(
                    curve["unknown_false_acceptance"], curve["known_correct_acceptance"],
                    color=color, alpha=0.72, label=f"{model.title()} (mean area {mean:.3f})" if i == 0 else None,
                )
        ax.set(title=f"{title} (n = {item['unknown_count']})", xlabel="Unknown false acceptance rate", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Correct and accepted known fraction")
    fig.suptitle(f"Assignment {assignment} test OSCR using {score}")
    return fig


def plot_paired_differences(results):
    """Show paired test-seed differences and mean ± sample SD."""
    import matplotlib.pyplot as plt

    score = results["primary_score"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, metric, target, title in zip(
        axes,
        ("familiar_host/auoscr", "known_macro_f1"),
        (AUOSCR_TARGET, -KNOWN_F1_TOLERANCE),
        ("Familiar-host AUOSCR", "Known-class macro-F1"),
    ):
        for index, assignment in enumerate(ASSIGNMENTS):
            values = results["assignments"][assignment]["summary"][score][metric]
            differences = list(values["paired_seed_differences"].values())
            ax.scatter(index + np.linspace(-0.1, 0.1, len(SEEDS)), differences, color="#396a9e", label="Paired seeds" if index == 0 else None)
            estimate = values["lesion_minus_baseline"]
            ax.errorbar(index, estimate["mean"], yerr=estimate["sd"], fmt="D", color="#b94e27", capsize=4, label="Mean ± SD" if index == 0 else None)
        ax.axhline(0, color="0.35", linewidth=1)
        ax.axhline(target, color="0.35", linestyle="--", label="Protocol criterion")
        ax.set(xticks=range(len(ASSIGNMENTS)), xticklabels=ASSIGNMENTS, xlabel="Assignment", title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Lesion minus baseline")
    axes[1].legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "freeze", "test"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--allow-incomplete", action="store_true", help="Validate only checkpoints that exist.")
    args = parser.parse_args()
    if args.stage == "validate":
        validate_grid(
            args.manifest, args.data_root, args.checkpoint_dir, args.out_dir, device=args.device,
            protocol_path=args.protocol, require_complete=not args.allow_incomplete,
        )
    elif args.stage == "freeze":
        result = freeze_choices(args.manifest, args.checkpoint_dir, args.out_dir, protocol_path=args.protocol)
        print(f"Choices frozen with primary score {result['primary_score']}.")
    else:
        result = evaluate_test(
            args.manifest, args.data_root, args.checkpoint_dir, args.out_dir, device=args.device,
            protocol_path=args.protocol,
        )
        print(f"Both assignments meet the protocol criteria: {result['supports_hypothesis']}")


if __name__ == "__main__":
    main()

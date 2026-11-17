"""Report figures for the formal Option 2 experiment (batch 1: offline).

Every figure is derived only from frozen artifacts (manifest, validation
grid, frozen choices, test results, per-image predictions, experiment log),
so they regenerate without the dataset or model weights. Batch-2 figures
(prediction gallery, lesion overlays) live here too but need restored data.

CLI:  python formal_figures.py [--all] [--outdir artifacts/formal/figures]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
F = ROOT / "artifacts/formal"
EVAL = F / "evaluation"
SEEDS = [42, 123, 2026]
ASSIGNMENTS = ["A", "B"]
MODELS = ["baseline", "lesion"]
MODEL_LABEL = {"baseline": "Baseline", "lesion": "Lesion-supervised"}
MODEL_COLOR = {"baseline": "#4c72b0", "lesion": "#c44e52"}
GROUP_COLOR = {"known": "#4c72b0", "familiar": "#dd8452", "unfamiliar": "#55a868"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
})


def _load(path):
    return json.loads(Path(path).read_text())


def _run_key(model, assignment, seed, lr=0.0001):
    return f"{model}_{assignment}_seed{seed}_lr{lr}_epochs20"


def _predictions(results, assignment):
    """Return {(model, seed): rows} plus the frozen MaxLogit thresholds."""
    frozen = _load(EVAL / "frozen_choices.json")
    lr = frozen["assignments"][assignment]["selected_learning_rate"]
    out = {}
    thresholds = {}
    for run in results["assignments"][assignment]["runs"]:
        key = _run_key(run["model"], assignment, run["seed"], lr)
        out[(run["model"], run["seed"])] = _load(EVAL / run["prediction_path"])
        thresholds[(run["model"], run["seed"])] = frozen["assignments"][assignment]["runs"][key]["thresholds"]
    return out, thresholds


# ---------------------------------------------------------------- F1 class bars
def class_frequency(outdir):
    manifest = _load(F / "manifest.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, assignment in zip(axes, ASSIGNMENTS):
        order = manifest["known_class_order"][assignment]
        counts = Counter(
            s["class_name"] for s in manifest["samples"][assignment]
            if s["split"] == "train")
        values = [counts[c] for c in order]
        ax.bar(range(len(order)), values, color="#4c72b0", alpha=.85)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([c.replace(" ", "\n", 1) for c in order],
                           rotation=90, fontsize=4.5, va="top")
        ax.set_title(f"Assignment {assignment}: known-class training counts")
        ax.set_ylabel("Training images")
        for i, v in enumerate(values):
            ax.text(i, v + 3, str(v), ha="center", fontsize=4.5)
        ax.set_ylim(0, max(values) * 1.12)
    fig.tight_layout()
    path = Path(outdir) / "class_frequency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------- F2 training curves
def _parse_log_curves():
    text = (F / "experiment.log").read_text().splitlines()
    curve = defaultdict(dict)  # (model, assignment, seed, lr) -> {epoch: f1}
    pattern = re.compile(
        r"^(baseline|lesion) (A|B) seed (\d+) lr ([\d.e-]+) epoch (\d+)/20 "
        r"\| val macro-F1 ([\d.]+) \|")
    for line in text:
        m = pattern.match(line)
        if m:
            model, a, seed, lr, epoch, f1 = m.groups()
            curve[(model, a, int(seed), lr)][int(epoch)] = float(f1)
    return curve


def training_curves(outdir):
    grid = _load(EVAL / "validation_grid.json")
    frozen = _load(EVAL / "frozen_choices.json")
    curve = _parse_log_curves()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    epochs = np.arange(1, 21)
    for ax, assignment in zip(axes, ASSIGNMENTS):
        lr = frozen["assignments"][assignment]["selected_learning_rate"]
        for model in MODELS:
            runs = [curve[(model, assignment, seed, f"{lr:g}")] for seed in SEEDS]
            matrix = np.array([[runs[s][e] for e in epochs] for s in range(len(SEEDS))])
            mean = matrix.mean(axis=0)
            ax.plot(epochs, mean, color=MODEL_COLOR[model], label=MODEL_LABEL[model], lw=1.6)
            ax.fill_between(epochs, matrix.min(axis=0), matrix.max(axis=0),
                            color=MODEL_COLOR[model], alpha=.15, lw=0)
            selected = [grid["runs"][_run_key(model, assignment, seed, lr)]["selected_epoch"]
                        for seed in SEEDS]
            ax.scatter(selected, [mean[e - 1] for e in selected],
                       color=MODEL_COLOR[model], s=22, zorder=3, marker="o",
                       edgecolor="white", linewidth=.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Known validation macro-F1")
        ax.set_title(f"Assignment {assignment} (selected learning rate)")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Validation learning curves, mean over 3 seeds (band: min-max); dots mark selected checkpoints", fontsize=9, y=1.02)
    fig.tight_layout()
    path = Path(outdir) / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------ F3 confusion + destination
def confusion_heatmap(outdir, top_k=12):
    results = _load(EVAL / "test_results.json")
    for assignment in ASSIGNMENTS:
        preds, _ = _predictions(results, assignment)
        order = _load(F / "manifest.json")["known_class_order"][assignment]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
        for ax, model in zip(axes, MODELS):
            cm = np.zeros((25, 25))
            for seed in SEEDS:
                for row in preds[(model, seed)]:
                    if row["role"] == "known":
                        cm[row["target"], row["predicted_known_index"]] += 1
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(25)); ax.set_yticks(range(25))
            ax.set_xticklabels(order, rotation=90, fontsize=4)
            ax.set_yticklabels(order, fontsize=4)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
            ax.set_title(f"{MODEL_LABEL[model]} (summed over 3 seeds)")
            for i in range(25):
                for j in range(25):
                    if cm[i, j] >= 15:  # annotate only substantial cells
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                                fontsize=3.5, color="white" if cm_norm[i, j] > .5 else "#333")
        fig.colorbar(im, ax=axes, shrink=.7, label="Row-normalised recall")
        fig.suptitle(f"Assignment {assignment}: known-class test confusion", fontsize=11)
        path = Path(outdir) / f"confusion_{assignment}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        # destination of false-accepted familiar unknowns
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        for ax, model in zip(axes, MODELS):
            dest = Counter()
            total = Counter()
            for seed in SEEDS:
                for row in preds[(model, seed)]:
                    if row["role"] == "familiar_test":
                        total[row["class_name"]] += 1
                        if row["accepted"]["MaxLogit"]:
                            dest[(row["class_name"], order[row["predicted_known_index"]])] += 1
            pairs = dest.most_common(top_k)
            labels = [f"{a.replace(' ', '\n', 1)}\n→ {b.replace(' ', '\n', 1)}" for (a, b), _ in pairs]
            values = [v for _, v in pairs]
            ax.barh(range(len(pairs)), values, color=MODEL_COLOR[model], alpha=.85)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(labels, fontsize=5)
            ax.invert_yaxis()
            ax.set_xlabel("False-accepted images (of 3 seeds)")
            ax.set_title(MODEL_LABEL[model])
        fig.suptitle(f"Assignment {assignment}: where false-accepted familiar-host unknowns go (MaxLogit)", fontsize=10)
        fig.tight_layout()
        path_d = Path(outdir) / f"unknown_destination_{assignment}.png"
        fig.savefig(path_d, bbox_inches="tight")
        plt.close(fig)
    return [Path(outdir) / f"confusion_{a}.png" for a in ASSIGNMENTS] + \
           [Path(outdir) / f"unknown_destination_{a}.png" for a in ASSIGNMENTS]


# ------------------------------------------------------- F4 score distributions
def score_distributions(outdir, score="MaxLogit"):
    results = _load(EVAL / "test_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, assignment in zip(axes, ASSIGNMENTS):
        preds, thresholds = _predictions(results, assignment)
        groups = {"known": [], "familiar": [], "unfamiliar": []}
        thr = {m: [] for m in MODELS}
        for (model, seed), rows in preds.items():
            for row in rows:
                key = {"known": "known", "familiar_test": "familiar",
                       "unfamiliar_test": "unfamiliar"}[row["role"]]
                groups[key].append(row["scores"][score])
            thr[model].append(thresholds[(model, seed)][score])
        group_label = {"known": "known", "familiar": "familiar-host unknown",
                       "unfamiliar": "unfamiliar-host unknown"}
        for key, color in GROUP_COLOR.items():
            ax.hist(groups[key], bins=45, density=True, alpha=.55, color=color,
                    label=f"{group_label[key]} (n={len(groups[key]) // 6})")
        for model in MODELS:
            for t in thr[model]:
                ax.axvline(t, color=MODEL_COLOR[model], lw=.8, ls="--", alpha=.8)
        ax.set_xlabel(f"{score} score")
        ax.set_ylabel("Density")
        ax.set_title(f"Assignment {assignment}")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Test score distributions under the frozen {score} rule (dashed: per-seed thresholds)", fontsize=10, y=1.02)
    fig.tight_layout()
    path = Path(outdir) / "score_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------ F5 paired seeds
def paired_seeds(outdir):
    results = _load(EVAL / "test_results.json")
    frozen = _load(EVAL / "frozen_choices.json")
    prim = results["primary_score"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
    metrics = [("familiar_host/auoscr", "Familiar-host AUOSCR"),
               ("known_macro_f1", "Known macro-F1")]
    for col, assignment in enumerate(ASSIGNMENTS):
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row][col]
            b_vals, l_vals = [], []
            for s in SEEDS:
                rb = next(r for r in results["assignments"][assignment]["runs"]
                          if r["model"] == "baseline" and r["seed"] == s)["metrics"]
                rl = next(r for r in results["assignments"][assignment]["runs"]
                          if r["model"] == "lesion" and r["seed"] == s)["metrics"]
                if metric == "known_macro_f1":
                    b_vals.append(rb["known_macro_f1"]); l_vals.append(rl["known_macro_f1"])
                else:
                    b_vals.append(rb["scores"][prim]["groups"]["familiar_host"]["auoscr"])
                    l_vals.append(rl["scores"][prim]["groups"]["familiar_host"]["auoscr"])
            lims = [min(b_vals + l_vals) - .02, max(b_vals + l_vals) + .02]
            ax.plot(lims, lims, color="#999", lw=.8, ls=":")
            for i, s in enumerate(SEEDS):
                ax.annotate("", xy=(b_vals[i], l_vals[i]), xytext=(b_vals[i], b_vals[i]),
                            arrowprops=dict(arrowstyle="-", color="#bbb", lw=.7))
                ax.scatter(b_vals[i], l_vals[i], s=26, color=MODEL_COLOR["lesion"],
                           edgecolor="white", linewidth=.6, zorder=3)
                ax.annotate(f"seed {s}", (b_vals[i], l_vals[i]), fontsize=6.5,
                            xytext=(4, 4), textcoords="offset points")
            ax.errorbar(np.mean(b_vals), np.mean(l_vals),
                        xerr=np.std(b_vals, ddof=1), yerr=np.std(l_vals, ddof=1),
                        fmt="D", color="#222", ms=5, capsize=3, zorder=4, label="mean ± SD")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"Baseline {label}")
            ax.set_ylabel(f"Lesion-supervised {label}")
            ax.set_title(f"Assignment {assignment} — paired seeds", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    path = Path(outdir) / "paired_seeds.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------- F6 shared vs unshared names
def shared_unshared(outdir):
    results = _load(EVAL / "test_results.json")
    prim = results["primary_score"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = .35
    xs = []
    x = 0
    for assignment in ASSIGNMENTS:
        for group in ("familiar_shared_name", "familiar_unshared_name"):
            xs.append(x)
            x += 1
        x += .6
    positions = {m: [] for m in MODELS}
    labels = []
    i = 0
    for assignment in ASSIGNMENTS:
        for group, tag in (("familiar_shared_name", "shared"),
                           ("familiar_unshared_name", "unseen")):
            labels.append(f"{assignment}\n{tag}")
            positions["baseline"].append(xs[i] - width / 2)
            positions["lesion"].append(xs[i] + width / 2)
            i += 1
    for model in MODELS:
        means, sds, seeds_all = [], [], []
        for assignment in ASSIGNMENTS:
            for group in ("familiar_shared_name", "familiar_unshared_name"):
                vals = []
                for run in results["assignments"][assignment]["runs"]:
                    if run["model"] == model:
                        vals.append(run["metrics"]["scores"][prim]["groups"][group]["auoscr"])
                means.append(np.mean(vals)); sds.append(np.std(vals, ddof=1)); seeds_all.append(vals)
        ax.bar(positions[model], means, width, yerr=sds, capsize=3,
               color=MODEL_COLOR[model], alpha=.85, label=MODEL_LABEL[model],
               error_kw={"lw": 1})
        for pos, vals in zip(positions[model], seeds_all):
            ax.scatter([pos] * len(vals), vals, color="white", s=14, zorder=3,
                       edgecolor=MODEL_COLOR[model], linewidth=.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Familiar-host AUOSCR")
    ax.set_title("Test familiar-host unknowns split by shared vs unseen disease names (bars: seed mean ± SD; dots: seeds)")
    ax.legend(frameon=False)
    ax.set_ylim(0.45, 0.75)
    fig.tight_layout()
    path = Path(outdir) / "shared_unshared.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------ F7 prediction gallery
def _resolve_data_root():
    import os
    for candidate in (os.environ.get("PLANTSEG_ROOT"), "data/plantseg", "plantseg"):
        if candidate and (Path(candidate) / "Metadata.csv").is_file():
            return Path(candidate)
    return None


def _load_display_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = 224 / max(image.size)
    size = tuple(max(1, round(v * scale)) for v in image.size)
    return image.resize(size, Image.Resampling.BILINEAR)


def prediction_gallery(outdir, data_root=None):
    """Test-set prediction gallery: correct accepts, correct rejects, false accepts.

    Uses the saved per-image predictions (median MaxLogit over the three frozen
    lesion-supervised seeds) and only needs the restored images, not weights.
    """
    data_root = Path(data_root) if data_root else _resolve_data_root()
    if data_root is None:
        print("prediction_gallery skipped: dataset not restored")
        return []
    results = _load(EVAL / "test_results.json")
    frozen = _load(EVAL / "frozen_choices.json")
    paths = []
    for assignment in ASSIGNMENTS:
        order = _load(F / "manifest.json")["known_class_order"][assignment]
        lr = frozen["assignments"][assignment]["selected_learning_rate"]
        runs = [r for r in results["assignments"][assignment]["runs"] if r["model"] == "lesion"]
        merged = {}
        for run in runs:
            key = _run_key("lesion", assignment, run["seed"], lr)
            threshold = frozen["assignments"][assignment]["runs"][key]["thresholds"]["MaxLogit"]
            for row in _load(EVAL / run["prediction_path"]):
                entry = merged.setdefault(row["image_path"], {
                    "class_name": row["class_name"], "role": row["role"],
                    "scores": [], "accepted": [], "preds": []})
                entry["scores"].append(row["scores"]["MaxLogit"])
                entry["accepted"].append(row["accepted"]["MaxLogit"])
                entry["preds"].append(row["predicted_known_index"])
                entry["threshold"] = threshold
        entries = []
        for path_, e in merged.items():
            e["image_path"] = path_
            e["median"] = float(np.median(e["scores"]))
            e["n_accept"] = sum(e["accepted"])
            e["pred_names"] = {order[i] for i in e["preds"]}
            entries.append(e)
        correct_known = [e for e in entries if e["role"] == "known"
                         and len(e["pred_names"]) == 1
                         and e["pred_names"] == {e["class_name"]}]
        row_accept = sorted(correct_known, key=lambda e: -e["median"])[:4]
        row_reject = sorted(
            [e for e in entries if e["role"] == "familiar_test" and e["n_accept"] == 0],
            key=lambda e: e["median"])[:4]
        row_false = sorted(
            [e for e in entries if e["role"] == "familiar_test"
             and e["n_accept"] == 3 and len(e["pred_names"]) == 1],
            key=lambda e: -e["median"])[:4]
        rows = [("Correctly accepted known", row_accept, "#55a868"),
                ("Correctly rejected unknown", row_reject, "#4c72b0"),
                ("False-accepted unknown", row_false, "#c44e52")]
        fig, axes = plt.subplots(3, 4, figsize=(11, 8.2))
        for r, (label, items, color) in enumerate(rows):
            for c, ax in enumerate(axes[r]):
                ax.axis("off")
                if c >= len(items):
                    continue
                e = items[c]
                ax.imshow(_load_display_image(data_root / e["image_path"]))
                if r == 0:
                    verdict = f"✓ accepted  {e['median']:.1f}"
                elif r == 1:
                    verdict = f"✓ rejected  {e['median']:.1f} < thr {e['threshold']:.1f}"
                else:
                    verdict = f"✗ → {e['pred_names'].pop()}  {e['median']:.1f} ≥ thr {e['threshold']:.1f}"
                ax.set_title(f"{e['class_name']}\n{verdict}", fontsize=7.5, color=color)
            axes[r][0].text(-0.08, 0.5, label, rotation=90, va="center", ha="center",
                            transform=axes[r][0].transAxes, fontsize=9, color=rows[r][2])
        fig.suptitle(f"Assignment {assignment} test gallery — frozen MaxLogit, "
                     "median over 3 lesion-supervised seeds", fontsize=11)
        fig.tight_layout()
        path = Path(outdir) / f"prediction_gallery_{assignment}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def generate_all(outdir=None):
    outdir = Path(outdir or (F / "figures"))
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [class_frequency(outdir), training_curves(outdir),
             score_distributions(outdir), paired_seeds(outdir), shared_unshared(outdir)]
    paths += confusion_heatmap(outdir)
    paths += prediction_gallery(outdir)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    for path in generate_all(args.outdir):
        print(f"saved {path}")


if __name__ == "__main__":
    main()

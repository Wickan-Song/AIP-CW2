"""Publication-grade report figures for the CW2 Option 2 formal experiment.

Figures regenerated entirely from frozen artifacts (manifest, frozen
choices, test results, per-image predictions). No dataset or weights are
required (the prediction gallery additionally needs the restored images).
Colour scheme: Okabe-Ito, colour-blind friendly, consistent across all
figures (baseline blue, lesion-supervised vermillion). Exports one PDF
(vector) and one PNG (preview) per figure plus a provenance manifest.

Canvas sizing is content-driven: heights and label gutters derive from the
number of categories and measured label widths, panels are rearranged or
figures split when labels would otherwise shrink below the 6pt readability
floor, and every figure passes a programmatic audit (no clipped text, no
sub-floor fonts, no tick-label collisions) at its exported page size.

Run:  .venv/bin/python report_figures.py
Outputs: artifacts/formal/figures/report/
"""

import json
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from figure_audit import audit_figure, mm_from_pt, text_width_mm
from formal_figures import _load, _run_key, _predictions  # reuse frozen loaders

ROOT = Path(__file__).resolve().parent
F = ROOT / "artifacts/formal"
EVAL = F / "evaluation"
OUT = F / "figures" / "report"
SEEDS = [42, 123, 2026]
ASSIGNMENTS = ["A", "B"]
MODELS = ["baseline", "lesion"]

# Okabe-Ito on white
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "sky": "#56B4E9", "yellow": "#E69F00", "purple": "#CC79A7",
      "black": "#000000", "grey": "#7F7F7F"}
MODEL_COLOR = {"baseline": OI["blue"], "lesion": OI["vermillion"]}
MODEL_LABEL = {"baseline": "Baseline", "lesion": "Lesion-supervised"}
GROUP_COLOR = {"known": OI["blue"], "familiar": OI["vermillion"], "unfamiliar": OI["green"]}
ROLE_COLOR = {"known": OI["blue"], "familiar": OI["vermillion"], "unfamiliar": OI["green"]}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MM = 1 / 25.4
FULL_W = 183 * MM   # full page width
HALF_W = 89 * MM    # single column

MIN_FONT_PT = 6.0   # readability floor at final print size
PROVENANCE = []
AUDIT = {}          # figure name -> list of violation strings


def _audit_figure(fig, name, min_pt=MIN_FONT_PT):
    problems = audit_figure(fig, name, min_pt)
    AUDIT[name] = problems
    return problems


def _save(fig, name, caption_note):
    problems = _audit_figure(fig, name)
    if problems:
        print(f"WARNING {name}: {len(problems)} audit problem(s)")
        for p in problems:
            print(f"  - {p}")
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    PROVENANCE.append({
        "figure": name, "pdf": str(pdf.relative_to(ROOT)), "png": str(png.relative_to(ROOT)),
        "size_in": [float(fig.get_size_inches()[0]), float(fig.get_size_inches()[1])],
        "audit": {"min_font_pt": MIN_FONT_PT, "violations": problems},
        "note": caption_note,
        "sources": ["artifacts/formal/manifest.json",
                    "artifacts/formal/evaluation/test_results.json",
                    "artifacts/formal/evaluation/frozen_choices.json",
                    "artifacts/formal/evaluation/test_predictions/"],
    })
    print(f"saved {pdf.name} + {png.name}")


def _per_seed_metric(results, assignment, metric):
    """Return {model: [per-seed values]} for metric in {auoscr, macro_f1, accuracy, fa_rate}."""
    prim = results["primary_score"]
    out = {m: [] for m in MODELS}
    for run in sorted(results["assignments"][assignment]["runs"], key=lambda r: r["seed"]):
        m = run["metrics"]
        if metric == "macro_f1":
            out[run["model"]].append(m["known_macro_f1"])
        elif metric == "accuracy":
            out[run["model"]].append(m["known_accuracy"])
        elif metric == "auoscr":
            out[run["model"]].append(
                m["scores"][prim]["groups"]["familiar_host"]["auoscr"])
        elif metric == "fa_rate":
            out[run["model"]].append(
                m["scores"][prim]["groups"]["familiar_host"]["unknown_false_acceptance"])
    return out


# ------------------------------------------------------------------ Figure 1
def fig1_data_and_design():
    """Partition and role composition. The 25-class frequency panel was split
    into fig1b so that no axis label renders below the 6pt readability floor."""
    manifest = _load(F / "manifest.json")
    fig, ax = plt.subplots(figsize=(FULL_W, 74 * MM), layout="constrained")
    roles = [("known", "Known"), ("familiar", "Familiar-host unknown"),
             ("unfamiliar", "Unfamiliar-host control")]
    splits = ["train", "val", "test"]
    width = 0.38
    totals = {a: np.zeros(3) for a in ASSIGNMENTS}
    for i, assignment in enumerate(ASSIGNMENTS):
        bottoms = np.zeros(3)
        for role_key, role_label in roles:
            vals = []
            for sp in splits:
                n = sum(1 for s in manifest["samples"][assignment]
                        if s["split"] == sp and
                        (s["role"] == "known" if role_key == "known"
                         else s["role"] in ("familiar_dev", "familiar_test") if role_key == "familiar"
                         else s["role"] in ("unfamiliar_dev", "unfamiliar_test")))
                vals.append(n)
            pos = np.arange(3) + (i - 0.5) * width
            ax.bar(pos, vals, width, bottom=bottoms,
                   color=ROLE_COLOR[role_key], alpha=.35 + .3 * i,
                   label=role_label if i == 0 else None)
            for j, v in enumerate(vals):
                if 80 <= v < 1000:
                    ax.text(pos[j], bottoms[j] + v / 2, str(v), ha="center", va="center",
                            fontsize=6.5, color="white")
            bottoms += np.array(vals)
            totals[assignment] = bottoms
        for j, sp in enumerate(splits):
            ax.text(np.arange(3)[j] + (i - 0.5) * width, totals[assignment][j] + 55,
                    f"{int(totals[assignment][j])}", ha="center", fontsize=7, color=OI["black"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Train", "Validation", "Test"])
    ax.set_ylabel("Images")
    ax.set_ylim(0, 1980)
    ax.set_title("Partition and role composition (solid: assignment A, translucent: assignment B)",
                 loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, "fig1_data_and_design",
          "Stacked partition/role counts per assignment; the class-role rule is stated in the caption.")


def fig1b_class_counts():
    """Known-class training frequencies as horizontal bars: every class name
    stays horizontal and >= 7pt instead of 4.4pt rotated labels."""
    manifest = _load(F / "manifest.json")
    orderA = manifest["known_class_order"]["A"]
    countsA = Counter(s["class_name"] for s in manifest["samples"]["A"] if s["split"] == "train")
    countsB = Counter(s["class_name"] for s in manifest["samples"]["B"] if s["split"] == "train")
    shared = [c for c in orderA if c in countsB]
    label_fs = 7.0
    pitch_mm = max(4.0, mm_from_pt(label_fs * 1.45))  # one text row per class
    fig_h = len(shared) * pitch_mm + 24             # title + x label + margins
    fig, ax = plt.subplots(figsize=(FULL_W, fig_h * MM), layout="constrained")
    y = np.arange(len(shared))[::-1]
    ax.barh(y + 0.19, [countsA[c] for c in shared], 0.38, color=OI["blue"], alpha=.9,
            label="Assignment A")
    ax.barh(y - 0.19, [countsB[c] for c in shared], 0.38, color=OI["vermillion"], alpha=.9,
            label="Assignment B")
    for yi, c in zip(y, shared):
        wa, wb = countsA[c], countsB[c]
        ax.text(max(wa, wb) + 2.5, yi, f"{wa} / {wb}", va="center", ha="left",
                fontsize=6, color=OI["grey"])
    ax.set_yticks(y)
    ax.set_yticklabels(shared, fontsize=label_fs)
    ax.set_xlabel("Known-class training images (A / B)")
    ax.set_xlim(0, max(max(countsA[c], countsB[c]) for c in shared) * 1.14)
    ax.xaxis.set_major_locator(MaxNLocator(prune="upper"))  # keep edge tick labels inside
    ax.grid(axis="x", alpha=.3, lw=.5)
    ax.set_axisbelow(True)
    ax.set_title("Training-set class frequencies (right margin: images A / B)", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, "fig1b_class_counts",
          "Horizontal frequencies keep all 25 class names readable at print size.")


# ------------------------------------------------------------------ Figure 2
def fig2_score_distributions():
    results = _load(EVAL / "test_results.json")
    score = results["primary_score"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 66 * MM), layout="constrained",
                             sharex=True, sharey=True)
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
        labels = {"known": "Known", "familiar": "Familiar-host unknown",
                  "unfamiliar": "Unfamiliar-host unknown"}
        for key in ("unfamiliar", "familiar", "known"):   # draw known last for readability
            ax.hist(groups[key], bins=40, density=True, alpha=.55,
                    color=GROUP_COLOR[key], label=labels[key])
        for model in MODELS:
            for t in thr[model]:
                ax.axvline(t, color=MODEL_COLOR[model], lw=.9, ls=(0, (4, 2)), alpha=.9)
        ax.axvline(thr["baseline"][0], color=MODEL_COLOR["baseline"], lw=.9,
                   ls=(0, (4, 2)), label="Baseline thresholds")
        ax.axvline(thr["lesion"][0], color=MODEL_COLOR["lesion"], lw=.9,
                   ls=(0, (4, 2)), label="Lesion-supervised thresholds")
        ax.set_xlabel(f"{score} score")
        ax.set_title(f"Assignment {assignment}", loc="left")
        if assignment == "A":
            ax.set_ylabel("Density")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(prune="upper", nbins=8))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncols=3, frameon=False, fontsize=7)
    _save(fig, "fig2_score_distributions",
          "Test score distributions under the frozen MaxLogit rule; dashed lines are per-seed thresholds.")


# ------------------------------------------------------------------ Figure 3
def fig3_paired_seeds():
    results = _load(EVAL / "test_results.json")
    # 2x2 grid: each panel is ~85 mm wide, so two-line titles and tick labels
    # fit without clipping (the previous 1x4 row squeezed them to ~44 mm).
    # No shared x-axis: the two rows are different quantities, so each panel
    # needs its own square limits for the identity line to mean anything.
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 108 * MM), layout="constrained")
    for r, (metric, label) in enumerate((("auoscr", "Familiar-host AUOSCR"),
                                         ("macro_f1", "Known-class macro-F1"))):
        for c, assignment in enumerate(ASSIGNMENTS):
            ax = axes[r][c]
            short = "AUOSCR" if metric == "auoscr" else "macro-F1"
            data = _per_seed_metric(results, assignment, metric)
            b, l = np.array(data["baseline"]), np.array(data["lesion"])
            lo = min(b.min(), l.min()) - 0.015
            hi = max(b.max(), l.max()) + 0.015
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.plot([lo, hi], [lo, hi], color=OI["grey"], lw=.8, ls=":")
            for i, seed in enumerate(SEEDS):
                ax.plot([b[i], b[i]], [b[i], l[i]], color=OI["grey"], lw=.8, zorder=1)
                ax.scatter(b[i], l[i], s=20, color=MODEL_COLOR["lesion"],
                           edgecolor="white", linewidth=.6, zorder=3)
            ax.errorbar(b.mean(), l.mean(), xerr=b.std(ddof=1), yerr=l.std(ddof=1),
                        fmt="D", ms=4.5, color=OI["black"], capsize=2.5, zorder=4)
            d = l - b
            ax.set_title(f"Assignment {assignment} — mean diff {d.mean():+.4f} "
                         f"({sum(d > 0)}/3 seeds > 0)", loc="left", fontsize=8, pad=6)
            ax.set_xlabel(f"Baseline {short}")
            ax.xaxis.set_major_locator(MaxNLocator(prune="upper", nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(prune="upper", nbins=5))
            if c == 0:
                ax.set_ylabel(f"Lesion-supervised\n{short}")
            if r == 0 and c == 0:
                ax.annotate("diamond: mean ± SD\n(3 paired seeds)", xy=(0.04, 0.96),
                            xycoords="axes fraction", va="top", fontsize=7)
    _save(fig, "fig3_paired_seeds",
          "Per-seed paired comparison against the identity line; diamonds are seed means with SD.")


# ------------------------------------------------------------------ Figure 4
def fig4_false_acceptance_destinations(top_k=10, name_suffix=""):
    results = _load(EVAL / "test_results.json")
    score = results["primary_score"]
    # Stacked (2 rows) instead of side-by-side: destination labels get the
    # width they need at 7pt and the bars keep a usable plotting area.
    fig_h = 30 + top_k * 6 + 12
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, fig_h * MM), layout="constrained",
                             sharex=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m], alpha=.9)
               for m in MODELS]
    for ax, assignment in zip(axes, ASSIGNMENTS):
        order = _load(F / "manifest.json")["known_class_order"][assignment]
        preds, _ = _predictions(results, assignment)
        width = 0.38
        pair_index = {}
        labels = []
        for seed in SEEDS:
            for row in preds[("lesion", seed)]:
                if row["role"] == "familiar_test" and row["accepted"][score]:
                    key = (row["class_name"], order[row["predicted_known_index"]])
                    pair_index.setdefault(key, None)
        # rank pairs by total false accepts over both models
        totals = Counter()
        per_model = {m: Counter() for m in MODELS}
        for (model, seed), rows in preds.items():
            for row in rows:
                if row["role"] == "familiar_test" and row["accepted"][score]:
                    key = (row["class_name"], order[row["predicted_known_index"]])
                    per_model[model][key] += 1
                    totals[key] += 1
        top = [k for k, _ in totals.most_common(top_k)]
        y = np.arange(len(top))
        for i, model in enumerate(MODELS):
            vals = [per_model[model][k] for k in top]
            ax.barh(y + (i - 0.5) * width, vals, width, color=MODEL_COLOR[model], alpha=.9)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} → {b}" for a, b in top], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Assignment {assignment}", loc="left")
        ax.set_xlabel("False-accepted images (3 seeds pooled)")
    axes[0].legend(handles, [MODEL_LABEL[m] for m in MODELS], frameon=False,
                   fontsize=7.5, loc="lower right")
    _save(fig, f"fig4_false_acceptance_destinations{name_suffix}",
          "Most frequent true-unknown to predicted-known destinations at the frozen thresholds.")


# ------------------------------------------------------------------ Figure 5
def _oscr_points(known_scores, known_correct, unknown_scores):
    n_k, n_u = len(known_scores), len(unknown_scores)
    rows = sorted([("k", s, c) for s, c in zip(known_scores, known_correct)]
                  + [("u", s, None) for s in unknown_scores], key=lambda r: -r[1])
    points = [(0.0, 0.0)]
    au = ac = 0
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][1] == rows[i][1]:
            kind, _, corr = rows[j]
            if kind == "u":
                au += 1
            elif corr:
                ac += 1
            j += 1
        points.append((au / n_u, ac / n_k))
        i = j
    if points[-1][0] < 1.0:
        points.append((1.0, sum(known_correct) / n_k))
    return np.array(points)


def fig5_oscr_curves():
    results = _load(EVAL / "test_results.json")
    score = results["primary_score"]
    fig, axes = plt.subplots(1, 2, figsize=(HALF_W * 1.9, 66 * MM), layout="constrained",
                             sharey=True)
    for ax, assignment in zip(axes, ASSIGNMENTS):
        preds, thresholds = _predictions(results, assignment)
        for model in MODELS:
            for seed in SEEDS:
                rows = preds[(model, seed)]
                known = [r for r in rows if r["role"] == "known"]
                ks = [r["scores"][score] for r in known]
                corr = [r["target"] == r["predicted_known_index"] for r in known]
                us = [r["scores"][score] for r in rows if r["role"] == "familiar_test"]
                pts = _oscr_points(ks, corr, us)
                ax.plot(pts[:, 0], pts[:, 1], color=MODEL_COLOR[model],
                        lw=1.5 if seed == SEEDS[0] else .8, alpha=.95 if seed == SEEDS[0] else .45,
                        label=MODEL_LABEL[model] if seed == SEEDS[0] else None)
        ax.plot([0, 1], [0, 1], color=OI["grey"], lw=.8, ls=":")
        ax.set_xlabel("Unknown false-acceptance rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.9)
        ax.set_title(f"Assignment {assignment}", loc="left")
        if assignment == "A":
            ax.set_ylabel("Correct-and-accepted\nknown rate")
            ax.legend(frameon=False, loc="lower right", fontsize=7)
    _save(fig, "fig5_oscr_curves",
          "OSCR curves per seed; thick line seed 42, thin lines seeds 123 and 2026.")


# ------------------------------------------------------------------ Figure 6
def fig6_shared_unseen():
    results = _load(EVAL / "test_results.json")
    prim = results["primary_score"]
    fig, ax = plt.subplots(figsize=(HALF_W * 1.5, 60 * MM), layout="constrained")
    groups = [("familiar_shared_name", "shared"), ("familiar_unshared_name", "unseen")]
    positions, labels, ns = [], [], []
    x = 0.0
    for assignment in ASSIGNMENTS:
        for _, tag in groups:
            positions.append(x)
            labels.append(f"{assignment} {tag}")
            x += 1.0
        x += 0.5
    width = 0.38
    for i, model in enumerate(MODELS):
        means, sds, dots, counts = [], [], [], []
        for assignment in ASSIGNMENTS:
            for group, _ in groups:
                vals, n = [], None
                for run in results["assignments"][assignment]["runs"]:
                    if run["model"] == model:
                        g = run["metrics"]["scores"][prim]["groups"][group]
                        vals.append(g["auoscr"])
                        n = g["unknown_count"]
                means.append(np.mean(vals))
                sds.append(np.std(vals, ddof=1))
                dots.append(vals)
                counts.append(n)
        pos = [p + (i - 0.5) * width for p in positions]
        ax.bar(pos, means, width, yerr=sds, capsize=2.5, color=MODEL_COLOR[model],
               alpha=.9, label=MODEL_LABEL[model], error_kw={"lw": .9})
        for p, vals in zip(pos, dots):
            ax.scatter([p] * len(vals), vals, s=12, color="white",
                       edgecolor=MODEL_COLOR[model], linewidth=.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, counts)])
    ax.tick_params(labelsize=7.5)
    ax.set_ylabel("Familiar-host AUOSCR")
    ax.set_ylim(0.45, 0.75)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    _save(fig, "fig6_shared_unseen_names",
          "Familiar-host unknowns split by disease name shared with a known class on another host.")


# ------------------------------------------------------------------ Figure 7
def fig7_confusion_matrices():
    """Full 25x25 known-class test confusion per assignment (colour only, no
    per-cell numbers — the readable host-level aggregation is Figure 7b and
    per-class counts live in the frozen prediction files). Canvas geometry is
    derived from the class count and measured label widths; the two model
    panels share one y-label gutter so both stay >= 6pt."""
    results = _load(EVAL / "test_results.json")
    manifest = _load(F / "manifest.json")
    label_fs = 6.0
    for assignment in ASSIGNMENTS:
        order = manifest["known_class_order"][assignment]
        preds, _ = _predictions(results, assignment)
        gutter = max(text_width_mm(c, label_fs) for c in order) + 1.5  # shared y labels
        xlab_h = max(text_width_mm(c, label_fs) for c in order) + 3   # rotated x labels
        pitch = max(2.5, mm_from_pt(label_fs * 1.4))
        axis_mm = len(order) * pitch
        fig_h = axis_mm + xlab_h + 18
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, fig_h * MM), layout="constrained",
                                 sharey=True)
        im = None
        for ax, model in zip(axes, MODELS):
            cm = np.zeros((25, 25))
            for seed in SEEDS:
                for row in preds[(model, seed)]:
                    if row["role"] == "known":
                        cm[row["target"], row["predicted_known_index"]] += 1
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1,
                           interpolation="nearest")
            ax.set_xticks(range(25))
            ax.set_yticks(range(25))
            ax.set_xticklabels(order, rotation=90, fontsize=label_fs, color="#444444")
            ax.set_yticklabels(order, fontsize=label_fs, color="#444444")
            ax.tick_params(length=0)
            ax.set_xlabel("Predicted class")
            if model == "baseline":
                ax.set_ylabel("True class")
            ax.set_title(MODEL_LABEL[model], loc="left", fontsize=9, pad=8)
            hosts = [c.split()[0] for c in order]
            boundaries = [i for i in range(1, 25) if hosts[i] != hosts[i - 1]]
            for b in boundaries:
                ax.axhline(b - 0.5, color="#999999", lw=.5)
                ax.axvline(b - 0.5, color="#999999", lw=.5)
        fig.colorbar(im, ax=axes, shrink=.55, label="Row-normalised recall (3 seeds pooled)")
        fig.suptitle(f"Assignment {assignment}: known-class test confusion", fontsize=10)
        _save(fig, f"fig7_confusion_{assignment}",
              "Colour-only full matrix; grey lines separate plant hosts.")


def fig7_host_confusion(name_suffix=""):
    """Compact host-level confusion for the report: one panel per assignment
    (lesion-supervised arm), 7x7 with readable counts. The full 25x25
    matrices and the four-panel arm comparison stay in the notebook."""
    results = _load(EVAL / "test_results.json")
    manifest = _load(F / "manifest.json")
    hosts_order = ["apple", "banana", "corn", "soybean", "tomato", "wheat", "zucchini"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 84 * MM), layout="constrained",
                             sharex=True, sharey=True)
    hidx = {h: i for i, h in enumerate(hosts_order)}
    im = None
    for ax, assignment in zip(axes, ASSIGNMENTS):
        order = manifest["known_class_order"][assignment]
        preds, _ = _predictions(results, assignment)
        hm = np.zeros((7, 7))
        for seed in SEEDS:
            for row in preds[("lesion", seed)]:
                if row["role"] == "known":
                    ti = hidx[order[row["target"]].split()[0]]
                    pi = hidx[order[row["predicted_known_index"]].split()[0]]
                    hm[ti, pi] += 1
        hm_norm = hm / hm.sum(axis=1, keepdims=True)
        im = ax.imshow(hm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        for i in range(7):
            for j in range(7):
                dark = hm_norm[i, j] > 0.5
                ax.text(j, i, int(hm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if dark else "#333333")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(hosts_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(hosts_order, fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(f"Assignment {assignment} — Lesion-supervised", loc="left", fontsize=8.5)
        if assignment == "A":
            ax.set_ylabel("True host", fontsize=8)
        ax.set_xlabel("Predicted host", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=.85, pad=.02, label="Row-normalised share")
    _save(fig, f"fig7_host_confusion{name_suffix}",
          "Report-sized host aggregation of the known-class test confusion; "
          "the full 25x25 matrices and the baseline panels are in the notebook.")


def fig7b_confusion_by_host():
    """Host-level 7x7 aggregation of the same pooled counts, with readable
    per-cell numbers — the report-facing confusion view."""
    results = _load(EVAL / "test_results.json")
    manifest = _load(F / "manifest.json")
    hosts_order = ["apple", "banana", "corn", "soybean", "tomato", "wheat", "zucchini"]
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 150 * MM), layout="constrained",
                             sharex=True, sharey=True)
    im = None
    for r, assignment in enumerate(ASSIGNMENTS):
        order = manifest["known_class_order"][assignment]
        hidx = {h: i for i, h in enumerate(hosts_order)}
        preds, _ = _predictions(results, assignment)
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            hm = np.zeros((7, 7))
            for seed in SEEDS:
                for row in preds[(model, seed)]:
                    if row["role"] == "known":
                        ti = hidx[order[row["target"]].split()[0]]
                        pi = hidx[order[row["predicted_known_index"]].split()[0]]
                        hm[ti, pi] += 1
            hm_norm = hm / hm.sum(axis=1, keepdims=True)
            im = ax.imshow(hm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            for i in range(7):
                for j in range(7):
                    dark = hm_norm[i, j] > 0.5
                    ax.text(j, i, int(hm[i, j]), ha="center", va="center",
                            fontsize=7.5, color="white" if dark else "#333333")
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels(hosts_order, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(hosts_order, fontsize=8)
            ax.tick_params(length=0)
            ax.set_title(f"Assignment {assignment} — {MODEL_LABEL[model]}",
                         loc="left", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("True host", fontsize=8)
            if r == 1:
                ax.set_xlabel("Predicted host", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=.85, label="Row-normalised share (3 seeds pooled)")
    fig.suptitle("Known-class test confusion aggregated to plant hosts "
                 "(counts are pooled seed-level judgments)", fontsize=10)
    _save(fig, "fig7b_confusion_by_host",
          "7x7 host aggregation of Figure 7 counts with per-cell numbers.")


# ------------------------------------------------------------------ Figure 8
def fig8_training_curves():
    from formal_figures import _parse_log_curves
    grid = _load(EVAL / "validation_grid.json")
    frozen = _load(EVAL / "frozen_choices.json")
    curve = _parse_log_curves()
    epochs = np.arange(1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 66 * MM), layout="constrained")
    for ax, assignment in zip(axes, ASSIGNMENTS):
        lr = frozen["assignments"][assignment]["selected_learning_rate"]
        for model in MODELS:
            runs = [curve[(model, assignment, seed, f"{lr:g}")] for seed in SEEDS]
            matrix = np.array([[runs[s][e] for e in epochs] for s in range(3)])
            mean = matrix.mean(axis=0)
            ax.plot(epochs, mean, color=MODEL_COLOR[model], lw=1.6,
                    label=MODEL_LABEL[model])
            ax.fill_between(epochs, matrix.min(axis=0), matrix.max(axis=0),
                            color=MODEL_COLOR[model], alpha=.15, lw=0)
            for s, seed in enumerate(SEEDS):
                sel = grid["runs"][_run_key(model, assignment, seed, lr)]["selected_epoch"]
                ax.scatter(sel, matrix[s][sel - 1], s=24, color=MODEL_COLOR[model],
                           edgecolor="white", linewidth=.7, zorder=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Known validation macro-F1")
        ax.set_title(f"Assignment {assignment}", loc="left")
        ax.set_xticks([1, 5, 10, 15, 20])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncols=2, frameon=False)
    fig.suptitle("Validation learning curves: mean over 3 seeds (band: min-max); "
                 "dots mark each seed's selected checkpoint", fontsize=9)
    _save(fig, "fig8_training_curves",
          "Curves parsed from experiment.log at the selected learning rate.")


# ------------------------------------------------------------------ Figure 9
def fig9_prediction_gallery(width_mm=183.0, name_suffix=""):
    from formal_figures import _resolve_data_root, _load_display_image
    data_root = _resolve_data_root()
    if data_root is None:
        print("fig9_prediction_gallery skipped: dataset not restored")
        return
    results = _load(EVAL / "test_results.json")
    frozen = _load(EVAL / "frozen_choices.json")
    score = results["primary_score"]
    for assignment in ASSIGNMENTS:
        order = _load(F / "manifest.json")["known_class_order"][assignment]
        lr = frozen["assignments"][assignment]["selected_learning_rate"]
        merged = {}
        for run in results["assignments"][assignment]["runs"]:
            if run["model"] != "lesion":
                continue
            key = _run_key("lesion", assignment, run["seed"], lr)
            threshold = frozen["assignments"][assignment]["runs"][key]["thresholds"][score]
            for row in _load(EVAL / run["prediction_path"]):
                e = merged.setdefault(row["image_path"], {
                    "class_name": row["class_name"], "role": row["role"],
                    "scores": [], "accepted": [], "preds": set(), "threshold": threshold})
                e["scores"].append(row["scores"][score])
                e["accepted"].append(row["accepted"][score])
                e["preds"].add(row["predicted_known_index"])
        entries = []
        for path, e in merged.items():
            e["image_path"] = path
            e["median"] = float(np.median(e["scores"]))
            e["n_accept"] = sum(e["accepted"])
            entries.append(e)
        correct_known = [e for e in entries if e["role"] == "known"
                         and e["n_accept"] == 3 and len(e["preds"]) == 1
                         and order[next(iter(e["preds"]))] == e["class_name"]]
        rejected = [e for e in entries if e["role"] == "familiar_test" and e["n_accept"] == 0]
        false_ok = [e for e in entries if e["role"] == "familiar_test"
                    and e["n_accept"] == 3 and len(e["preds"]) == 1]
        rows = [
            ("Correctly accepted known", sorted(correct_known, key=lambda e: -e["median"])[:4],
             OI["green"]),
            ("Correctly rejected unknown", sorted(rejected, key=lambda e: e["median"])[:4],
             OI["blue"]),
            ("False-accepted unknown", sorted(false_ok, key=lambda e: -e["median"])[:4],
             OI["vermillion"]),
        ]
        fig, axes = plt.subplots(3, 4, figsize=(width_mm * MM, 150 * MM), layout="constrained")
        for r, (label, items, color) in enumerate(rows):
            axes[r][0].text(-0.14, 0.5, label.replace(" ", "\n"), rotation=90,
                            va="center", ha="center", fontsize=8, color=color,
                            transform=axes[r][0].transAxes)
            for c, ax in enumerate(axes[r]):
                ax.axis("off")
                if c >= len(items):
                    continue
                e = items[c]
                ax.imshow(_load_display_image(data_root / e["image_path"]))
                if r == 0:
                    verdict = f"accepted, {e['median']:.1f}"
                elif r == 1:
                    verdict = f"rejected, {e['median']:.1f} < thr {e['threshold']:.1f}"
                else:
                    verdict = f"as {order[next(iter(e['preds']))]}, {e['median']:.1f} ≥ thr {e['threshold']:.1f}"
                name_line = "\n".join(textwrap.wrap(e["class_name"], 14))
                verdict_line = "\n".join(textwrap.wrap(verdict, 20))
                ax.set_title(f"{name_line}\n{verdict_line}", fontsize=7.2, color=color, pad=5)
        fig.suptitle(f"Assignment {assignment} test gallery: frozen {score} rule, "
                     "median over 3 lesion-supervised seeds", fontsize=10)
        _save(fig, f"fig9_prediction_gallery_{assignment}{name_suffix}",
              "Rows: correct accepts, correct rejects, false accepts.")


def _final_size_previews():
    """Render every exported PNG at its final paper display size (183 mm page
    width) for a readability check: 96 dpi simulates reading the PDF at 100%
    screen zoom, 150 dpi a close-reading zoom."""
    from PIL import Image, ImageDraw
    names = [p["figure"] for p in PROVENANCE]
    for screen_dpi in (96, 150):
        w_px = round(183 / 25.4 * screen_dpi)
        label_h = max(18, round(screen_dpi / 6))
        pages, total_h = [], 20
        for name in names:
            im = Image.open(OUT / f"{name}.png").convert("RGB")
            h = round(im.height * w_px / im.width)
            pages.append((name, im.resize((w_px, h), Image.LANCZOS)))
            total_h += h + label_h
        sheet = Image.new("RGB", (w_px + 20, total_h), "white")
        draw = ImageDraw.Draw(sheet)
        y = 10
        for name, im in pages:
            draw.text((0, y + label_h // 2 - 5), f"{name}", fill=(200, 0, 0))
            sheet.paste(im, (10, y))
            y += label_h + im.height
        path = OUT / f"final_size_preview_{screen_dpi}dpi.png"
        sheet.save(path, dpi=(300, 300))
        print(f"wrote {path.name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_data_and_design()
    fig1b_class_counts()
    fig2_score_distributions()
    fig3_paired_seeds()
    fig4_false_acceptance_destinations()
    fig4_false_acceptance_destinations(top_k=6, name_suffix="_top6")
    fig5_oscr_curves()
    fig6_shared_unseen()
    fig7_confusion_matrices()
    fig7_host_confusion("_report")
    fig7b_confusion_by_host()
    fig8_training_curves()
    fig9_prediction_gallery()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "report_figures.py",
        "python": sys.version.split()[0],
        "matplotlib": matplotlib.__version__,
        "palette": "Okabe-Ito (colour-blind friendly); baseline=#0072B2, lesion=#D55E00",
        "readability_audit": {
            "min_font_pt": MIN_FONT_PT,
            "violations": {name: problems for name, problems in AUDIT.items() if problems},
            "note": ("Canvas sizes derive from content (label widths, class counts); "
                     "each figure is checked for clipped text, sub-floor fonts and "
                     "tick-label collisions at its exported page size."),
        },
        "frozen_inputs": {
            "manifest_sha256_pinned": "bd98c23f968b2b4f6e572a32226fcad20e42639deb6262ddc115a3c0b149be1e",
            "test_results": "artifacts/formal/evaluation/test_results.json",
            "frozen_choices": "artifacts/formal/evaluation/frozen_choices.json",
        },
        "figures": PROVENANCE,
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _final_size_previews()
    failures = {n: p for n, p in AUDIT.items() if p}
    if failures:
        print(f"AUDIT FAILED for {len(failures)} figure(s):", file=sys.stderr)
        for n, p in failures.items():
            for problem in p:
                print(f"  {n}: {problem}", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {OUT / 'provenance.json'} — readability audit clean")


if __name__ == "__main__":
    main()

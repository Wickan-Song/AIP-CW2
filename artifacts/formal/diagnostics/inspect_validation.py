"""Inspect RGB validation predictions with the frozen checkpoints and thresholds.

Run from the project root with the project Python environment.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from formal_evaluation import _disease_groups
from formal_training import (
    IMAGE_SIZE, RGBDataset, RunSpec, build_model, load_manifest, read_checkpoint,
    resize_rgb, resolve_data_root, resolve_device,
)

FORMAL = ROOT / 'artifacts' / 'formal'
OUTPUT = Path(__file__).parent
SEEDS = (42, 123, 2026)
SCORE = 'MaxLogit'


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_assignment(assignment, manifest, manifest_hash, frozen, grid, data_root, device):
    classes = manifest['known_class_order'][assignment]
    rows = [row for row in manifest['samples'][assignment] if row['split'] == 'val']
    groups = _disease_groups(rows, manifest['class_roles'][assignment], 'val')
    images = [resize_rgb(data_root / row['image_path'])[0] for row in rows]
    loader = DataLoader(RGBDataset(rows, data_root), batch_size=32)
    choice = frozen['assignments'][assignment]
    records = [dict(row, predictions={}) for row in rows]
    mean_maps = np.zeros((len(rows), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    accepted = {}
    for key, run in choice['runs'].items():
        variant, seed = run['model'], run['seed']
        spec = RunSpec(assignment, seed, choice['selected_learning_rate'], variant)
        path = FORMAL / 'checkpoints' / f'{key}.pt'
        assert digest(path) == run['checkpoint_sha256']
        checkpoint = read_checkpoint(path, spec, manifest_hash, classes)
        model = build_model(len(classes), variant)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        model.to(device).eval()
        threshold = run['thresholds'][SCORE]
        decisions = []
        start = 0
        with torch.inference_mode():
            for batch, _ in loader:
                batch = batch.to(device)
                logits = model(batch)
                if variant == 'lesion':
                    branch_logits, pixel_logits = model(batch, predict_lesions=True)
                    torch.testing.assert_close(branch_logits, logits)
                    maps = pixel_logits.sigmoid()[:, 0].cpu().numpy()
                    mean_maps[start:start + len(batch)] += maps / len(SEEDS)
                scores, indices = logits.max(dim=1)
                keep = (scores >= threshold).cpu().numpy()
                decisions.extend(keep.tolist())
                for offset, (score, index, decision) in enumerate(zip(scores.tolist(), indices.tolist(), keep.tolist())):
                    records[start + offset]['predictions'][f'{variant}_{seed}'] = {
                        'predicted_class': classes[index], 'score': score,
                        'margin': score - threshold, 'accepted': decision,
                    }
                start += len(batch)
        expected = grid['runs'][key]['metrics']['scores'][SCORE]
        decisions = np.asarray(decisions, dtype=bool)
        assert threshold == expected['threshold']
        for group, mask in groups.items():
            assert int(decisions[mask].sum()) == expected['groups'][group]['unknown_accepted_count']
        known = np.asarray([row['role'] == 'known' for row in rows])
        assert int(decisions[known].sum()) == expected['known_accepted_count']
        accepted[variant, seed] = decisions
        print(f'Checked {key} against saved validation acceptance counts', flush=True)
        del model

    crops = []
    for image, probability, row in zip(images, mean_maps, records):
        left = (IMAGE_SIZE - image.width) // 2
        top = (IMAGE_SIZE - image.height) // 2
        crop = probability[top:top + image.height, left:left + image.width]
        crops.append(crop)
        row['map_p95'] = float(np.quantile(crop, 0.95))
        row['map_area_at_0_5'] = float(np.mean(crop >= 0.5))
        for variant in ('baseline', 'lesion'):
            votes = [row['predictions'][f'{variant}_{seed}'] for seed in SEEDS]
            row[f'{variant}_accepted_seeds'] = sum(vote['accepted'] for vote in votes)
            row[f'{variant}_median_margin'] = float(np.median([vote['margin'] for vote in votes]))

    base = np.stack([accepted['baseline', seed] for seed in SEEDS])
    lesion = np.stack([accepted['lesion', seed] for seed in SEEDS])
    by_group = {}
    for group, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            continue
        by_group[group] = {
            'images': n, 'baseline_accepts': int(base[:, mask].sum()),
            'lesion_accepts': int(lesion[:, mask].sum()),
            'baseline_only_accepts': int((base[:, mask] & ~lesion[:, mask]).sum()),
            'lesion_only_accepts': int((lesion[:, mask] & ~base[:, mask]).sum()),
            'accepted_by_both_in_all_seeds': int((base[:, mask] & lesion[:, mask]).all(axis=0).sum()),
        }
    familiar = [i for i, r in enumerate(rows) if r['role'] == 'familiar_dev']
    by_disease = {}
    for name in sorted({rows[i]['class_name'] for i in familiar}):
        indices = [i for i in familiar if rows[i]['class_name'] == name]
        guesses = Counter(records[i]['predictions'][f'lesion_{seed}']['predicted_class']
                          for i in indices for seed in SEEDS if accepted['lesion', seed][i])
        by_disease[name] = {
            'images': len(indices), 'baseline_accepts': int(base[:, indices].sum()),
            'lesion_accepts': int(lesion[:, indices].sum()),
            'most_common_accepted_lesion_guesses': guesses.most_common(3),
        }
    by_role = {}
    for role in ('known', 'familiar_dev', 'unfamiliar_dev'):
        members = [r for r in records if r['role'] == role]
        by_role[role] = {
            'images': len(members),
            'median_map_p95': float(np.median([r['map_p95'] for r in members])),
            'median_map_area_at_0_5': float(np.median([r['map_area_at_0_5'] for r in members])),
            'images_with_no_pixel_at_0_5': sum(r['map_area_at_0_5'] == 0 for r in members),
        }
    return {'by_group': by_group, 'by_disease': by_disease, 'by_role': by_role,
            'images': records}, images, crops


def draw_gallery(results, category):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout='constrained')
    for row_axes, (assignment, (summary, images, maps)) in zip(axes, results.items()):
        rows = summary['images']
        if category == 'known':
            ordered = sorted((i for i, r in enumerate(rows) if r['role'] == 'known'),
                             key=lambda i: rows[i]['map_p95'])
            selections = [ordered[round(q * (len(ordered) - 1))] for q in (0.1, 0.4, 0.7, 0.95)]
        else:
            candidates = [i for i, r in enumerate(rows) if r['role'] == 'familiar_dev']
            candidates.sort(key=lambda i: (rows[i]['lesion_accepted_seeds'], rows[i]['baseline_accepted_seeds'],
                                           rows[i]['lesion_median_margin']), reverse=True)
            selections = []
            used = set()
            for index in candidates:
                disease = rows[index]['class_name']
                if disease not in used:
                    selections.append(index)
                    used.add(disease)
                if len(selections) == 4:
                    break
        for ax, i in zip(row_axes, selections):
            row, probability = rows[i], maps[i]
            ax.imshow(images[i])
            overlay = ax.imshow(probability, cmap='magma', vmin=0, vmax=1,
                                alpha=np.clip(probability * 0.85, 0, 0.8))
            if category == 'known':
                details = f"95th percentile pixel score {row['map_p95']:.2f}"
            else:
                guesses = Counter(row['predictions'][f'lesion_{seed}']['predicted_class'] for seed in SEEDS)
                guess = guesses.most_common(1)[0][0]
                details = (f"Most common guess: {guess}\nAccepted by {row['lesion_accepted_seeds']}/3 lesion models "
                           f"and {row['baseline_accepted_seeds']}/3 baselines")
            ax.set_title(f"{assignment}  |  {row['class_name']}\n" + '\n'.join(
                textwrap.fill(line, 38) for line in details.splitlines()), fontsize=9)
            ax.axis('off')
    fig.colorbar(overlay, ax=axes, shrink=0.78, label='Mean predicted lesion probability over three seeds')
    title = ('Known-class validation images across activation levels' if category == 'known'
             else 'Recurring false accepts on familiar-host unknown validation images')
    fig.suptitle(title)
    fig.savefig(OUTPUT / f'validation_{category}_overlays.png', dpi=150)
    plt.close(fig)


def main():
    manifest, manifest_hash = load_manifest(FORMAL / 'manifest.json')
    evaluation = FORMAL / 'evaluation'
    frozen = read_json(evaluation / 'frozen_choices.json')
    grid = read_json(evaluation / 'validation_grid.json')
    assert frozen['primary_score'] == SCORE
    assert frozen['provenance']['manifest_sha256'] == manifest_hash
    assert frozen['validation_grid_sha256'] == digest(evaluation / 'validation_grid.json')
    data_root, device = resolve_data_root(ROOT / 'data' / 'plantseg'), resolve_device()
    results = {assignment: inspect_assignment(assignment, manifest, manifest_hash, frozen, grid, data_root, device)
               for assignment in ('A', 'B')}
    for category in ('known', 'unknown'):
        draw_gallery(results, category)
    (OUTPUT / 'validation_diagnostics.json').write_text(json.dumps({
        'split': 'val', 'score': SCORE, 'seeds': SEEDS,
        'frozen_choices_sha256': digest(evaluation / 'frozen_choices.json'),
        'assignments': {assignment: item[0] for assignment, item in results.items()},
    }, indent=2) + '\n')
    for assignment, (summary, _, _) in results.items():
        print(assignment, json.dumps({key: value for key, value in summary.items() if key != 'images'}))


if __name__ == '__main__':
    main()

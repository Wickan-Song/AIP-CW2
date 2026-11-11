"""Verify the contents and optionally run a CPU inference demo."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / 'artifacts' / 'formal'
DATA = ROOT / 'data' / 'plantseg'


def sha256(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def packaged_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Expected a relative package path: {relative}')
    local = root.joinpath(*path.parts)
    if not local.is_file() or not local.resolve().is_relative_to(root.resolve()):
        raise FileNotFoundError(local)
    return local


def check_files(full):
    lines = (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    for line in lines:
        expected, relative = line.split('  ', 1)
        path = packaged_path(ROOT, relative)
        if full and sha256(path) != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
    manifest = json.loads((FORMAL/'manifest.json').read_bytes())
    for records in manifest['samples'].values():
        for row in records:
            packaged_path(DATA, row['image_path'])
            if row.get('mask_path'):
                packaged_path(DATA, row['mask_path'])
    print(f'{len(lines):,} listed files present. All manifest image and training mask paths resolve.')
    if full:
        print('All file SHA-256 checksums verified.')


def demo():
    import torch
    from formal_training import RGBDataset, RunSpec, build_model, load_manifest, read_checkpoint

    manifest, manifest_hash = load_manifest(FORMAL/'manifest.json')
    frozen = json.loads((FORMAL/'evaluation/frozen_choices.json').read_bytes())
    assert frozen['provenance']['manifest_sha256'] == manifest_hash
    assignment = frozen['assignments']['A']
    row = next(r for r in manifest['samples']['A'] if r['split']=='val' and r['role']=='known')
    classes = manifest['known_class_order']['A']
    image, _ = RGBDataset([row], DATA)[0]
    for key, run in assignment['runs'].items():
        if run['seed'] != 42:
            continue
        path = FORMAL/'checkpoints'/f'{key}.pt'
        if sha256(path) != run['checkpoint_sha256']:
            raise ValueError(f'Frozen checkpoint checksum mismatch: {key}')
        spec = RunSpec('A', 42, assignment['selected_learning_rate'], run['model'])
        saved = read_checkpoint(path, spec, manifest_hash, classes)
        model = build_model(len(classes), run['model']).eval()
        model.load_state_dict(saved['model_state_dict'])
        with torch.inference_mode():
            scores = model(image.unsqueeze(0))
        if not torch.isfinite(scores).all() or scores.shape != (1, len(classes)):
            raise ValueError('Invalid inference output.')
        score, prediction = scores[0].max(0)
        threshold = run['thresholds']['MaxLogit']
        label = classes[prediction.item()] if score.item() >= threshold else 'unknown'
        print(f'{run["model"]} CPU demo: {label} (MaxLogit {score.item():.4f}; threshold {threshold:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Check each file SHA-256.')
    parser.add_argument('--demo', action='store_true', help='Run frozen models on one validation RGB image.')
    args = parser.parse_args()
    check_files(args.full)
    if args.demo:
        demo()

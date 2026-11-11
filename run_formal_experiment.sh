#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

python_bin=${AIPCW2_PYTHON:-python}
formal_dir=artifacts/formal
manifest=$formal_dir/manifest.json
protocol=$formal_dir/protocol.json
checkpoints=$formal_dir/checkpoints
results=$formal_dir/evaluation

"$python_bin" - <<'PY'
import json
from pathlib import Path
from formal_training import BATCH_SIZE, DEFAULT_ASSIGNMENTS, DEFAULT_LEARNING_RATES, DEFAULT_SEEDS, EPOCHS, MODEL_ARCHITECTURES
base = Path('artifacts/formal')
protocol = json.loads((base / 'protocol.json').read_text())
manifest = json.loads((base / 'manifest.json').read_text())
actual = {
    'assignments': list(DEFAULT_ASSIGNMENTS), 'seeds': list(DEFAULT_SEEDS),
    'learning_rates': list(DEFAULT_LEARNING_RATES), 'models': list(MODEL_ARCHITECTURES),
    'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
}
for key, value in actual.items():
    if protocol[key] != value:
        raise ValueError(f'Training settings differ from formal protocol: {key}')
if manifest['formal_audit']['pilot_manifest_sha256'] != protocol['pilot_manifest_sha256']:
    raise ValueError('Formal manifest and protocol refer to different pilot manifests.')
PY

printf 'Training started at %s\n' "$(date -Iseconds)"
"$python_bin" -u -m formal_training --manifest "$manifest" --output-dir "$checkpoints"

evaluation_args=(--manifest "$manifest" --checkpoint-dir "$checkpoints" --out-dir "$results" --protocol "$protocol")
printf 'Validation started at %s\n' "$(date -Iseconds)"
"$python_bin" -u -m formal_evaluation validate "${evaluation_args[@]}"
"$python_bin" -u -m formal_evaluation freeze "${evaluation_args[@]}"
printf 'Frozen test started at %s\n' "$(date -Iseconds)"
"$python_bin" -u -m formal_evaluation test "${evaluation_args[@]}"
printf 'Formal experiment finished at %s\n' "$(date -Iseconds)"

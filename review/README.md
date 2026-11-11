# CW2 independent review scripts

Reproducible audit of the frozen CW2 Option 2 artifacts against the task sheet.
Standard library only; run from the repository root:

```shell
python3 review/01_manifest_audit.py
python3 review/02_protocol_freeze_audit.py
python3 review/03_test_recomputation.py
python3 review/04_log_notebook_audit.py
python3 review/05_code_conformity_checks.py
# or all at once:
for s in review/0*_*.py; do python3 "$s" || break; done
```

Each script exits non-zero on failure and its saved output lives in
`review/results/`. Findings and the full task-sheet evidence matrix are in
[CW2_independent_review.md](CW2_independent_review.md).

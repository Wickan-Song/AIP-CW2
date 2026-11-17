"""Independent audit 04: run log and notebook integrity.

Parses experiment.log for run coverage, timing and protocol echo; scans both
notebooks for saved-output integrity (no error outputs, figures present,
training flags default off) and confirms notebooks are in English.

Run from the repository root:  python3 review/04_log_notebook_audit.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = (ROOT / "artifacts/formal/experiment.log").read_text().splitlines()

findings = []
def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append(f"{label}: {detail}")

def main():
    # --- experiment.log -----------------------------------------------------
    run_heads = [l for l in LOG if l.startswith("Run ")]
    check("log lists 24 run headers", len(run_heads) == 24, f"n={len(run_heads)}")
    check("log start/end timestamps present",
          LOG[0].startswith("Training started at")
          and any(l.startswith("Formal experiment finished at") for l in LOG))
    devices = {re.search(r"on (\w+)", l).group(1) for l in LOG
               if re.search(r"on \w+ with \d+ train", l)}
    check("all 24 runs trained on one recorded device (mps)", devices == {"mps"}, str(devices))
    times = [float(m.group(1)) for l in LOG if (m := re.search(r"\| (\d+(?:\.\d+)?)s$", l))]
    check("480 epoch timings recorded", len(times) == 480, f"n={len(times)}")
    total_min = sum(times) / 60
    print(f"    total logged epoch time: {total_min:.1f} min "
          f"(mean {sum(times)/len(times):.1f}s/epoch)")
    val_lines = [l for l in LOG if " val macro-F1 " in l]
    nan_lines = [l for l in val_lines if "nan" in l.lower()]
    check("no NaN validation epoch in log", not nan_lines, str(nan_lines[:2]))
    check("protocol verdict echoed in log (criteria not met)",
          any("meet the protocol criteria: False" in l for l in LOG))

    # --- notebooks -----------------------------------------------------------
    for name in ("Predict.ipynb", "Formal_Experiment.ipynb"):
        nb = json.loads((ROOT / name).read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
        error_outputs = [
            (i, o.get("ename"))
            for i, c in enumerate(nb["cells"])
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"]
        check(f"{name}: no error outputs saved", not error_outputs, str(error_outputs[:3]))
        n_imgs = sum(1 for c in nb["cells"] for o in c.get("outputs", [])
                     if "image/png" in o.get("data", {}))
        check(f"{name}: saved figure outputs present", n_imgs >= 3, f"n={n_imgs}")
        check(f"{name}: markdown cells present", len(md) >= 3, f"n={len(md)}")
        executed = sum(1 for c in code if c.get("execution_count"))
        check(f"{name}: code cells carry execution counts", executed >= len(code) - 1,
              f"{executed}/{len(code)}")

        text = "".join(
            "".join(c["source"]) for c in nb["cells"]) + "".join(
            "".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", []))
        cjk = {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}
        check(f"{name}: no CJK characters in source or saved outputs", not cjk,
              str(sorted(cjk)[:10]))

        if name == "Formal_Experiment.ipynb":
            src = "".join("".join(c["source"]) for c in code)
            flags = {k: (m.group(2) if (m := re.search(rf"(\w*{k}\w*\s*=\s*)(\w+)", src)) else None)
                     for k in ("RUN_TRAINING", "RUN_VALIDATION", "RUN_TEST")}
            check("Formal_Experiment.ipynb: training flags default to False",
                  all(v == "False" for v in flags.values()), str(flags))

    print(f"\n{len(findings)} finding(s).")
    sys.exit(1 if findings else 0)

if __name__ == "__main__":
    main()

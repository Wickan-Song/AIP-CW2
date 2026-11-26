# AIP304 Coursework 2: Plant disease recognition with lesion supervision

**Option 2: open-set recognition on PlantSeg v7.** This project tests whether auxiliary lesion-mask supervision helps a classifier reject unseen diseases on familiar plants while preserving known-disease performance.

Start with the [five-page report](report/main.pdf), the [formal experiment notebook](Formal_Experiment.ipynb), or the [experiment overview](docs/experiment-overview.md). The 24 formal training runs, frozen test evaluation, and validation error analysis are complete. A run in a fresh Colab or Kaggle environment has yet to be verified.

## Submission entry points

| Item | Where to start |
|---|---|
| Formal notebook | [Formal_Experiment.ipynb](Formal_Experiment.ipynb). Run all cells in review mode to regenerate the saved results and report figures. |
| Report | [Five-page PDF](report/main.pdf). The [LaTeX source and figure list](report/README.md) are provided for reproducibility. |
| Inference | Use the final notebook section after restoring the assets, or run `python verify_package.py --demo` from the repository root. |

The coursework video is a separate submission. The task sheet specifies a three-minute maximum, a visible presenter, and a working inference demonstration.

## Results

| Assignment | Known accuracy, baseline → lesion | Familiar-host AUOSCR, baseline → lesion | Paired AUOSCR change |
|---|---:|---:|---:|
| A | 78.50% → 78.44% | 0.6245 → 0.6316 | +0.0071 |
| B | 74.71% → 74.78% | 0.6007 → 0.5987 | −0.0020 |

Values are means over three paired seeds under the frozen MaxLogit rule. Both assignments met the known-class macro-F1 tolerance. Neither reached the predefined AUOSCR gain of 0.02. See the [validation error analysis](artifacts/formal/diagnostics/validation_findings.md) for lesion maps and false-acceptance examples.

## Setup and inference

1. Clone this repository or download and extract the source ZIP. Run commands from the repository root.
2. Restore the split assets from [release v0.1.0](https://github.com/Wickan-Song/AIP-CW2/releases/tag/v0.1.0), then verify the package:

   ```sh
   python download_release_assets.py
   python verify_package.py --full
   ```

   If you already have both original archives, run `python restore_assets.py plantseg.zip AIP-CW2-weights.zip` instead. The restore scripts verify and extract the original [PlantSeg v7](https://zenodo.org/records/17719108) archive and 26 checkpoints. For notebook and experiment code, you can use an alternate extracted location by pointing `PLANTSEG_ROOT` to the directory containing `Metadata.csv`, `images/`, and `annotations/`. The package verifier and its CPU demo expect the standard directory layout.
3. Create a Python 3.9–3.12 virtual environment with `python -m venv .venv`. Activate it with `source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

   ```sh
   python -m pip install -r requirements.txt
   python verify_package.py --demo
   jupyter lab
   ```

   The demo loads two frozen checkpoints and runs inference on a validation image on CPU. Select the same Python environment as your notebook kernel.

## Reading and reproducing the experiments

- [Formal_Experiment.ipynb](Formal_Experiment.ipynb) contains the main evaluation and saved figures. Training, validation, and test rerun flags default to `False`. [Predict.ipynb](Predict.ipynb) records the earlier pilot.
- [Experiment handoff](docs/experiment-handoff.md) records split construction, training choices, verification, and limitations. The [task sheet](<docs/AIP304 CW2 Task Sheet.pdf>) defines the original assignment.
- [Formal artifacts](artifacts/formal/) contain the split manifest, protocol, logs, and [frozen evaluation outputs](artifacts/formal/evaluation/). The [near-duplicate audit](artifacts/near_duplicate_audit/README.md) documents the exclusions.
- [Report sources](report/README.md) explain how to compile the report. Run `python report_figures.py` to regenerate report figures. Without restored images, the prediction gallery is skipped.

To rerun training without touching the frozen results, follow the flags in the notebook. Its outputs go under `.cw2_outputs/`. For the original shell workflow, use [run_formal_experiment.sh](run_formal_experiment.sh) in a separate checkout on macOS, Linux, or WSL. The training code detects CUDA, Apple MPS, or CPU. A first training run may require downloading ImageNet weights.

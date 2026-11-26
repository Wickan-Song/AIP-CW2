# AIP304 CW2 handoff

Read `README.md` first. `docs/experiment-handoff.md` records project history, decisions, verification, and remaining work. `docs/AIP304 CW2 Task Sheet.pdf` is the assignment specification. `docs/experiment-overview.md` explains Option 2 and the A/B assignments using an apple example.

Treat this folder as the project root. Open the notebook from here. Restore the original dataset and frozen weights from the GitHub Release using `download_release_assets.py` as described in the README. The extracted dataset is under `data/plantseg`, with images, annotations and metadata. Manifest paths are relative to that folder. `python verify_package.py --full` checks all supplied files. With dependencies installed, `python verify_package.py --demo` loads two frozen weights and runs one validation image on CPU.

`Predict.ipynb` is the pilot. `Formal_Experiment.ipynb` is the formal experiment, with saved outputs and training flags disabled by default. Use `artifacts/formal/protocol.json`, `artifacts/formal/manifest.json`, `artifacts/formal/evaluation/frozen_choices.json` and `artifacts/formal/evaluation/test_results.json` as the formal evidence. The main rule is MaxLogit and the selected learning rate is 0.0001 for both assignments. The primary metric differences (lesion minus baseline) are +0.0071 for A and -0.0020 for B. The predefined gain criterion was +0.02 in each assignment. Treat further experiments as exploratory and keep the frozen result reproducible.

Formal training, test evaluation, validation error analysis, and the English report are available. A run in a fresh Colab or Kaggle environment has yet to be verified.

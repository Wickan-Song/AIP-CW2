# AIP CW2

**Plant disease recognition with lesion supervision · AIP304 Coursework 2, Option 2**

这是截至 2026 年 11 月 11 日的 Option 2 项目快照。24 次正式训练、冻结后的测试和验证误判分析已经完成。中文流程报告用于读懂当前工作。最终英文报告、演示视频、文献整理和云端复现仍待完成。

## 快速开始

1. Clone 本仓库，或下载仓库的 Source code ZIP 后解压。从包含 `README.md` 的目录打开 Codex 或 IDE。
2. 在仓库的 [Releases](../../releases) 下载 `plantseg.zip` 和 `AIP-CW2-weights.zip`。保留这两个附件的文件名，从项目根目录运行下面的命令。参数可以换成文件实际的下载路径。

```shell
python restore_assets.py plantseg.zip AIP-CW2-weights.zip
python verify_package.py --full
```

3. 让 Codex 先读 `AGENTS.md`、本 README 和 [CW2 流程记忆](<docs/CW2 流程记忆.md>)。也可以先读[中文实验流程报告](<docs/AIP304 CW2 实验流程与阶段结果.pdf>)，再打开 [Formal_Experiment.ipynb](Formal_Experiment.ipynb)。正式 Notebook 保存了结果表和图，默认查看模式可以在恢复附件后重新生成输出。

## 正式结果

| 划分 | 已知病准确率，基线 → 病斑组 | 熟悉宿主 AUOSCR，基线 → 病斑组 | 配对差值 |
|---|---:|---:|---:|
| A | 78.50% → 78.44% | 0.6245 → 0.6316 | +0.0071 |
| B | 74.71% → 74.78% | 0.6007 → 0.5987 | -0.0020 |

数值为冻结 MaxLogit 规则下三个随机种子的平均值。两套划分均满足已知病 Macro-F1 容差，均未达到预设的 AUOSCR 平均增益 0.02。[验证误判排查](artifacts/formal/diagnostics/validation_findings.md)记录了实际病斑图和误判案例。

## 文件在哪里

| 内容 | 包内路径 |
|---|---|
| 任务单、中文报告、Codex 流程记忆及复查笔记 | `docs/` |
| 试跑和正式实验 Notebook | `Predict.ipynb`、`Formal_Experiment.ipynb` |
| 清单生成、训练及评估代码 | `formal_manifest.py`、`formal_training.py`、`formal_evaluation.py` |
| PlantSeg v7 的完整已解压数据 | `data/plantseg/` |
| 近重复候选、接触表和排除决策 | `artifacts/near_duplicate_audit/` |
| 两份试跑模型权重、指标和试跑复查 | `artifacts/`、`artifacts/review_20261111/` |
| 24 份正式权重、训练日志、方案与样本清单 | `artifacts/formal/` |
| 冻结规则、正式结果、每张测试图的预测 | `artifacts/formal/evaluation/` |
| 验证误判、病斑热图及复现脚本 | `artifacts/formal/diagnostics/` |

Release 的 `plantseg.zip` 是未经改动的原始数据包。恢复脚本校验原始附件和每个恢复文件，并把数据放在项目根目录下的 `data/plantseg`，把权重放回 `artifacts/`。若你把数据单独移到别处，可自行设置 `PLANTSEG_ROOT` 指向包含 `Metadata.csv`、`images/` 和 `annotations/` 的目录。保留数据内的元数据和原始分区。原始下载记录及来源是 [PlantSeg v7](https://zenodo.org/records/17719108)。

## 检查文件与运行

仅查看 PDF、日志、流程记忆和 Notebook 保存的输出时，可以直接打开文件。完整校验使用 Python 标准库。在项目根目录执行：

```shell
python verify_package.py --full
```

如果要运行 Notebook 或做推理，用 Python 3.9–3.12 新建自己的虚拟环境，再安装依赖。macOS/Linux 用 `source .venv/bin/activate` 激活，Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活。

```shell
python -m venv .venv
```

激活后执行：

```shell
python -m pip install -r requirements.txt
python verify_package.py --demo
jupyter lab
```

选择该环境的 Python 内核。CPU 可以完成自检和读结果。训练代码可自动选择 CUDA、Apple MPS 或 CPU，具体取决于安装的 PyTorch 和机器。正式 Notebook 的 `RUN_TRAINING`、`RUN_VALIDATION`、`RUN_TEST` 默认为 `False`。要用已有模型看结果，按默认值运行。要从头复训，先在项目的副本中保留好打包时的正式产物，再用独立的输出目录运行训练、验证、冻结和测试。需要下载 ImageNet 预训练权重时，首次训练需要网络。`run_formal_experiment.sh` 是 macOS/Linux/WSL 的命令行入口，Windows 也可以在 Notebook 中运行相同的 Python 函数。

用于交接的改动仅涉及包内路径、解释器查找和跨平台文件锁。冻结清单、模型权重、结果和原来的 Notebook 内容随包保留。交接版本已在 Mac 的新目录核验文件、模型加载和正式 Notebook 的查看模式。课程要求的干净 Colab/Kaggle 验证仍在待办中。

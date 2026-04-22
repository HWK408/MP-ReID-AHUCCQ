# Repository Guidelines

## Project Structure & Module Organization
This repository implements MP-ReID training and evaluation in Python. Top-level entry points are `train.py` and `test.py` for the baseline pipeline, and `train_uniprompt.py` and `test_uniprompt.py` for Uni-Prompt experiments. Core code is organized by responsibility: `datasets/` contains dataset definitions and dataloader builders; `model/` contains model construction and CLIP components; `loss/`, `solver/`, and `processor/` contain objectives, optimizers/schedulers, and train/inference loops; `utils/` contains logging, metrics, reranking, and I/O helpers. YAML experiment configs live in `configs/`, especially `configs/ours/`. Dataset notes are in `dataset.md`; generated logs, checkpoints, and experiment outputs are under `log/` or configured `OUTPUT_DIR`.

## Build, Test, and Development Commands
Create the expected environment with Python 3.8 and CUDA-enabled PyTorch:

```bash
conda create -n UniPrompt python=3.8
conda activate UniPrompt
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Run baseline training:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/ours/cctv_ir_cctv_rgb.yml
```

Run Uni-Prompt training or evaluation:

```bash
python train_uniprompt.py --config_file configs/ours/cctv_ir_cctv_rgb.yml
python test_uniprompt.py --config_file configs/ours/cctv_ir_cctv_rgb.yml TEST.WEIGHT log/exp_cctv_ir_cctv_rgb/ViT-B-16_60.pth
```

Use trailing config options, such as `OUTPUT_DIR log/debug`, to override YAML values without editing files.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and follow existing module patterns. Keep dataset classes and model builders in their matching packages, and name new config files by dataset or experiment setting, for example `configs/ours/uav_ir_uav_rgb.yml`. Prefer descriptive function names such as `make_dataloader` and `do_inference`. No formatter or linter config is committed; keep imports tidy and avoid unrelated rewrites.

## Testing Guidelines
There is no standalone unit-test suite. Treat `test.py` and `test_uniprompt.py` as evaluation/smoke-test entry points and verify changes with the smallest relevant config before launching long training jobs. When adding pure utility logic, add focused `test_*.py` files or a `tests/` package and document any required fixtures or dataset assumptions.

## Commit & Pull Request Guidelines
Recent commits use short messages such as `Update README.md` and `Upload training code`. Keep messages concise, imperative, and scoped, for example `Update MMMP dataloader paths`. Pull requests should describe the experiment or bug fix, list changed configs, include key metrics when behavior changes, and note required dataset paths or checkpoint weights. Avoid committing large generated checkpoints or logs unless they are intentional release artifacts.

## Security & Configuration Tips
Do not hard-code private dataset roots, credentials, or local absolute paths in shared configs. Prefer documenting required overrides in the PR and passing paths through YAML copies or command-line options.

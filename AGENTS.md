# Repository Guidelines

## Project Structure & Modules
Core training logic lives in `pretrain.py`, with evaluation entry points in `evaluate.py`. Modeling components reside under `models/`, while shared utilities sit in `utils/`. Configuration defaults are in `config/cfg_pretrain.yaml` and architecture variants in `config/arch/`. Dataset builders live in `dataset/`, writing artifacts to `data/`. Checkpoints and tracking outputs land in `checkpoints/` and `wandb/`; keep large generated files out of commits. Visual references are stored in `assets/`, with research notes under `notes/`.

## Build, Data, and Runtime Commands
Install dependencies with `pip install -r requirements.txt`. Prepare ARC data via `python dataset/build_arc_dataset.py`; build Sudoku data with `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`. A single-GPU training smoke test: `OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=10 eval_interval=2`. Multi-GPU jobs use `torchrun --nproc-per-node 8 pretrain.py data_path=...`. Log in to Weights & Biases beforehand using `wandb login`.

## Coding Style & Naming
Stick to PEP 8 with four-space indentation and `snake_case` for functions, modules, and Hydra overrides (e.g., `arch.loss.loss_type=softmax_cross_entropy`). Reuse existing pydantic and OmegaConf patterns, and add type hints on public functions. Brief inline comments should explain non-obvious tensor shapes or control flow.

## Testing & Evaluation
There is no dedicated unit-test suite; rely on short training runs and evaluation scripts. After data or model changes, rebuild a small dataset shard and verify formatting with `puzzle_visualizer.html`. Validate checkpoints with `python evaluate.py checkpoint=/path/to/step_xxx.pt` and inspect metrics in the terminal or W&B dashboard. Document any reproducibility commands in your PR.

## Commit & Pull Requests
Use concise, imperative commit titles such as `Refine halt scheduler`. Collapse local fix-up commits before pushing. Pull requests should summarize motivation, list configuration overrides, and link to supporting experiment logs or issue IDs. Include screenshots or metric deltas when behavior changes, and call out any new directories collaborators must create.

## Configuration & Security Notes
Never commit generated `data/`, `wandb/`, or private checkpoints; add new ignores if another artifact directory appears. Store secrets (API keys, W&B tokens) in your shell environment, not in tracked files. When adding configs, mirror the structure in `config/arch/` and explain expected overrides so teammates can reproduce results.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Hierarchical Reasoning Model (HRM) repository - a novel recurrent architecture for complex reasoning tasks. HRM uses hierarchical processing with high-level and low-level modules to solve tasks like Sudoku, mazes, and ARC puzzles with minimal training data.

## Key Commands

### Training

```bash
# Single GPU training (e.g., Sudoku with 1k examples)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU training (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-aug-1000

# Training with custom config
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=<PATH> epochs=<N> lr=<LR>
```

### Dataset Building

```bash
# ARC datasets
python dataset/build_arc_dataset.py  # ARC-1 (960 examples)
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2

# Sudoku datasets
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000  # 1k examples

# Maze dataset
python dataset/build_maze_dataset.py  # 1000 examples
```

### Evaluation

```bash
# Evaluate trained model
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# For ARC evaluation, also use arc_eval.ipynb notebook
```

## Architecture Overview

### Core Model Structure

The model is implemented in `models/hrm/hrm_act_v1.py` with two main components:

1. **High-level Module (H)**: Slow, abstract planning - processes information over H_cycles
2. **Low-level Module (L)**: Rapid, detailed computations - runs L_cycles per H_cycle

Both modules use transformer-based blocks with self-attention and feed-forward networks. The model uses adaptive computation time (ACT) with Q-learning for dynamic halting.

### Key Configuration

Training configuration is managed through Hydra configs in `config/`:
- `cfg_pretrain.yaml`: Main training hyperparameters
- `arch/hrm_v1.yaml`: Model architecture parameters (H_cycles, L_cycles, hidden_size, etc.)

### Training Components

- **Optimizer**: AdamATan2 with separate learning rates for model weights and puzzle embeddings
- **Loss**: Configurable (softmax_cross_entropy or binary_cross_entropy)
- **Sparse Embeddings**: Uses CastedSparseEmbedding for efficient puzzle-specific embeddings
- **Mixed Precision**: Supports bfloat16 training

### Dataset Format

Puzzles are stored as structured data with:
- Input/output grids (variable sizes)
- Puzzle identifiers for sparse embeddings
- Train/validation/test splits
- Augmentation support (rotations, flips for geometric invariance)

## Dependencies

- PyTorch with CUDA support
- FlashAttention 2 or 3 (depending on GPU architecture)
- Weights & Biases for experiment tracking
- Hydra for configuration management

## Important Notes

- Small-sample learning shows ±2 points accuracy variance
- Use early stopping for Sudoku-Extreme to avoid late-stage overfitting
- The model achieves strong performance with only 27M parameters and 1000 training samples
- Evaluation metrics are tracked in W&B under `eval/exact_accuracy`
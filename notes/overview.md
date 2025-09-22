# HRM Training Pipeline Overview

This document provides a high-level overview of the Hierarchical Reasoning Model (HRM) training pipeline, starting from the `pretrain.py` entry point.

## Entry Point: pretrain.py

The training process for the Sudoku task begins with the following command:
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

## Program Flow

### 1. Configuration Loading (pretrain.py)
- **Hydra Configuration**: Uses Hydra framework to load configurations from YAML files
- **Base Config**: `config/cfg_pretrain.yaml` defines training hyperparameters
- **Architecture Config**: `config/arch/hrm_v1.yaml` specifies model architecture
- **Synchronized Config**: In distributed training, rank 0 loads config and broadcasts to all processes

### 2. Distributed Setup (pretrain.py: launch())
- Initializes PyTorch distributed training with NCCL backend
- Sets up process rank and world size for multi-GPU training
- Each GPU handles a portion of the global batch

### 3. Dataset Loading (puzzle_dataset.py)
- **PuzzleDataset**: Custom IterableDataset for loading puzzle data
- **Data Sharding**: Automatically splits data across GPUs
- **Memory Mapping**: Uses numpy memory mapping for efficient large dataset handling
- **Augmentation**: Supports data augmentation (rotations, flips) for geometric invariance

### 4. Model Initialization (models/hrm/hrm_act_v1.py)
- **HierarchicalReasoningModel**: Main model with two-level hierarchy
  - High-level module (H): Slow, abstract planning
  - Low-level module (L): Rapid, detailed computations
- **Sparse Embeddings**: Puzzle-specific embeddings for each training example
- **Loss Head**: ACTLossHead wrapper for computing losses

### 5. Optimizer Setup (pretrain.py: create_model())
- **Two Optimizers**:
  1. AdamATan2 for model parameters
  2. CastedSparseEmbeddingSignSGD for puzzle embeddings
- **Learning Rate Scheduling**: Cosine schedule with warmup

### 6. Training Loop (pretrain.py: launch())
```
For each epoch:
  → Load batches from training dataset
  → Forward pass through model
  → Compute losses (language modeling + Q-learning)
  → Backward propagation
  → All-reduce gradients across GPUs
  → Update parameters
  → Log metrics to Weights & Biases

  Every eval_interval epochs:
    → Run evaluation on test set
    → Save checkpoint if configured
```

### 7. Evaluation (pretrain.py: evaluate())
- Runs model in inference mode on test dataset
- Computes accuracy metrics without gradient computation
- Aggregates results across all GPUs
- Logs evaluation metrics to W&B

### 8. Checkpointing (pretrain.py: save_train_state())
- Saves model state dictionary at specified intervals
- Stores configuration and code for reproducibility
- Supports resuming training from checkpoints

## Key Components

### Model Architecture
- **Transformer Blocks**: Self-attention + feed-forward networks
- **Adaptive Computation Time (ACT)**: Dynamic halting with Q-learning
- **Hierarchical Processing**: H_cycles and L_cycles for reasoning depth

### Loss Computation
- **Language Modeling Loss**: Cross-entropy on output predictions
- **Q-Learning Losses**: For learning when to halt computation
- **Gradient Accumulation**: Handles large effective batch sizes

### Distributed Training
- **Data Parallel**: Each GPU processes different data
- **Gradient Synchronization**: All-reduce operation across GPUs
- **Parameter Broadcasting**: Initial model parameters from rank 0

### Monitoring
- **Weights & Biases**: Real-time metrics tracking
- **Progress Bar**: TQDM for training progress visualization
- **Metrics**: Accuracy, loss, Q-values, computation steps

## File Dependencies

```
pretrain.py
├── config/
│   ├── cfg_pretrain.yaml
│   └── arch/hrm_v1.yaml
├── puzzle_dataset.py
├── models/
│   ├── hrm/hrm_act_v1.py
│   ├── losses.py
│   ├── sparse_embedding.py
│   └── layers.py
├── utils/functions.py
└── adam_atan2.py
```

## Next Steps

For detailed explanations of specific components:
- [Training Flow Details](training_flow.md) - Step-by-step training cycle
- [Model Architecture](model_architecture.md) - HRM model components
- [Dataset Pipeline](dataset_pipeline.md) - Data loading and preprocessing
- [Distributed Training](distributed_training.md) - Multi-GPU setup and synchronization
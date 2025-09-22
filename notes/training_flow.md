# Detailed Training Flow

This document provides an in-depth explanation of the training cycle in the HRM model, focusing on the Sudoku task.

## Training Cycle Components

### 1. Batch Processing (pretrain.py: train_batch())

Each training step processes one batch of data through the following stages:

#### a. Data Transfer to GPU
```python
batch = {k: v.cuda() for k, v in batch.items()}
```
- Moves input tensors to GPU memory
- Batch contains: `inputs`, `labels`, `puzzle_identifiers`

#### b. Carry Initialization
```python
if train_state.carry is None:
    train_state.carry = train_state.model.initial_carry(batch)
```
- **Carry**: Maintains state across computation steps
- Contains:
  - `z_H`: High-level module hidden states
  - `z_L`: Low-level module hidden states
  - `steps`: Current computation step count
  - `halted`: Binary mask indicating which examples have finished
  - `current_data`: Current batch data

#### c. Forward Pass
```python
train_state.carry, loss, metrics, _, _ = train_state.model(
    carry=train_state.carry,
    batch=batch,
    return_keys=[]
)
```

### 2. Forward Pass Details (hrm_act_v1.py)

The forward pass implements Adaptive Computation Time (ACT) with hierarchical reasoning:

#### a. Input Processing
1. **Token Embeddings**: Converts input tokens to embeddings
2. **Puzzle Embeddings**: Adds puzzle-specific learned embeddings
3. **Position Encodings**: Applies RoPE (Rotary Position Embeddings) or learned positions

#### b. Hierarchical Computation Loop
```
While not all examples halted AND steps < max_steps:
    For H_cycle in range(H_cycles):
        → Update high-level representation (H module)

        For L_cycle in range(L_cycles):
            → Update low-level representation (L module)

    → Compute halting probabilities (Q-learning)
    → Update halted mask
    → Increment steps
```

#### c. Output Generation
- **Logits**: Final predictions through language model head
- **Q-values**: Halting decisions (continue vs. stop)
- **Metrics**: Accuracy, steps taken, Q-learning statistics

### 3. Loss Computation (losses.py: ACTLossHead)

Three types of losses are computed:

#### a. Language Modeling Loss
```python
lm_loss = cross_entropy(logits, labels)
```
- Standard cross-entropy for sequence prediction
- Uses stablemax or softmax cross-entropy
- Ignores padding tokens (IGNORE_LABEL_ID = -100)

#### b. Q-Halt Loss
```python
q_halt_loss = binary_cross_entropy(q_halt_logits, is_correct)
```
- Trains model to halt when answer is correct
- Binary classification: should halt or continue?

#### c. Q-Continue Loss (Bootstrapping)
```python
q_continue_loss = binary_cross_entropy(q_continue_logits, target_q_continue)
```
- Helps bootstrap Q-learning
- Trains value estimates for continuing computation

### 4. Backward Propagation

#### a. Scaled Loss Backward
```python
((1 / global_batch_size) * loss).backward()
```
- Scales loss by global batch size for proper averaging
- Computes gradients through automatic differentiation

#### b. Gradient Synchronization (Multi-GPU)
```python
if world_size > 1:
    for param in train_state.model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad)
```
- All-reduce operation: sums gradients across all GPUs
- Ensures consistent parameter updates

### 5. Optimizer Step

#### a. Learning Rate Scheduling
```python
lr_this_step = cosine_schedule_with_warmup_lr_lambda(
    current_step=train_state.step,
    base_lr=base_lr,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=train_state.total_steps,
    min_ratio=config.lr_min_ratio
)
```
- Warmup phase: Linear increase from 0 to base_lr
- Cosine decay: Smooth decrease to min_lr

#### b. Parameter Updates
```python
for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
    for param_group in optim.param_groups:
        param_group['lr'] = lr_this_step
    optim.step()
    optim.zero_grad()
```

Two optimizers update different parameters:

1. **AdamATan2** (Model Parameters):
   - Uses adaptive learning rates per parameter
   - Betas: (0.9, 0.95) for momentum
   - Weight decay: 1.0 for regularization

2. **CastedSparseEmbeddingSignSGD** (Puzzle Embeddings):
   - Specialized for sparse embeddings
   - Sign-based SGD for memory efficiency
   - Separate learning rate and weight decay

### 6. Metrics Collection and Logging

#### a. Training Metrics
```python
reduced_metrics = {
    "train/accuracy": token_accuracy,
    "train/exact_accuracy": sequence_accuracy,
    "train/lm_loss": language_model_loss,
    "train/q_halt_loss": halting_loss,
    "train/steps": average_computation_steps,
    "train/lr": current_learning_rate
}
```

#### b. Metric Aggregation (Multi-GPU)
```python
dist.reduce(metric_values, dst=0)  # Gather to rank 0
```
- Only rank 0 logs to Weights & Biases
- Avoids duplicate logging

### 7. Evaluation Cycle (pretrain.py: evaluate())

Every `eval_interval` epochs:

#### a. Model in Eval Mode
```python
train_state.model.eval()
```
- Disables dropout
- Uses deterministic behavior

#### b. Inference Loop
```python
with torch.inference_mode():
    for set_name, batch, global_batch_size in eval_loader:
        # Forward pass without gradients
        carry, _, metrics, preds, all_finish = train_state.model(...)
```
- No gradient computation (memory efficient)
- Processes until all examples halt

#### c. Metric Computation
```python
eval_metrics = {
    "accuracy": average_token_accuracy,
    "exact_accuracy": percentage_puzzles_solved,
    "steps": average_steps_to_solution
}
```

### 8. Checkpointing

#### a. Model State Saving
```python
torch.save(train_state.model.state_dict(),
           f"step_{train_state.step}")
```

#### b. Configuration Backup
- Saves training config as YAML
- Copies model source code
- Logs to W&B for experiment tracking

## Training Dynamics

### Adaptive Computation Time (ACT)

The model learns to dynamically allocate computation:
- Easy puzzles: Halt early (fewer steps)
- Hard puzzles: Continue computing (more steps)
- Q-learning: Learns optimal halting policy

### Hierarchical Processing

- **H-module** (2 cycles default): Strategic planning
- **L-module** (2 cycles per H-cycle): Detailed execution
- Total steps: Variable based on difficulty

### Memory Efficiency

- **Mixed Precision**: Uses bfloat16 for forward pass
- **Gradient Accumulation**: Implicit through batch scaling
- **Memory Mapped Data**: Efficient large dataset handling

### Convergence Monitoring

Key indicators of training progress:
1. **exact_accuracy**: Percentage of completely solved puzzles
2. **steps**: Should decrease as model improves
3. **q_halt_accuracy**: Model's ability to know when it's correct
4. **loss curves**: Should show steady decrease

## Sudoku-Specific Considerations

For the Sudoku task with 1000 training examples:
- High variance in early training (±2 points accuracy)
- Convergence typically around 10,000-15,000 epochs
- Early stopping recommended to prevent overfitting
- Augmentation (1000x) crucial for generalization
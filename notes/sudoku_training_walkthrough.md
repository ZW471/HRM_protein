# Complete Sudoku Training Loop Walkthrough

This document provides a detailed, step-by-step walkthrough of the full training loop for the HRM model using Sudoku data as an example. We'll trace through a single training iteration, examining all classes and functions involved.

## Overview: Training Pipeline Flow

```
Dataset Building → Data Loading → Model Initialization → Training Loop → Evaluation → Checkpointing
```

## Part 1: Dataset Preparation

### 1.1 Building the Sudoku Dataset (`dataset/build_sudoku_dataset.py`)

Before training begins, we need to build our dataset:

```python
# Command: python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

**Key Steps:**
1. **Download raw data** from HuggingFace: `sapientinc/sudoku-extreme`
2. **Parse CSV files** containing puzzle strings:
   - Input: `"..3.2.6..9..3.5..1..18.64...."` (dots are blank cells)
   - Convert to numpy array: `[0,0,3,0,2,0,6,0,0,9,...]` (0 represents blank)
3. **Apply augmentations** (1000 variations per puzzle):
   - Digit permutation: swap digits consistently (e.g., all 1s→7, all 2s→3)
   - Row/column shuffling within 3x3 blocks
   - Transpose operation (50% probability)
4. **Save as memory-mapped arrays**:
   - `train__inputs.npy`: Shape (1,001,000, 81) - augmented puzzles
   - `train__labels.npy`: Shape (1,001,000, 81) - solutions
   - `train__puzzle_identifiers.npy`: Shape (1,001,000,) - original puzzle IDs
   - `train__puzzle_indices.npy`: Boundaries between puzzles
   - `train__group_indices.npy`: Groups of augmented versions

### 1.2 Dataset Metadata (`dataset/common.py:PuzzleDatasetMetadata`)

Generated metadata stored in `dataset.json`:
```python
{
    "vocab_size": 10,              # Digits 0-9
    "seq_len": 81,                 # 9x9 grid
    "num_puzzle_identifiers": 1000,  # Unique puzzles
    "total_groups": 1000,          # Original puzzles before augmentation
    "mean_puzzle_examples": 1001,  # Including augmentations
    "sets": ["train"],             # Available splits
    "pad_id": 0,                   # Padding token
    "blank_identifier_id": -1,     # Invalid puzzle marker
    "ignore_label_id": 0           # Skip in loss calculation
}
```

## Part 2: Model Initialization

### 2.1 Configuration Loading (`pretrain.py:load_synced_config`)

```python
config = PretrainConfig(
    arch=ArchConfig(
        name="hrm@models/hrm/hrm_act_v1.py:HierarchicalReasoningModel_ACTV1",
        loss=LossConfig(name="losses@models/losses.py:ACTLossHead")
    ),
    data_path="data/sudoku-extreme-1k-aug-1000",
    global_batch_size=384,
    epochs=20000,
    lr=7e-5,
    lr_min_ratio=0.1,
    lr_warmup_steps=100,
    weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    puzzle_emb_lr=7e-5,
    puzzle_emb_weight_decay=1.0,
    eval_interval=2000
)
```

### 2.2 DataLoader Creation (`pretrain.py:create_dataloader`)

```python
train_loader, train_metadata = create_dataloader(
    config=config,
    split="train",
    test_set_mode=False,
    epochs_per_iter=2000,  # Batches 2000 epochs together
    global_batch_size=384,
    rank=0,  # GPU ID
    world_size=1  # Number of GPUs
)
```

**Inside `PuzzleDataset.__init__`:**
- Loads metadata from `dataset.json`
- Calculates local batch size: `384 / 1 = 384` (single GPU)
- Prepares memory-mapped access to `.npy` files

### 2.3 Model Creation (`pretrain.py:create_model`)

```python
# Step 1: Initialize the HRM model
model = HierarchicalReasoningModel_ACTV1({
    "batch_size": 384,
    "vocab_size": 10,
    "seq_len": 81,
    "num_puzzle_identifiers": 1000,
    "causal": False,  # Non-autoregressive

    # Architecture parameters (from config/arch/hrm_v1.yaml)
    "H_cycles": 2,
    "L_cycles": 2,
    "hidden_size": 512,
    "num_heads": 8,
    "num_layers": 3,
    "ffn_hidden_size": 2048,
    "max_steps": 32
})

# Step 2: Wrap with loss computation head
model = ACTLossHead(model, lm_loss_weight=1.0, q_halt_weight=0.1)

# Step 3: Compile for optimization (unless disabled)
model = torch.compile(model, dynamic=False)
```

### 2.4 Optimizer Setup (`pretrain.py:create_model`)

Two optimizers for different parameter groups:

```python
# 1. Sparse Embedding Optimizer (for puzzle-specific embeddings)
puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
    model.model.puzzle_emb.buffers(),  # Sparse embedding buffers
    lr=0,  # Set by scheduler
    weight_decay=1.0
)

# 2. Main Model Optimizer (for all other parameters)
model_optimizer = AdamATan2(
    model.parameters(),
    lr=0,  # Set by scheduler
    weight_decay=1.0,
    betas=(0.9, 0.95)
)
```

The sparse embedding optimizer uses a specialized approach for memory-efficient training of puzzle-specific embeddings. For a detailed explanation of how `CastedSparseEmbedding` works, why it's important, and how its optimization differs from standard embeddings, see [**casted_sparse_embedding_explained.md**](./casted_sparse_embedding_explained.md).

The key insight is that the sparse embedding optimizer only touches the rows that appear in the current batch and applies sign-SGD style updates, avoiding dense gradient materialization for the entire embedding table. This keeps memory use and cross-device communication low while still letting the global scheduler control the effective learning rate.

## Part 3: Training Loop Execution

### 3.1 Main Training Loop (`pretrain.py:launch`)

```python
for iter_id in range(10):  # 20000 epochs / 2000 eval_interval
    # Train for 2000 epochs
    model.train()
    for set_name, batch, global_batch_size in train_loader:
        metrics = train_batch(config, train_state, batch, global_batch_size)
        wandb.log(metrics, step=train_state.step)

    # Evaluate
    model.eval()
    eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata)

    # Checkpoint
    if checkpoint_every_eval:
        save_train_state(config, train_state)
```

### 3.2 Data Loading Iteration (`puzzle_dataset.py:PuzzleDataset._iter_train`)

For each batch, the dataset:

```python
# 1. Shuffle puzzle groups for this epoch
rng = np.random.Generator(seed=config.seed + epoch)
group_order = rng.permutation(1000)  # Shuffle 1000 puzzle groups

# 2. Sample a batch (via _sample_batch function)
while current_size < 384:
    # Pick a random group
    group_id = group_order[start_index]

    # Pick random augmented version from that group
    puzzle_id = rng.integers(group_start, group_end)

    # Add examples from this puzzle to batch
    batch.append(puzzle_examples)
    start_index += 1

# 3. Collate batch data
batch = {
    "inputs": np.array([[0,0,3,0,2,0,...], ...]),  # Shape: (384, 81)
    "labels": np.array([[4,8,3,9,2,1,...], ...]),  # Shape: (384, 81)
    "puzzle_identifiers": np.array([42, 42, 156, ...])  # Shape: (384,)
}

# 4. Convert to PyTorch tensors and move to GPU
batch = {k: torch.from_numpy(v).cuda() for k, v in batch.items()}
```

## Part 4: Forward Pass

### 4.1 Batch Processing (`pretrain.py:train_batch`)

```python
def train_batch(config, train_state, batch, global_batch_size):
    train_state.step += 1

    # Move batch to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # Initialize carry (hidden states) if first step
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    # Forward pass through model
    carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry,
        batch=batch,
        return_keys=[]
    )

    # Update carry for next iteration
    train_state.carry = carry
```

### 4.2 Model Forward Pass (`models/hrm/hrm_act_v1.py:HierarchicalReasoningModel_ACTV1.forward`)

The forward pass implements Adaptive Computation Time with hierarchical processing:

```python
def forward(self, carry, batch):
    # 1. Extract inputs
    inputs = batch["inputs"]  # Shape: (384, 81)
    labels = batch["labels"]  # Shape: (384, 81)
    puzzle_ids = batch["puzzle_identifiers"]  # Shape: (384,)

    # 2. Embed inputs
    x = self.token_emb(inputs)  # Token embeddings
    x += self.puzzle_emb(puzzle_ids)  # Add puzzle-specific embeddings
    x += self.pos_emb  # Add position embeddings

    # 3. Initialize or use carry
    z_H = carry["z_H"] if carry else zeros(384, 81, 512)  # High-level state
    z_L = carry["z_L"] if carry else zeros(384, 81, 512)  # Low-level state
    steps = carry["steps"] if carry else 0
    halted = carry["halted"] if carry else zeros(384, dtype=bool)

    # 4. Adaptive computation loop
    while not all(halted) and steps < max_steps:
        # High-level processing (2 cycles)
        for h in range(H_cycles):
            z_H = self.H_blocks[h](z_H, x)  # Self-attention + FFN

        # Low-level processing (2 cycles per H cycle)
        for l in range(L_cycles):
            z_L = self.L_blocks[l](z_L, z_H)  # Self-attention + FFN

        # Compute outputs and halting decision
        logits = self.lm_head(z_L)  # Predictions
        q_halt = self.q_head(z_L)  # Should we halt?

        # Update halting mask based on Q-values
        should_halt = q_halt > threshold
        halted = halted | should_halt

        steps += 1

    # 5. Return updated carry and outputs
    new_carry = {
        "z_H": z_H,
        "z_L": z_L,
        "steps": steps,
        "halted": halted,
        "current_data": batch
    }

    return new_carry, logits, q_values, metrics
```

### 4.3 Loss Computation (`models/losses.py:ACTLossHead.forward`)

The loss head wraps the model and computes three losses:

```python
def forward(self, carry, batch):
    # 1. Get model outputs
    carry, logits, q_halt, q_continue = self.model(carry, batch)

    # 2. Language modeling loss (main task)
    lm_loss = cross_entropy(
        logits.view(-1, vocab_size),  # Shape: (384*81, 10)
        labels.view(-1),  # Shape: (384*81,)
        ignore_index=IGNORE_LABEL_ID
    )

    # 3. Q-halt loss (learn when to stop)
    predictions = logits.argmax(dim=-1)
    is_correct = (predictions == labels).all(dim=1)  # Per-puzzle accuracy

    q_halt_loss = binary_cross_entropy(
        q_halt,  # Model's confidence it should halt
        is_correct.float()  # Ground truth: halt if correct
    )

    # 4. Q-continue loss (value bootstrapping)
    with torch.no_grad():
        # Target: max Q-value from next step
        target_q = max(q_halt_next, q_continue_next)

    q_continue_loss = mse_loss(q_continue, target_q)

    # 5. Combine losses
    total_loss = (
        self.lm_loss_weight * lm_loss +
        self.q_halt_weight * q_halt_loss +
        self.q_continue_weight * q_continue_loss
    )

    # 6. Compute metrics
    metrics = {
        "lm_loss": lm_loss.item(),
        "q_halt_loss": q_halt_loss.item(),
        "accuracy": (predictions == labels).float().mean(),
        "exact_accuracy": is_correct.float().mean(),
        "steps": steps,
        "count": batch_size
    }

    return carry, total_loss, metrics, predictions, all_halted
```

## Part 5: Backward Pass and Optimization

### 5.1 Gradient Computation (`pretrain.py:train_batch`)

```python
# Scale loss by global batch size for proper averaging
scaled_loss = (1 / 384) * total_loss
scaled_loss.backward()  # Compute gradients via autograd
```

### 5.2 Gradient Synchronization (Multi-GPU)

When using multiple GPUs:
```python
if world_size > 1:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad)  # Sum gradients across GPUs
```

### 5.3 Learning Rate Scheduling (`pretrain.py:compute_lr`)

Cosine schedule with warmup:
```python
def cosine_schedule_with_warmup_lr_lambda(current_step):
    if current_step < 100:  # Warmup phase
        return 7e-5 * (current_step / 100)

    # Cosine decay
    progress = (current_step - 100) / (total_steps - 100)
    return 7e-5 * (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress)))
```

### 5.4 Parameter Updates (`pretrain.py:train_batch`)

```python
# Update learning rates
lr_this_step = compute_lr(7e-5, config, train_state)

# Apply to both optimizers
for optimizer in [puzzle_emb_optimizer, model_optimizer]:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    optimizer.step()  # Update parameters
    optimizer.zero_grad()  # Clear gradients for next iteration
```

**Inside Optimizers:**

1. **AdamATan2** (main model):
   ```python
   # Momentum and adaptive learning rates
   m = beta1 * m + (1 - beta1) * grad  # Momentum
   v = beta2 * v + (1 - beta2) * grad²  # Variance

   # ATan2 transformation for stability
   update = atan2(m, sqrt(v) + eps)
   param = param - lr * (update + weight_decay * param)
   ```

2. **CastedSparseEmbeddingSignSGD** (puzzle embeddings):
   ```python
   # Sign-based SGD for memory efficiency
   update = sign(grad) + weight_decay * embedding
   embedding = embedding - lr * update

   # Only update active embeddings (sparse)
   active_ids = batch["puzzle_identifiers"].unique()
   update_only(active_ids)
   ```

## Part 6: Metrics and Logging

### 6.1 Metric Collection (`pretrain.py:train_batch`)

After each batch:
```python
# Reduce metrics across GPUs if distributed
if world_size > 1:
    metric_values = torch.stack([metrics[k] for k in sorted(metrics.keys())])
    dist.reduce(metric_values, dst=0)  # Gather to rank 0

# Process and log (only on rank 0)
if rank == 0:
    reduced_metrics = {
        "train/lm_loss": lm_loss / 384,
        "train/accuracy": token_accuracy,  # Per-token accuracy
        "train/exact_accuracy": puzzle_solved_rate,  # Full puzzle accuracy
        "train/q_halt_loss": q_halt_loss,
        "train/steps": average_computation_steps,
        "train/lr": current_learning_rate
    }

    wandb.log(reduced_metrics, step=train_state.step)
```

## Part 7: Evaluation

### 7.1 Evaluation Loop (`pretrain.py:evaluate`)

Every 2000 epochs:
```python
def evaluate(config, train_state, eval_loader, eval_metadata):
    model.eval()  # Disable dropout, use deterministic behavior

    with torch.inference_mode():  # No gradient tracking
        for set_name, batch, global_batch_size in eval_loader:
            # Initialize fresh carry for each batch
            carry = model.initial_carry(batch)

            # Run until all examples halt
            while True:
                carry, _, metrics, preds, all_finish = model(carry, batch)
                if all_finish:
                    break

            # Accumulate metrics
            aggregate_metrics(metrics)

    return compute_final_metrics()
```

### 7.2 Test Dataset Iteration (`puzzle_dataset.py:PuzzleDataset._iter_test`)

Unlike training, test iteration is deterministic:
```python
# Sequential iteration through all test examples
for start in range(0, total_examples, batch_size):
    batch = dataset[start:start+batch_size]
    yield "test", batch, batch_size
```

## Part 8: Checkpointing

### 8.1 Model Saving (`pretrain.py:save_train_state`)

```python
def save_train_state(config, train_state):
    checkpoint_path = f"checkpoints/{project_name}/{run_name}"

    # Save model weights
    torch.save(
        train_state.model.state_dict(),
        f"{checkpoint_path}/step_{train_state.step}"
    )

    # Save predictions (optional)
    if config.eval_save_outputs:
        torch.save(predictions, f"{checkpoint_path}/step_{step}_preds.pt")
```

### 8.2 Code and Config Backup (`pretrain.py:save_code_and_config`)

```python
# Copy model source files
shutil.copy("models/hrm/hrm_act_v1.py", checkpoint_path)
shutil.copy("models/losses.py", checkpoint_path)

# Save configuration
with open(f"{checkpoint_path}/all_config.yaml", "w") as f:
    yaml.dump(config.model_dump(), f)

# Log to Weights & Biases
wandb.run.log_code(checkpoint_path)
```

## Example: Processing One Sudoku Puzzle

Let's trace a single Sudoku puzzle through the entire pipeline:

### Input Puzzle
```
Original string: "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
```

### Step 1: Encoding
```python
inputs = [0,0,3,0,2,0,6,0,0,9,0,0,3,0,5,0,0,1,...]  # 0 = blank
labels = [4,8,3,9,2,1,6,5,7,9,6,7,3,4,5,8,2,1,...]  # Solution
puzzle_id = 42  # Unique identifier for this puzzle
```

### Step 2: Embedding
```python
x = token_emb([0,0,3,0,2,0,6,0,0,...])  # Shape: (81, 512)
x += puzzle_emb(42)  # Add puzzle-specific embedding
x += pos_emb[0:81]  # Add position information
```

### Step 3: Hierarchical Processing (Adaptive Steps)
```python
step 1: H-cycles → L-cycles → Check accuracy (60% correct) → Continue
step 2: H-cycles → L-cycles → Check accuracy (75% correct) → Continue
step 3: H-cycles → L-cycles → Check accuracy (85% correct) → Continue
step 4: H-cycles → L-cycles → Check accuracy (100% correct) → Halt!
```

### Step 4: Output
```python
predictions = [4,8,3,9,2,1,6,5,7,9,6,7,3,4,5,8,2,1,...]
exact_match = True  # Puzzle solved correctly
computation_steps = 4  # Took 4 iterations
```

### Step 5: Loss Computation
```python
lm_loss = 0.02  # Low, as predictions match labels
q_halt_loss = 0.01  # Low, as model correctly halted when solved
total_loss = 1.0 * 0.02 + 0.1 * 0.01 = 0.021
```

### Step 6: Gradient Update
```python
loss.backward()  # Compute gradients
optimizer.step()  # Update model weights
```

## Training Progress Over Time

### Early Training (Epochs 1-1000)
- Model learns basic Sudoku constraints
- Accuracy: ~40-60%
- Steps: Uses maximum (32) for most puzzles
- High variance between batches

### Mid Training (Epochs 1000-10000)
- Model develops solving strategies
- Accuracy: ~70-85%
- Steps: Begins to halt early for easy puzzles
- Q-learning improves halting decisions

### Late Training (Epochs 10000-20000)
- Model refines solutions
- Accuracy: ~85-95% (with variance)
- Steps: Efficient computation (4-8 steps average)
- Risk of overfitting to training puzzles

### Final Performance
- Test accuracy: ~87% (±2% variance)
- Average steps: 6-7
- Generalizes to unseen Sudoku puzzles
- Best checkpoint typically around epoch 15000

## Key Design Decisions

1. **Hierarchical Architecture**: H-module for strategy, L-module for execution
2. **Adaptive Computation**: Variable steps based on puzzle difficulty
3. **Sparse Embeddings**: Puzzle-specific parameters for memorization
4. **Augmentation**: 1000x data multiplication for generalization
5. **Q-Learning**: Learn optimal halting policy
6. **Mixed Precision**: Memory efficiency with bfloat16
7. **Distributed Training**: Scale across multiple GPUs

## Summary

The training loop orchestrates:
1. **Data flow**: Dataset → Batches → GPU → Model → Loss → Gradients → Updates
2. **Computation**: Hierarchical reasoning with adaptive steps
3. **Learning**: Three objectives (task, halting, value estimation)
4. **Optimization**: Dual optimizers with cosine scheduling
5. **Evaluation**: Periodic validation and checkpointing
6. **Monitoring**: Comprehensive metrics via Weights & Biases

This design enables the model to learn complex reasoning with minimal data (1000 examples) while maintaining computational efficiency through adaptive processing.
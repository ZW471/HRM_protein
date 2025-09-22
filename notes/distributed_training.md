# Distributed Training

This document explains the multi-GPU training setup for the HRM model using PyTorch's distributed training capabilities.

## Launch Command

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

### Command Breakdown
- **OMP_NUM_THREADS=8**: OpenMP thread limit per process
- **torchrun**: PyTorch's distributed launcher
- **--nproc-per-node 8**: 8 GPU processes per node
- Config parameters passed via Hydra

## Distributed Initialization

### 1. Process Group Setup (pretrain.py)

```python
if "LOCAL_RANK" in os.environ:
    # Initialize NCCL backend for GPU communication
    dist.init_process_group(backend="nccl")

    RANK = dist.get_rank()           # Process ID (0-7 for 8 GPUs)
    WORLD_SIZE = dist.get_world_size()  # Total processes (8)

    # Set CUDA device for this process
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

### 2. Environment Variables

torchrun automatically sets:
- **LOCAL_RANK**: GPU index on current node (0-7)
- **RANK**: Global process rank
- **WORLD_SIZE**: Total number of processes
- **MASTER_ADDR**: Coordination server address
- **MASTER_PORT**: Coordination server port

## Configuration Synchronization

### 1. Broadcast Config from Rank 0

```python
def load_synced_config(hydra_config, rank, world_size):
    objects = [None]

    if rank == 0:
        # Only rank 0 loads and processes config
        config = PretrainConfig(**hydra_config)
        config.run_name = generate_name()  # Random name
        objects = [config]

    if world_size > 1:
        # Broadcast config to all ranks
        dist.broadcast_object_list(objects, src=0)

    return objects[0]
```

**Benefits:**
- Consistent configuration across all GPUs
- Single source of truth (rank 0)
- Automatic serialization/deserialization

## Data Parallelism

### 1. Dataset Sharding

```python
class PuzzleDataset(IterableDataset):
    def __init__(self, config, split):
        self.rank = config.rank
        self.num_replicas = config.num_replicas
        self.local_batch_size = config.global_batch_size // self.num_replicas
```

#### Training Data Distribution
```python
def _iter_train(self):
    # Each rank processes different slice
    batch_indices = all_indices[
        self.rank * self.local_batch_size:
        (self.rank + 1) * self.local_batch_size
    ]
```

#### Test Data Distribution
```python
def _iter_test(self):
    # Deterministic split for evaluation
    local_start = start_index + self.rank * self.local_batch_size
    local_end = min(
        start_index + (self.rank + 1) * self.local_batch_size,
        total_examples
    )
```

### 2. Batch Size Scaling

```
Global Batch Size: 768 (example)
World Size: 8 GPUs
Local Batch Size per GPU: 768 / 8 = 96
```

## Model Initialization

### 1. Parameter Broadcasting

```python
def create_model(config, metadata, world_size):
    # Create model on GPU
    with torch.device("cuda"):
        model = HierarchicalReasoningModel(config)

    # Broadcast initial parameters from rank 0
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)
```

**Purpose:**
- Ensures identical initial weights
- Prevents divergence during training
- Reproducible results

### 2. Distributed Optimizer Setup

```python
# Sparse embedding optimizer with distributed support
optimizer_sparse = CastedSparseEmbeddingSignSGD_Distributed(
    model.puzzle_emb.buffers(),
    lr=config.puzzle_emb_lr,
    weight_decay=config.puzzle_emb_weight_decay,
    world_size=world_size  # Aware of distribution
)
```

## Gradient Synchronization

### 1. All-Reduce Operation

```python
def train_batch(config, train_state, batch, rank, world_size):
    # Forward pass (local to each GPU)
    loss, metrics = model(batch)

    # Backward pass (local gradients)
    loss.backward()

    # Synchronize gradients across GPUs
    if world_size > 1:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
```

### 2. All-Reduce Mechanics

```
GPU 0: grad_0 ─┐
GPU 1: grad_1 ─┼─→ Sum: Σ(grad_i) ─→ Each GPU gets sum
GPU 2: grad_2 ─┤
...           ─┘
```

**NCCL Optimizations:**
- Ring algorithm for bandwidth efficiency
- Overlapping communication with computation
- GPU-direct communication (bypasses CPU)

## Metrics Aggregation

### 1. Training Metrics

```python
def train_batch(...):
    # Collect metrics locally
    metrics = {"loss": loss, "accuracy": acc, "steps": steps}

    # Reduce to rank 0
    if world_size > 1:
        metric_tensor = torch.stack(list(metrics.values()))
        dist.reduce(metric_tensor, dst=0)  # Sum to rank 0

    # Only rank 0 logs
    if rank == 0:
        wandb.log(metrics, step=train_state.step)
```

### 2. Evaluation Metrics

```python
def evaluate(train_state, eval_loader, rank, world_size):
    # Each rank evaluates its portion
    local_metrics = compute_metrics(predictions, labels)

    # Aggregate across all ranks
    if world_size > 1:
        dist.reduce(local_metrics, dst=0)

    if rank == 0:
        # Average or sum as appropriate
        final_metrics = process_aggregated_metrics(local_metrics)
        return final_metrics
```

## Sparse Embedding Distribution

### 1. Distributed Sparse SGD

```python
class CastedSparseEmbeddingSignSGD_Distributed:
    def step(self):
        # Update only active embeddings
        active_indices = get_active_indices()

        # Local update
        embeddings[active_indices] -= lr * sign(gradients)

        # Synchronize active embeddings
        if self.world_size > 1:
            all_indices = all_gather(active_indices)
            all_updates = all_gather(embedding_updates)
            apply_updates(all_indices, all_updates)
```

**Benefits:**
- Only sync used embeddings
- Memory efficient for large embedding tables
- Handles puzzle-specific parameters

## Checkpointing

### 1. Single-Rank Saving

```python
def save_train_state(config, train_state):
    # Only rank 0 saves checkpoints
    if rank == 0:
        torch.save(
            train_state.model.state_dict(),
            f"checkpoint_step_{train_state.step}.pt"
        )
```

### 2. Distributed Checkpoint (Optional)

```python
# For very large models
if use_distributed_checkpoint:
    dist.checkpoint.save_state_dict(
        state_dict=model.state_dict(),
        storage_writer=dist.checkpoint.FileSystemWriter(path),
        planner=dist.checkpoint.DefaultSavePlanner()
    )
```

## Performance Optimizations

### 1. NCCL Configuration

```bash
# Environment variables for optimization
export NCCL_DEBUG=INFO          # Debugging
export NCCL_IB_DISABLE=0         # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # Network interface
export NCCL_P2P_LEVEL=NVL        # NVLink usage
```

### 2. Gradient Accumulation (Implicit)

```python
# Scale loss by world size for proper averaging
scaled_loss = loss / config.global_batch_size
scaled_loss.backward()

# After all-reduce, gradients represent global average
```

### 3. Mixed Precision Training

```python
# Automatic mixed precision (if enabled)
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

## Distributed Training Flow

### 1. Initialization Phase
```
1. torchrun launches 8 processes
2. Each process initializes NCCL
3. Rank 0 creates config, broadcasts to others
4. All ranks create identical model
5. Rank 0 broadcasts initial parameters
```

### 2. Training Loop
```
For each batch:
1. Each GPU loads different data slice
2. Forward pass (parallel, independent)
3. Backward pass (parallel, independent)
4. All-reduce gradients (synchronized)
5. Optimizer step (parallel, identical)
6. Rank 0 logs metrics
```

### 3. Evaluation
```
1. Each GPU evaluates its test portion
2. Metrics reduced to rank 0
3. Rank 0 computes final metrics
4. Rank 0 logs to W&B
```

## Common Issues and Solutions

### 1. Hanging Training

**Symptoms:** Training stops progressing
**Causes:**
- Imbalanced batch sizes
- One rank crashes silently
- Network issues

**Solutions:**
```python
# Add timeout to operations
dist.all_reduce(tensor, timeout=timedelta(seconds=60))

# Check all ranks reach same point
dist.barrier()
```

### 2. Memory Imbalance

**Issue:** Rank 0 uses more memory (logging, checkpointing)

**Solution:**
```python
# Offload non-critical operations
if rank == 0:
    # Use CPU for metrics processing
    metrics = {k: v.cpu() for k, v in metrics.items()}
```

### 3. Reproducibility

**Ensure consistent results:**
```python
# Synchronized random seeds
torch.manual_seed(config.seed + rank)
np.random.seed(config.seed + rank)

# Deterministic operations
torch.backends.cudnn.deterministic = True
```

## Scaling Efficiency

### 1. Communication Overhead

```
Computation Time: O(model_size / world_size)
Communication Time: O(model_size)

Efficiency = Computation / (Computation + Communication)
```

### 2. Optimal Batch Size

```python
# Scale batch size with GPUs
optimal_global_batch = base_batch * sqrt(world_size)

# Adjust learning rate accordingly
scaled_lr = base_lr * sqrt(world_size)
```

### 3. Performance Metrics

Monitor scaling efficiency:
- **Throughput**: Examples/second
- **GPU Utilization**: nvidia-smi
- **Communication Time**: NCCL profiling

## Multi-Node Training (Advanced)

For training across multiple machines:

```bash
# Node 0 (master)
torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
    --master-addr="10.0.0.1" --master-port=12345 \
    pretrain.py ...

# Node 1
torchrun --nproc-per-node=8 --nnodes=2 --node-rank=1 \
    --master-addr="10.0.0.1" --master-port=12345 \
    pretrain.py ...
```

## Best Practices

1. **Start Small**: Test with 2 GPUs before scaling
2. **Monitor Metrics**: Watch for communication bottlenecks
3. **Profile Performance**: Use PyTorch profiler
4. **Handle Failures**: Implement checkpoint recovery
5. **Optimize Batch Size**: Find sweet spot for hardware
6. **Use Fast Interconnect**: NVLink > PCIe for GPUs
7. **Debugging**: Set NCCL_DEBUG=INFO for issues
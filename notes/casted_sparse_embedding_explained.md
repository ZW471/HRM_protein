# Casted Sparse Embedding: Deep Dive

This document provides a comprehensive explanation of the `CastedSparseEmbedding` module and its custom optimizer `CastedSparseEmbeddingSignSGD_Distributed` used in the HRM model, implemented in `models/sparse_embedding.py`.

## Table of Contents
1. [Why Sparse Embeddings Matter](#why-sparse-embeddings-matter)
2. [The Problem with Standard Embeddings](#the-problem-with-standard-embeddings)
3. [Understanding "Casted" Embeddings](#understanding-casted-embeddings)
4. [How CastedSparseEmbedding Works](#how-castedsparseembedding-works)
5. [The Sign-SGD Optimizer](#the-sign-sgd-optimizer)
6. [Distributed Training Considerations](#distributed-training-considerations)
7. [Memory and Performance Analysis](#memory-and-performance-analysis)
8. [Code Walkthrough](#code-walkthrough)

## Why Sparse Embeddings Matter

In the HRM model, each puzzle has a unique identifier that maps to a learnable embedding vector. For Sudoku with 1000 training puzzles:

```python
num_puzzles = 1000
embedding_dim = 512
# Standard approach would require: 1000 × 512 × 4 bytes = 2MB
```

While 2MB seems small, consider:
- ARC dataset: 10,000+ unique puzzles → 20MB
- Future scaling: 100,000+ puzzles → 200MB
- **Key insight**: In any batch, we only use ~384 puzzle IDs out of 1000+

This sparsity pattern makes standard PyTorch embeddings inefficient.

## The Problem with Standard Embeddings

### Standard PyTorch Embedding (`nn.Embedding`)

```python
# Standard approach
embedding = nn.Embedding(num_puzzles, embedding_dim)
output = embedding(puzzle_ids)  # Shape: (batch_size, embedding_dim)
```

**Problems:**
1. **Dense gradients**: Computes gradients for entire embedding table
2. **Memory overhead**: Optimizer states (Adam momentum, variance) for all embeddings
3. **Communication cost**: All-reduce synchronizes full embedding gradients across GPUs
4. **Wasted computation**: Updates all embeddings even if only 1% were used

### Example Memory Breakdown

For 10,000 puzzles with Adam optimizer:
```python
embeddings: 10,000 × 512 × 4 bytes = 20MB
adam_momentum: 10,000 × 512 × 4 bytes = 20MB
adam_variance: 10,000 × 512 × 4 bytes = 20MB
gradients: 10,000 × 512 × 4 bytes = 20MB
Total: 80MB per GPU
```

## Understanding "Casted" Embeddings

The term "casted" refers to **dynamic type casting** between precisions:

### 1. Storage Precision (FP32)
```python
self.weights = nn.Buffer(
    torch.empty((num_embeddings, embedding_dim)),  # FP32 storage
    persistent=True
)
```
- Embeddings stored in full FP32 precision
- Ensures numerical stability for sparse updates

### 2. Computation Precision (BF16)
```python
def forward(self, inputs):
    return self.local_weights.to(self.cast_to)  # Cast to bfloat16
```
- Forward pass uses bfloat16 for efficiency
- Reduces memory bandwidth and computation cost
- Compatible with mixed-precision training

### 3. Why This Matters
```python
# Memory savings in forward pass:
FP32: batch_size × embedding_dim × 4 bytes
BF16: batch_size × embedding_dim × 2 bytes  # 50% reduction
```

## How CastedSparseEmbedding Works

### Architecture Overview (`models/sparse_embedding.py:CastedSparseEmbedding`)

```python
class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std, cast_to):
        # 1. Full embedding table (persistent)
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim))),
            persistent=True
        )

        # 2. Local working copy (non-persistent)
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, embedding_dim, requires_grad=True),
            persistent=False
        )

        # 3. Track which embeddings are active
        self.local_ids = nn.Buffer(
            torch.zeros(batch_size, dtype=torch.int32),
            persistent=False
        )
```

### Forward Pass Mechanism

```python
def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    if not self.training:
        # Test mode: Direct lookup, no gradient tracking
        return self.weights[inputs].to(self.cast_to)

    # Training mode: Create local copy for gradient computation
    with torch.no_grad():
        # Copy only the embeddings we need
        self.local_weights.copy_(self.weights[inputs])
        # Remember which embeddings these are
        self.local_ids.copy_(inputs)

    # Return casted local copy (with gradient tracking)
    return self.local_weights.to(self.cast_to)
```

### Key Design Decisions

1. **Local Working Copy**
   - Only batch_size embeddings have gradients
   - Avoids materializing gradients for entire table

2. **Buffer vs Parameter**
   - `weights` is a Buffer, not Parameter
   - Prevents standard optimizers from updating it
   - Requires custom optimizer

3. **Persistent vs Non-Persistent**
   - `weights`: Persistent (saved in checkpoint)
   - `local_weights`: Non-persistent (temporary)
   - `local_ids`: Non-persistent (recreated each batch)

## The Sign-SGD Optimizer

### Why Sign-SGD? (`models/sparse_embedding.py:CastedSparseEmbeddingSignSGD_Distributed`)

Sign-SGD uses only the sign of gradients:
```python
update = -lr * sign(gradient)
```

**Advantages for Sparse Embeddings:**
1. **Memory efficient**: No momentum/variance states
2. **Communication efficient**: 1 bit per gradient value
3. **Robust to scale**: Sign normalizes gradient magnitudes
4. **Empirically effective**: Works well for embedding tables

### Optimizer Implementation

```python
class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def step(self):
        # 1. Extract components
        local_weights_grad = ...  # Gradients from current batch
        local_ids = ...           # Which embeddings were used
        weights = ...             # Full embedding table

        # 2. Apply sparse update
        _sparse_emb_signsgd_dist(
            local_weights_grad, local_ids, weights,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            world_size=group["world_size"]
        )
```

### The Update Function

```python
def _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, lr, weight_decay, world_size):
    # 1. Gather gradients from all GPUs
    if world_size > 1:
        all_weights_grad = all_gather(local_weights_grad)
        all_ids = all_gather(local_ids)

    # 2. Find unique embedding IDs and accumulate gradients
    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = scatter_add(all_weights_grad, indices=inv)

    # 3. Apply Sign-SGD with weight decay
    p = weights[grad_ids]  # Get current values
    p.mul_(1.0 - lr * weight_decay)  # Weight decay
    p.add_(torch.sign(grad), alpha=-lr)  # Sign-SGD update

    # 4. Write back updated embeddings
    weights[grad_ids] = p
```

## Distributed Training Considerations

### The Synchronization Challenge

With multiple GPUs, each processes different batches:
```
GPU 0: puzzle_ids = [5, 12, 87, ...]
GPU 1: puzzle_ids = [5, 33, 91, ...]  # Note: 5 appears on both!
GPU 2: puzzle_ids = [12, 45, 87, ...] # 12 and 87 overlap with GPU 0
```

### Solution: All-Gather and Accumulate

```python
# Step 1: Each GPU computes local gradients
local_grad[5] = gradient_from_gpu_0_batch
local_grad[12] = gradient_from_gpu_0_batch

# Step 2: All-gather across GPUs
all_gradients = [gpu0_grads, gpu1_grads, gpu2_grads]
all_ids = [gpu0_ids, gpu1_ids, gpu2_ids]

# Step 3: Accumulate gradients for duplicate IDs
final_grad[5] = gpu0_grad[5] + gpu1_grad[5]  # Sum duplicates
final_grad[12] = gpu0_grad[12] + gpu2_grad[12]

# Step 4: Apply update to unique embeddings
weights[5] -= lr * sign(final_grad[5])
```

### Communication Efficiency

**Standard All-Reduce:**
```python
# Communicates: num_embeddings × embedding_dim × 4 bytes
# For 10,000 embeddings: 10,000 × 512 × 4 = 20MB per step
```

**Sparse All-Gather:**
```python
# Communicates: batch_size × embedding_dim × 4 bytes
# For batch_size=384: 384 × 512 × 4 = 0.8MB per step
# 25x reduction!
```

## Memory and Performance Analysis

### Memory Comparison

| Component | Standard Embedding | Casted Sparse Embedding |
|-----------|-------------------|------------------------|
| Weights | N × D × 4 bytes | N × D × 4 bytes |
| Gradients | N × D × 4 bytes | B × D × 4 bytes |
| Adam States | 2 × N × D × 4 bytes | None (Sign-SGD) |
| Forward Pass | B × D × 4 bytes | B × D × 2 bytes (BF16) |
| **Total** | 4ND + BD | ND + 0.5BD |

Where:
- N = num_embeddings (1000 for Sudoku)
- D = embedding_dim (512)
- B = batch_size (384)

**Concrete Example (Sudoku):**
- Standard: 4×1000×512×4 = 8MB + 0.8MB = 8.8MB
- Casted Sparse: 1000×512×4 + 384×512×2 = 2MB + 0.4MB = 2.4MB
- **Savings: 73%**

### Performance Benefits

1. **Reduced Memory Pressure**
   - Smaller working set fits in cache
   - Less memory bandwidth required

2. **Faster Updates**
   - Only update active embeddings
   - Sign operation is cheap

3. **Scalability**
   - Constant memory for optimizer
   - Scales to millions of embeddings

## Code Walkthrough

### Initialization Example

```python
# In model initialization (hrm_act_v1.py)
self.puzzle_emb = CastedSparseEmbedding(
    num_embeddings=1000,      # Total puzzles
    embedding_dim=512,        # Hidden dimension
    batch_size=384,          # Maximum batch size
    init_std=0.02,           # Initialization scale
    cast_to=torch.bfloat16   # Mixed precision
)
```

### Training Step Example

```python
# 1. Forward pass
puzzle_ids = torch.tensor([42, 17, 99, ...])  # Batch of puzzle IDs
embeddings = self.puzzle_emb(puzzle_ids)       # Returns BF16 embeddings

# 2. Use in model
x = token_embeddings + embeddings  # Add to token embeddings

# 3. Backward pass (automatic)
loss.backward()  # Gradients computed on local_weights

# 4. Optimizer step
optimizer.step()  # Custom optimizer updates only active embeddings
```

### Gradient Flow

```
Forward:
puzzle_ids → weights[ids] → local_weights → cast to BF16 → model

Backward:
model gradient → BF16 to FP32 → local_weights.grad → optimizer

Optimizer:
local_weights.grad + local_ids → gather across GPUs →
accumulate → sign → update weights[unique_ids]
```

## Summary

The `CastedSparseEmbedding` system achieves:

1. **Memory Efficiency**: 73% reduction vs standard embeddings
2. **Communication Efficiency**: 25x less data transfer in distributed training
3. **Computation Efficiency**: Only updates active embeddings
4. **Numerical Stability**: FP32 storage with BF16 computation
5. **Scalability**: Constant optimizer memory regardless of embedding table size

This design is crucial for the HRM model's ability to:
- Handle thousands of unique puzzles efficiently
- Scale to larger datasets without memory explosion
- Train quickly on multiple GPUs with minimal communication overhead
- Maintain puzzle-specific knowledge while keeping memory footprint small

The combination of sparse updates, sign-based optimization, and mixed-precision computation makes this approach both theoretically elegant and practically effective for learning puzzle-specific representations in the HRM architecture.
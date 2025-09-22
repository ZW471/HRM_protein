# HRM Model Architecture

This document details the components and structure of the Hierarchical Reasoning Model (HRM) with Adaptive Computation Time (ACT).

## Model Overview

The HRM is a recurrent architecture with two hierarchical modules that process information at different levels of abstraction. The model dynamically allocates computation time based on problem difficulty using Q-learning.

## Core Components

### 1. HierarchicalReasoningModel_ACTV1 (Main Wrapper)

The top-level model class that implements ACT logic:

```python
class HierarchicalReasoningModel_ACTV1(nn.Module):
    - inner: HierarchicalReasoningModel_ACTV1_Inner
    - Manages carry state across computation steps
    - Implements halting decisions via Q-learning
```

**Key Responsibilities:**
- Carry state management
- Halting decision logic
- Q-value computation for ACT
- Exploration during training

### 2. HierarchicalReasoningModel_ACTV1_Inner (Core Model)

The actual neural network architecture:

```python
class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    Components:
    - embed_tokens: Token embeddings
    - puzzle_emb: Sparse puzzle-specific embeddings
    - H_level: High-level reasoning module
    - L_level: Low-level reasoning module
    - lm_head: Language model output head
    - q_head: Q-value head for halting decisions
```

### 3. Embedding Components

#### a. Token Embeddings (CastedEmbedding)
- **Vocabulary Size**: Task-dependent (e.g., 10 for Sudoku digits 0-9)
- **Hidden Size**: 512 (default)
- **Initialization**: Scaled by 1/sqrt(hidden_size)
- **Casting**: Converts to bfloat16 for efficiency

#### b. Puzzle Embeddings (CastedSparseEmbedding)
- **Purpose**: Learns unique representations for each puzzle
- **Sparse Implementation**: Memory-efficient for large puzzle sets
- **Zero Initialization**: Starts neutral, learns during training
- **Dimension**: Same as hidden_size

#### c. Position Encodings
Two options supported:

1. **RoPE (Rotary Position Embeddings)**:
   - Rotation-based position encoding
   - Applied in attention mechanism
   - Base frequency: 10000.0

2. **Learned Positions**:
   - Trainable position embeddings
   - Added directly to token embeddings

### 4. Reasoning Modules

Both H and L modules share similar structure but operate at different abstraction levels:

#### a. HierarchicalReasoningModel_ACTV1ReasoningModule
```python
Structure:
- Input injection (residual connection)
- Multiple transformer blocks
- Post-norm architecture
```

#### b. HierarchicalReasoningModel_ACTV1Block (Transformer Block)
```python
Components:
1. Self-Attention Layer
   - Multi-head attention (8 heads default)
   - Non-causal (bidirectional)
   - RMS normalization

2. Feed-Forward Network (SwiGLU)
   - Gated linear unit with SiLU activation
   - Expansion factor: 4x
   - RMS normalization
```

### 5. Hierarchical Processing Flow

```
Initial State: z_H (high-level), z_L (low-level)

For H_cycle in [1, 2]:  # Default: 2 cycles
    For L_cycle in [1, 2]:  # Default: 2 cycles per H_cycle
        z_L = L_level(z_L, z_H + input_embeddings)

    z_H = H_level(z_H, z_L)
```

**Key Design Principles:**
- H-module updates less frequently (strategic planning)
- L-module updates more frequently (detailed execution)
- Information flows bidirectionally between levels

### 6. Output Heads

#### a. Language Model Head (lm_head)
- Linear projection: hidden_size → vocab_size
- No bias term
- Produces logits for token prediction

#### b. Q-Value Head (q_head)
- Linear projection: hidden_size → 2
- Two outputs:
  1. Q(halt): Value of stopping computation
  2. Q(continue): Value of continuing computation
- Initialized near zero for stable training

### 7. Carry State Management

The carry state maintains information across computation steps:

```python
HierarchicalReasoningModel_ACTV1Carry:
- inner_carry: Hidden states (z_H, z_L)
- steps: Computation steps taken
- halted: Binary mask of finished examples
- current_data: Current batch data
```

**Carry Flow:**
1. Initialize with empty/random states
2. Reset when starting new puzzles
3. Update after each forward pass
4. Detach gradients to prevent backprop through time

### 8. Adaptive Computation Time (ACT)

#### a. Halting Decision Logic
```python
During Training:
- Compute Q(halt) and Q(continue)
- Halt if Q(halt) > Q(continue)
- Add exploration noise (10% random halting)
- Enforce maximum steps limit (16 default)

During Evaluation:
- Always use maximum steps for consistency
```

#### b. Q-Learning Components
- **No Replay Buffer**: Uses large batch size as parallel environments
- **No Target Network**: Direct bootstrapping (similar to PQN)
- **Target Q-Value**: max(Q_halt, Q_continue) for next step

### 9. Model Configuration

Key hyperparameters from `arch/hrm_v1.yaml`:

```yaml
# Hierarchical cycles
H_cycles: 2        # High-level update frequency
L_cycles: 2        # Low-level updates per H-cycle

# Architecture
H_layers: 4        # Transformer blocks in H-module
L_layers: 4        # Transformer blocks in L-module

# Dimensions
hidden_size: 512   # Model dimension
num_heads: 8       # Attention heads
expansion: 4       # FFN expansion factor

# ACT parameters
halt_max_steps: 16              # Maximum computation steps
halt_exploration_prob: 0.1      # Exploration during training

# Training
forward_dtype: bfloat16        # Mixed precision
```

## Component Interactions

### 1. Input Processing Pipeline
```
Raw Input → Token Embeddings → Add Puzzle Embeddings → Add Position Encoding → Scale
```

### 2. Reasoning Pipeline
```
Input Embeddings → L-module (fast) ↔ H-module (slow) → Output Logits
```

### 3. Training Pipeline
```
Forward Pass → Compute Losses → Q-Learning Update → Gradient Backprop → Parameter Update
```

## Layer Components (models/layers.py)

### 1. Attention Layer
- **FlashAttention**: Optimized attention computation
- **Group Query Attention**: Supports different K/V head counts
- **RoPE Integration**: Position encoding in attention

### 2. SwiGLU Feed-Forward
- **Architecture**: gate * SiLU(input) * up_projection
- **Efficiency**: Better than standard FFN
- **Activation**: SiLU (Swish) function

### 3. RMS Normalization
- **Formula**: x / sqrt(mean(x²) + ε)
- **Advantage**: Simpler than LayerNorm
- **Location**: Post-norm architecture

### 4. CastedLinear/CastedEmbedding
- **Purpose**: Automatic dtype casting
- **Training**: Maintains weights in float32
- **Forward**: Casts to bfloat16
- **Gradient**: Computed in lower precision

## Sparse Embedding System (models/sparse_embedding.py)

### CastedSparseEmbedding
- **Sparse Storage**: Only stores active embeddings
- **Sign-SGD Optimizer**: Memory-efficient updates
- **Distributed Support**: Handles multi-GPU training
- **Dynamic Allocation**: Creates embeddings on-demand

**Advantages:**
- Memory efficient for large puzzle sets
- Fast sparse updates
- Supports millions of unique puzzles

## Model Size and Efficiency

### Parameter Count
- **Total Parameters**: ~27M
- **Breakdown**:
  - Token embeddings: vocab_size × 512
  - Transformer blocks: 8 blocks × ~3M each
  - Output heads: Negligible

### Memory Optimization
- **Mixed Precision**: bfloat16 forward pass
- **Gradient Checkpointing**: Optional (via torch.compile)
- **Sparse Embeddings**: Only active puzzles in memory

### Computational Efficiency
- **Dynamic Halting**: Saves computation on easy examples
- **Hierarchical Design**: Reduces redundant computation
- **Compiled Model**: torch.compile optimization

## Key Design Innovations

1. **Hierarchical Reasoning**: Separates strategic planning from detailed execution
2. **Adaptive Computation**: Learns to allocate computation based on difficulty
3. **Sparse Puzzle Embeddings**: Enables learning from limited examples
4. **Q-Learning Integration**: Self-supervised learning of halting policy
5. **Efficient Architecture**: 27M parameters achieve strong performance
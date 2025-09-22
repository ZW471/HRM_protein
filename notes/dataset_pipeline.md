# Dataset Pipeline

This document explains the data loading and preprocessing pipeline for the HRM model, focusing on the Sudoku dataset example.

## Dataset Structure

### 1. Dataset Format

The HRM uses a custom format optimized for puzzle-solving tasks:

```
data/sudoku-extreme-1k-aug-1000/
├── train/
│   ├── dataset.json           # Metadata
│   ├── train__inputs.npy      # Input grids
│   ├── train__labels.npy      # Solution grids
│   ├── train__puzzle_identifiers.npy  # Unique puzzle IDs
│   ├── train__puzzle_indices.npy      # Start index of each puzzle
│   └── train__group_indices.npy       # Grouping for sampling
└── test/
    └── (similar structure)
```

### 2. Data Components

#### a. Inputs and Labels
- **inputs.npy**: Flattened puzzle grids (e.g., 81 cells for 9x9 Sudoku)
- **labels.npy**: Complete solution grids
- **Format**: int32 arrays with vocabulary indices (0-9 for Sudoku)

#### b. Puzzle Identifiers
- Unique ID for each original puzzle
- Used for sparse embedding lookups
- Augmented versions share the same identifier

#### c. Indices Arrays
- **puzzle_indices**: Marks boundaries between different puzzles
- **group_indices**: Groups augmented versions of same puzzle
- Enables efficient sampling during training

## Dataset Building (build_sudoku_dataset.py)

### 1. Data Source
```python
source_repo: "sapientinc/sudoku-extreme"  # Hugging Face dataset
```

### 2. Processing Pipeline

#### a. Reading Raw Data
```python
with open(f"{set_name}.csv") as csvfile:
    for source, question, answer, rating in reader:
        # Convert string format to numpy arrays
        inputs = np.array(question.replace('.', '0'))  # '.' = blank
        labels = np.array(answer)
```

#### b. Subsampling (Optional)
```python
if subsample_size is not None:
    indices = np.random.choice(total_samples, size=subsample_size)
    inputs = [inputs[i] for i in indices]
```

#### c. Data Augmentation
For Sudoku, augmentation preserves puzzle structure:

```python
def shuffle_sudoku(board, solution):
    # 1. Digit permutation (1→7, 2→3, etc.)
    digit_map = np.random.permutation(range(1, 10))

    # 2. Row/column permutations (within 3x3 blocks)
    bands = np.random.permutation(3)  # Shuffle 3-row bands
    row_perm = [b*3 + np.random.permutation(3) for b in bands]

    # 3. Optional transpose
    if np.random.rand() < 0.5:
        board = board.T

    return apply_transformations(board)
```

**Augmentation Benefits:**
- 1000x data multiplication (num_aug=1000)
- Preserves Sudoku constraints
- Improves generalization

#### d. Serialization
```python
# Convert to numpy arrays
np.save("train__inputs.npy", all_inputs)
np.save("train__labels.npy", all_labels)
np.save("train__puzzle_identifiers.npy", puzzle_ids)
```

### 3. Metadata Generation

```python
metadata = PuzzleDatasetMetadata(
    vocab_size=10,           # 0-9 for Sudoku
    seq_len=81,              # 9x9 grid
    num_puzzle_identifiers=1000,  # Unique puzzles
    total_groups=1000,       # Original puzzles
    mean_puzzle_examples=1001,    # With augmentation
    sets=["train"],          # Dataset splits
    pad_id=0,               # Padding token
    blank_identifier_id=-1,  # For invalid puzzles
    ignore_label_id=0       # Skip in loss computation
)
```

## Data Loading (puzzle_dataset.py)

### 1. PuzzleDataset Class

```python
class PuzzleDataset(IterableDataset):
    - Lazy loading with memory mapping
    - Distributed sampling across GPUs
    - Two iteration modes: train vs test
```

### 2. Memory-Mapped Loading

```python
def _lazy_load_dataset(self):
    field_mmap_modes = {
        "inputs": "r",          # Read-only mmap
        "labels": "r",          # Read-only mmap
        "puzzle_identifiers": None,  # Load to memory
        "puzzle_indices": None,      # Load to memory
        "group_indices": None        # Load to memory
    }

    self._data[set_name] = {
        field: np.load(f"{field}.npy", mmap_mode=mode)
        for field, mode in field_mmap_modes.items()
    }
```

**Benefits:**
- Reduced memory usage
- Fast random access
- Handles large datasets efficiently

### 3. Training Data Iteration (_iter_train)

#### a. Group Shuffling
```python
# Shuffle puzzle groups for each epoch
rng = np.random.Generator(seed=config.seed + epoch)
group_order = rng.permutation(num_groups)
```

#### b. Batch Sampling
```python
def _sample_batch(rng, group_order, start_index, batch_size):
    batch = []
    while len(batch) < batch_size:
        # Pick a group
        group_id = group_order[start_index]

        # Random puzzle from group (augmented version)
        puzzle_id = rng.integers(group_start, group_end)

        # Add examples from puzzle
        batch.extend(puzzle_examples)
        start_index += 1

    return batch[:batch_size]
```

**Key Features:**
- Ensures diversity in each batch
- Mixes augmented versions
- Handles variable-length puzzles

#### c. Distributed Sampling
```python
# Each GPU gets different slice
local_start = rank * local_batch_size
local_end = (rank + 1) * local_batch_size
batch = all_examples[local_start:local_end]
```

### 4. Test Data Iteration (_iter_test)

```python
def _iter_test(self):
    for set_name, dataset in self._data.items():
        # Sequential iteration (no shuffling)
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start+batch_size]
            yield set_name, batch, batch_size
```

**Differences from Training:**
- Sequential order (deterministic)
- No augmentation mixing
- Complete coverage of test set

### 5. Batch Collation

```python
def _collate_batch(self, batch):
    # Convert data types
    batch = {k: v.astype(np.int32) for k, v in batch.items()}

    # Replace ignore labels
    batch["labels"][batch["labels"] == ignore_id] = IGNORE_LABEL_ID

    # Pad to batch size
    if len(batch) < local_batch_size:
        pad_values = {
            "inputs": pad_id,
            "labels": IGNORE_LABEL_ID,
            "puzzle_identifiers": blank_identifier_id
        }
        batch = np.pad(batch, pad_values)

    # Convert to PyTorch tensors
    return {k: torch.from_numpy(v) for k, v in batch.items()}
```

## DataLoader Integration (pretrain.py)

### 1. DataLoader Creation

```python
def create_dataloader(config, split, rank, world_size):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split == "test"),
            epochs_per_iter=config.eval_interval,
            rank=rank,
            num_replicas=world_size
        ),
        split=split
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Dataset handles batching
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader, dataset.metadata
```

### 2. DataLoader Configuration

#### a. Batching
- **batch_size=None**: Dataset internally handles batching
- **Global batch size**: Split across GPUs
- **Local batch size**: global_batch_size / world_size

#### b. Performance Optimization
- **prefetch_factor=8**: Preload 8 batches
- **pin_memory=True**: Faster GPU transfer
- **persistent_workers=True**: Reuse worker processes

## Sudoku-Specific Details

### 1. Input Representation
```
Original: "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
Encoded: [0,0,3,0,2,0,6,0,0,9,0,0,3,0,5,...]  # 0 = blank
```

### 2. Label Format
```
Solution: "483921657967345821251876493634517298715293846892674135"
Encoded: [4,8,3,9,2,1,6,5,7,9,6,7,3,4,5,...]
```

### 3. Vocabulary
- 0: Blank/padding
- 1-9: Sudoku digits
- Total vocab_size: 10

## Data Flow Summary

```
1. Build Dataset:
   CSV files → Parse → Augment → Serialize as .npy

2. Load Dataset:
   .npy files → Memory map → PuzzleDataset

3. Training Iteration:
   Shuffle groups → Sample batch → Distributed split → Collate → GPU

4. Batch Processing:
   Raw batch → Add embeddings → Model forward → Loss computation
```

## Performance Considerations

### 1. Memory Efficiency
- Memory mapping prevents loading entire dataset
- Sparse embeddings only for active puzzles
- Padding minimized through careful batching

### 2. I/O Optimization
- Prefetching hides data loading latency
- Persistent workers avoid recreation overhead
- Pin memory for faster GPU transfer

### 3. Augmentation Strategy
- 1000x augmentation provides diversity
- Augmented versions grouped together
- Random sampling ensures variety

### 4. Distributed Efficiency
- Each GPU loads only its portion
- No data duplication across ranks
- Synchronized random seeds for consistency
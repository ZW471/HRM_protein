from typing import Optional, List
import os
import pickle
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_dir: str = "data/cifar10/cifar-10-batches-py"
    output_dir: str = "data/cifar10-processed"

    subsample_size: Optional[int] = None
    num_aug: int = 0

    # ViT patch processing options
    patch_size: int = 8  # Size of each patch (8x8)
    normalize: bool = True  # Normalize pixel values to [0, 1]


def load_cifar_batch(filepath: str):
    """Load a CIFAR-10 batch file."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Convert to numpy arrays
    data = np.array(batch[b'data'])
    labels = np.array(batch[b'labels'])

    return data, labels


def load_cifar_meta(meta_path: str):
    """Load CIFAR-10 metadata."""
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta


def split_image_into_patches(data: np.ndarray, patch_size: int):
    """Split CIFAR-10 image into patches for ViT-style processing."""
    # Reshape from flattened to 32x32x3
    img = data.reshape(32, 32, 3)

    # Calculate number of patches per dimension
    patches_per_dim = 32 // patch_size
    num_patches = patches_per_dim * patches_per_dim

    # Split image into patches
    patches = []
    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            # Extract patch
            patch = img[i*patch_size:(i+1)*patch_size,
                       j*patch_size:(j+1)*patch_size, :]
            # Flatten patch to (patch_size*patch_size*3,)
            patch_flat = patch.reshape(-1)
            patches.append(patch_flat)

    return np.array(patches)  # Shape: (num_patches, patch_size*patch_size*3)


def convert_subset(set_name: str, config: DataProcessConfig):
    """Convert CIFAR-10 data to HRM format."""

    # Load data based on set name
    if set_name == "train":
        # Load all training batches
        all_data = []
        all_labels = []

        for i in range(1, 6):  # data_batch_1 to data_batch_5
            batch_path = os.path.join(config.input_dir, f"data_batch_{i}")
            data, labels = load_cifar_batch(batch_path)
            all_data.append(data)
            all_labels.append(labels)

        inputs = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)

    elif set_name == "test":
        # Load test batch
        test_path = os.path.join(config.input_dir, "test_batch")
        inputs, labels = load_cifar_batch(test_path)
    else:
        raise ValueError(f"Unknown set name: {set_name}")

    # Load metadata
    meta_path = os.path.join(config.input_dir, "batches.meta")
    meta = load_cifar_meta(meta_path)
    class_names = meta['label_names']

    # Normalize if requested
    if config.normalize:
        inputs = inputs.astype(np.float32) / 255.0

    # Subsample if requested (only for training)
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = inputs[indices]
            labels = labels[indices]

    # Generate dataset with patch processing (no augmentation by default)
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # Calculate patch dimensions
    patches_per_dim = 32 // config.patch_size
    num_patches = patches_per_dim * patches_per_dim
    patch_dim = config.patch_size * config.patch_size * 3

    print(f"Processing {len(inputs)} examples for {set_name} set...")
    print(f"Using {config.patch_size}x{config.patch_size} patches, {num_patches} patches per image, {patch_dim} dims per patch")

    for orig_inp, orig_label in zip(tqdm(inputs), labels):
        # Split image into patches
        patches = split_image_into_patches(orig_inp, config.patch_size)

        # Create sequence-to-sequence labels for HRM
        # Use ignore label (-100) for all positions except the last one
        seq_labels = np.full(num_patches, -100, dtype=np.int32)
        seq_labels[-1] = orig_label  # Only the last position has the true label

        # Add to results
        results["inputs"].append(patches)  # Shape: (num_patches, patch_dim)
        results["labels"].append(seq_labels)

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(orig_label)  # Use class as puzzle identifier

        # Push group (one group per example)
        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    def _seq_to_numpy(seq, dtype=np.float32):
        if len(seq) > 0 and hasattr(seq[0], 'shape'):
            # Handle array sequences (patches)
            return np.stack(seq).astype(dtype)
        else:
            # Handle scalar sequences
            return np.array(seq, dtype=dtype)

    # Prepare final results
    final_results = {
        "inputs": _seq_to_numpy(results["inputs"], dtype=np.float32 if config.normalize else np.uint8),
        "labels": _seq_to_numpy(results["labels"], dtype=np.int32),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Determine sequence length and vocab size for patch-based processing
    seq_len = num_patches
    if config.normalize:
        vocab_size = 256  # Continuous values, but we'll discretize for vocab
    else:
        vocab_size = 256  # 0-255 pixel values

    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size + 1,  # +1 for PAD token

        pad_id=0,
        ignore_label_id=-100,  # Standard ignore label for sequence tasks

        blank_identifier_id=0,
        num_puzzle_identifiers=10,  # 10 CIFAR-10 classes

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,  # One example per image (no augmentation by default)
        sets=["all"]
    )

    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    # Save data
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save class names mapping (for visualization)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved {set_name} set to {save_dir}")
    print(f"  - {len(final_results['inputs'])} examples")
    print(f"  - {len(final_results['group_indices']) - 1} groups")
    print(f"  - Input shape: {final_results['inputs'].shape} (batch_size, seq_len)")
    print(f"  - Labels shape: {final_results['labels'].shape}")
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Patches per image: {num_patches}")
    print(f"  - Patch dimension: {patch_dim}")
    print(f"  - Total sequence length: {seq_len}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess CIFAR-10 dataset for HRM training with ViT-style patches."""
    print("CIFAR-10 ViT Dataset Builder")
    print("=" * 50)
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Subsample size: {config.subsample_size}")
    print(f"Augmentation count: {config.num_aug}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Normalize: {config.normalize}")
    print()

    # Check if input directory exists
    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Process train and test sets
    convert_subset("train", config)
    convert_subset("test", config)

    print("\nDataset processing completed successfully!")


if __name__ == "__main__":
    cli()
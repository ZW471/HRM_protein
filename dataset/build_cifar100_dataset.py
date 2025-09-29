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
    input_dir: str = "data/cifar100/cifar-100-python"
    output_dir: str = "data/cifar100-processed"

    subsample_size: Optional[int] = None
    num_aug: int = 0

    # ViT patch processing options
    patch_size: int = 8  # Size of each patch (8x8)
    normalize: bool = True  # Normalize pixel values to [0, 1]

    # CIFAR-100 specific options
    use_coarse_labels: bool = False  # If True, use 20 coarse labels instead of 100 fine labels


def load_cifar100_batch(filepath: str):
    """Load a CIFAR-100 batch file."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Convert to numpy arrays
    data = np.array(batch[b'data'])
    fine_labels = np.array(batch[b'fine_labels'])  # 100 fine classes
    coarse_labels = np.array(batch[b'coarse_labels'])  # 20 coarse classes

    return data, fine_labels, coarse_labels


def load_cifar100_meta(meta_path: str):
    """Load CIFAR-100 metadata."""
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return meta


def split_image_into_patches(data: np.ndarray, patch_size: int):
    """Split CIFAR-100 image into patches for ViT-style processing."""
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


def convert_subset(set_name: str, config: DataProcessConfig, use_coarse_labels: bool = False):
    """Convert CIFAR-100 data to HRM format."""

    # Load data based on set name
    if set_name == "train":
        # Load training batch
        train_path = os.path.join(config.input_dir, "train")
        inputs, fine_labels, coarse_labels = load_cifar100_batch(train_path)
    elif set_name == "test":
        # Load test batch
        test_path = os.path.join(config.input_dir, "test")
        inputs, fine_labels, coarse_labels = load_cifar100_batch(test_path)
    else:
        raise ValueError(f"Unknown set name: {set_name}")

    # Choose which labels to use
    labels = coarse_labels if use_coarse_labels else fine_labels
    num_classes = 20 if use_coarse_labels else 100

    # Load metadata
    meta_path = os.path.join(config.input_dir, "meta")
    meta = load_cifar100_meta(meta_path)

    if use_coarse_labels:
        class_names = [name.decode('utf-8') for name in meta[b'coarse_label_names']]
    else:
        class_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

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

    # Generate dataset with patch processing and optional augmentation
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
    print(f"Using {'coarse' if use_coarse_labels else 'fine'} labels ({num_classes} classes)")

    for orig_inp, orig_label in zip(tqdm(inputs), labels):
        # Create augmentations if requested
        aug_inputs = [orig_inp]
        aug_labels = [orig_label]

        # Add augmented versions
        if config.num_aug > 0:
            # Reshape to 32x32x3 for augmentation
            img_2d = orig_inp.reshape(32, 32, 3)

            for _ in range(config.num_aug):
                # Apply random dihedral transformation
                tid = np.random.randint(0, 8)
                aug_img = dihedral_transform(img_2d, tid)
                # Flatten back to original format
                aug_inp = aug_img.reshape(-1)

                aug_inputs.append(aug_inp)
                aug_labels.append(orig_label)

        # Process all augmented versions
        for aug_inp, aug_label in zip(aug_inputs, aug_labels):
            # Split image into patches
            patches = split_image_into_patches(aug_inp, config.patch_size)

            # Create sequence-to-sequence labels for HRM
            # Use ignore label (-100) for all positions except the last one
            seq_labels = np.full(num_patches, -100, dtype=np.int32)
            seq_labels[-1] = aug_label  # Only the last position has the true label

            # Add to results
            results["inputs"].append(patches)  # Shape: (num_patches, patch_dim)
            results["labels"].append(seq_labels)

            example_id += 1

        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(orig_label)  # Use class as puzzle identifier

        # Push group (one group per original example, including augmentations)
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
        num_puzzle_identifiers=num_classes,  # 100 fine classes or 20 coarse classes

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1 + config.num_aug,  # Including augmentations
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

    # Save class names mapping (for visualization) - only save once from train set
    if set_name == "train":
        label_type = "coarse" if use_coarse_labels else "fine"
        identifiers_path = os.path.join(config.output_dir, f"identifiers_{label_type}.json")
        with open(identifiers_path, "w") as f:
            json.dump(class_names, f, indent=2)

    print(f"Saved {set_name} set to {save_dir}")
    print(f"  - {len(final_results['inputs'])} examples")
    print(f"  - {len(final_results['group_indices']) - 1} groups")
    print(f"  - Input shape: {final_results['inputs'].shape}")
    print(f"  - Labels shape: {final_results['labels'].shape}")
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Patches per image: {num_patches}")
    print(f"  - Patch dimension: {patch_dim}")
    print(f"  - Total sequence length: {seq_len}")
    print(f"  - Number of classes: {num_classes}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess CIFAR-100 dataset for HRM training with ViT-style patches."""
    print("CIFAR-100 ViT Dataset Builder")
    print("=" * 50)
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Subsample size: {config.subsample_size}")
    print(f"Augmentation count: {config.num_aug}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Normalize: {config.normalize}")
    print(f"Use coarse labels: {config.use_coarse_labels} ({'20 classes' if config.use_coarse_labels else '100 classes'})")
    print()

    # Check if input directory exists
    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Process train and test sets
    convert_subset("train", config, config.use_coarse_labels)
    convert_subset("test", config, config.use_coarse_labels)

    print("\\nDataset processing completed successfully!")


if __name__ == "__main__":
    cli()
from typing import Optional, List
import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_dir: str = "/nas-dev-slow/datasets/ImageNet"
    output_dir: str = "data/imagenet-processed"

    subsample_size: Optional[int] = None
    num_aug: int = 0

    # Image processing options
    image_size: int = 224  # Standard ImageNet size
    patch_size: int = 16   # ViT patch size (16x16)
    normalize: bool = True  # Normalize pixel values to [0, 1]

    # Dataset variant to use
    dataset_variant: str = "full-size-v2"  # or "160px-v2"


def get_label_names():
    """Get ImageNet class names mapping."""
    # These are the 10 Imagenette classes
    return [
        'tench',           # n01440764
        'English springer', # n02102040
        'cassette player', # n02979186
        'chain saw',       # n03000684
        'church',          # n03028079
        'French horn',     # n03394916
        'garbage truck',   # n03417042
        'gas pump',        # n03425413
        'golf ball',       # n03445777
        'parachute'        # n03888257
    ]


def load_imagenet_folder(data_path: str, transform=None):
    """Load ImageNet data using ImageFolder."""
    dataset = ImageFolder(data_path, transform=transform)
    return dataset


def resize_and_crop_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target size with center cropping."""
    pil_image = Image.fromarray(image)

    # Resize maintaining aspect ratio
    w, h = pil_image.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    pil_image = pil_image.crop((left, top, right, bottom))

    return np.array(pil_image)


def split_image_into_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Split image into patches for ViT-style processing."""
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, f"Image size {h}x{w} not divisible by patch size {patch_size}"

    patches_per_dim = h // patch_size
    num_patches = patches_per_dim * patches_per_dim

    # Split image into patches
    patches = []
    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            # Extract patch
            patch = image[i*patch_size:(i+1)*patch_size,
                         j*patch_size:(j+1)*patch_size, :]
            # Flatten patch to (patch_size*patch_size*3,)
            patch_flat = patch.reshape(-1)
            patches.append(patch_flat)

    return np.array(patches)  # Shape: (num_patches, patch_size*patch_size*3)


def convert_subset(set_name: str, config: DataProcessConfig):
    """Convert ImageNet data to HRM format."""

    # Determine data directory
    if set_name == "train":
        subset_dir = "train"
    elif set_name == "validation" or set_name == "val":
        subset_dir = "val"
    else:
        raise ValueError(f"Unknown set name: {set_name}")

    # Look for ImageFolder structure first (extracted images)
    data_path = None
    possible_paths = [
        os.path.join(config.input_dir, "downloads", "extracted", "*", "imagenette2", subset_dir),
        os.path.join(config.input_dir, "downloads", "extracted", "*", "imagenette2-160", subset_dir),
        os.path.join(config.input_dir, "imagenette", config.dataset_variant, "1.0.0", subset_dir),
        os.path.join(config.input_dir, subset_dir),
    ]

    import glob
    for path_pattern in possible_paths:
        matches = glob.glob(path_pattern)
        if matches and os.path.isdir(matches[0]):
            data_path = matches[0]
            break

    if data_path is None:
        raise FileNotFoundError(f"Could not find {set_name} data directory in {config.input_dir}")

    print(f"Loading {set_name} data from: {data_path}")

    # Create transform for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    # Load dataset using ImageFolder
    dataset = load_imagenet_folder(data_path, transform=transform)

    print(f"Found {len(dataset)} examples in {set_name} set")
    print(f"Classes: {dataset.classes}")

    # Create dataloader for efficient loading
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Collect all data
    all_images = []
    all_labels = []

    for image, label in tqdm(dataloader, desc=f"Loading {set_name} data"):
        # Convert from tensor to numpy
        image_np = image.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
        if config.normalize:
            # Already normalized by ToTensor()
            pass
        else:
            # Convert back to 0-255 range
            image_np = (image_np * 255).astype(np.uint8)

        all_images.append(image_np)
        all_labels.append(label.item())

    # Subsample if requested (only for training)
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(all_images)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            all_images = [all_images[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]

    # Generate dataset with patch processing and optional augmentation
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # Calculate patch dimensions
    patches_per_dim = config.image_size // config.patch_size
    num_patches = patches_per_dim * patches_per_dim
    patch_dim = config.patch_size * config.patch_size * 3

    print(f"Processing {len(all_images)} examples for {set_name} set...")
    print(f"Using {config.patch_size}x{config.patch_size} patches, {num_patches} patches per image, {patch_dim} dims per patch")

    for orig_image, orig_label in zip(tqdm(all_images, desc=f"Processing {set_name}"), all_labels):
        # Create augmentations if requested
        aug_images = [orig_image]
        aug_labels = [orig_label]

        # Add augmented versions
        if config.num_aug > 0:
            for _ in range(config.num_aug):
                # Apply random dihedral transformation
                tid = np.random.randint(0, 8)
                aug_img = dihedral_transform(orig_image, tid)
                aug_images.append(aug_img)
                aug_labels.append(orig_label)

        # Process all augmented versions
        for aug_image, aug_label in zip(aug_images, aug_labels):
            # Split image into patches
            patches = split_image_into_patches(aug_image, config.patch_size)

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
    num_classes = len(dataset.classes)
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size + 1,  # +1 for PAD token

        pad_id=0,
        ignore_label_id=-100,  # Standard ignore label for sequence tasks

        blank_identifier_id=0,
        num_puzzle_identifiers=num_classes,  # Number of classes

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
        class_names = dataset.classes
        with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
            json.dump(class_names, f, indent=2)

    print(f"Saved {set_name} set to {save_dir}")
    print(f"  - {len(final_results['inputs'])} examples")
    print(f"  - {len(final_results['group_indices']) - 1} groups")
    print(f"  - Input shape: {final_results['inputs'].shape}")
    print(f"  - Labels shape: {final_results['labels'].shape}")
    print(f"  - Image size: {config.image_size}x{config.image_size}")
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Patches per image: {num_patches}")
    print(f"  - Patch dimension: {patch_dim}")
    print(f"  - Total sequence length: {seq_len}")
    print(f"  - Number of classes: {num_classes}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess ImageNet/Imagenette dataset for HRM training with ViT-style patches."""
    print("ImageNet/Imagenette ViT Dataset Builder")
    print("=" * 50)
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Dataset variant: {config.dataset_variant}")
    print(f"Subsample size: {config.subsample_size}")
    print(f"Augmentation count: {config.num_aug}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Normalize: {config.normalize}")
    print()

    # Check if input directory exists
    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Process train and validation sets
    convert_subset("train", config)
    convert_subset("test", config)

    print("\nDataset processing completed successfully!")


if __name__ == "__main__":
    cli()
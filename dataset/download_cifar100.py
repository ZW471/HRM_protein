#!/usr/bin/env python3
"""
Download CIFAR-100 dataset.

The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes,
with 600 images per class. There are 500 training images and 100 testing
images per class.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel


cli = ArgParser()


class DownloadConfig(BaseModel):
    output_dir: str = "data/cifar100/cifar-100-python"
    url: str = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    force_download: bool = False


def download_and_extract_cifar100(config: DownloadConfig):
    """Download and extract CIFAR-100 dataset."""

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_files = [
        output_path / "train",
        output_path / "test",
        output_path / "meta"
    ]

    if not config.force_download and all(f.exists() for f in expected_files):
        print(f"CIFAR-100 dataset already exists at {output_path}")
        print("Use --force_download=True to re-download")
        return

    # Download the dataset
    tar_filename = output_path.parent / "cifar-100-python.tar.gz"

    print(f"Downloading CIFAR-100 from {config.url}")
    print(f"Saving to {tar_filename}")

    # Download with progress
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")

    urllib.request.urlretrieve(config.url, tar_filename, reporthook=progress_hook)
    print()  # New line after progress

    # Extract the tar file
    print(f"Extracting {tar_filename}")
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall(output_path.parent)

    # The extracted folder will be named "cifar-100-python"
    extracted_path = output_path.parent / "cifar-100-python"

    # If output_dir is different from the extracted name, rename it
    if extracted_path != output_path:
        if output_path.exists():
            # Remove existing directory if it exists
            import shutil
            shutil.rmtree(output_path)
        extracted_path.rename(output_path)

    # Clean up tar file
    tar_filename.unlink()

    print(f"CIFAR-100 dataset successfully downloaded and extracted to {output_path}")

    # Verify the extracted files
    print("\nVerifying extracted files:")
    for file_path in output_path.rglob("*"):
        if file_path.is_file():
            print(f"  {file_path.relative_to(output_path)} ({file_path.stat().st_size} bytes)")


@cli.command(singleton=True)
def download_cifar100(config: DownloadConfig):
    """Download CIFAR-100 dataset."""
    print("CIFAR-100 Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {config.output_dir}")
    print(f"Download URL: {config.url}")
    print(f"Force download: {config.force_download}")
    print()

    download_and_extract_cifar100(config)


if __name__ == "__main__":
    cli()
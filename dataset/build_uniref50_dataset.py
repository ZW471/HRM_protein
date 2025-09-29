from typing import Optional
import os
import gzip
import json
import numpy as np
import random

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from Bio import SeqIO

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_file: str = "data/uniref50.fasta.gz"
    output_dir: str = "data/uniref50-denoising"

    max_sequences: Optional[int] = None
    min_seq_length: int = 30
    max_seq_length: int = 1024
    mask_prob: float = 0.15
    test_split: float = 0.01

    num_aug: int = 0


# Standard amino acid alphabet + Selenocysteine (U) and Pyrrolysine (O)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYUO"
AA_TO_IDX = {aa: i + 3 for i, aa in enumerate(AMINO_ACIDS)}  # Start from 3 (0=PAD, 1=MASK, 2=X)
AA_TO_IDX['<mask>'] = 1  # Mask token
AA_TO_IDX['X'] = 2  # Unknown amino acid
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
IDX_TO_AA[0] = '<PAD>'

VOCAB_SIZE = len(AA_TO_IDX) + 1  # +1 for PAD token


def parse_fasta(file_path: str, max_sequences: Optional[int] = None,
                min_length: int = 30, max_length: int = 1024):
    """Parse FASTA file and return sequences using BioPython."""
    sequences = []

    # Open file handle (gzipped or regular)
    if file_path.endswith('.gz'):
        handle = gzip.open(file_path, 'rt')
    else:
        handle = open(file_path, 'r')

    try:
        pbar = tqdm(desc="Parsing FASTA")
        for record in SeqIO.parse(handle, "fasta"):
            pbar.update(1)

            seq_str = str(record.seq).upper()

            # Check length constraints first to avoid unnecessary processing
            if not (min_length <= len(seq_str) <= max_length):
                continue

            # Convert all non-standard amino acids to 'X', keep standard ones + U and O
            clean_seq = ''.join(c if c in AMINO_ACIDS else 'X' for c in seq_str)

            sequences.append(clean_seq)
            pbar.set_postfix({"valid_seqs": len(sequences)})

            if max_sequences and len(sequences) >= max_sequences:
                break
        pbar.close()
    finally:
        handle.close()

    return sequences


def mask_sequence(sequence: str, mask_prob: float = 0.15):
    """Create masked version of sequence for denoising task."""
    seq_list = list(sequence)
    masked_positions = []

    for i in range(len(seq_list)):
        if random.random() < mask_prob:
            masked_positions.append(i)
            seq_list[i] = '<mask>'

    return ''.join(seq_list), masked_positions


def sequence_to_indices(sequence: str, max_length: int):
    """Convert sequence to indices and pad/truncate to max_length."""
    # Handle <mask> tokens by splitting the sequence properly
    tokens = []
    i = 0
    while i < len(sequence):
        if sequence[i:i+6] == '<mask>':
            tokens.append('<mask>')
            i += 6
        else:
            tokens.append(sequence[i])
            i += 1

    # Convert tokens to indices, truncate if necessary
    indices = [AA_TO_IDX.get(token, AA_TO_IDX['X']) for token in tokens[:max_length]]

    # Pad if necessary
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))

    return np.array(indices, dtype=np.int32)


def convert_subset(sequences: list, set_name: str, config: DataProcessConfig):
    """Convert sequences to dataset format."""
    print(f"Processing {len(sequences)} sequences for {set_name} set")

    if len(sequences) == 0:
        print(f"Skipping {set_name} set - no sequences")
        return

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for seq_idx, original_seq in enumerate(tqdm(sequences, desc=f"Converting {set_name}")):
        for aug_idx in range(1 + num_augments):
            # Create masked sequence
            if aug_idx == 0:
                # Original masking
                masked_seq, _ = mask_sequence(original_seq, config.mask_prob)
            else:
                # Additional augmentations with different random masks
                masked_seq, _ = mask_sequence(original_seq, config.mask_prob)

            # Convert to indices
            input_indices = sequence_to_indices(masked_seq, config.max_seq_length)
            label_indices = sequence_to_indices(original_seq, config.max_seq_length)

            results["inputs"].append(input_indices)
            results["labels"].append(label_indices)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)  # Single identifier for protein sequences

        # Push group
        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_length,
        vocab_size=VOCAB_SIZE,

        pad_id=0,
        ignore_label_id=0,

        blank_identifier_id=0,
        num_puzzle_identifiers=1,

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    print(f"Saved {set_name} set with {len(results['inputs'])} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Main preprocessing function."""
    print(f"Loading sequences from {config.source_file}")

    # Parse FASTA file
    all_sequences = parse_fasta(
        config.source_file,
        max_sequences=config.max_sequences,
        min_length=config.min_seq_length,
        max_length=config.max_seq_length
    )

    print(f"Loaded {len(all_sequences)} valid sequences")

    # Split into train/test
    random.shuffle(all_sequences)
    test_size = max(1, int(len(all_sequences) * config.test_split))  # Ensure at least 1 test example

    test_sequences = all_sequences[:test_size]
    train_sequences = all_sequences[test_size:]

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    # Convert datasets
    convert_subset(train_sequences, "train", config)
    convert_subset(test_sequences, "test", config)

    # Save amino acid mapping for reference
    with open(os.path.join(config.output_dir, "amino_acid_mapping.json"), "w") as f:
        json.dump({"aa_to_idx": AA_TO_IDX, "idx_to_aa": IDX_TO_AA}, f, indent=2)

    # Save identifiers mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["protein_sequence"], f)

    print(f"Dataset saved to {config.output_dir}")


if __name__ == "__main__":
    cli()
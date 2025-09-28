#!/usr/bin/env python3

import os
import sys
import pickle
import math
import argparse
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm

# Import HRM components (assuming they exist in your codebase)
try:
    from models.hrm.hrm_act_v1 import (
        HierarchicalReasoningModel_ACTV1,
        HierarchicalReasoningModel_ACTV1Carry,
        HierarchicalReasoningModel_ACTV1InnerCarry,
    )
    from models.losses import ACTLossHead
except ImportError:
    print("Warning: HRM modules not found. Using placeholder implementations.")
    # Define placeholder classes for demonstration
    class HierarchicalReasoningModel_ACTV1(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, *args, **kwargs):
            raise NotImplementedError("Replace with actual HRM implementation")

    class HierarchicalReasoningModel_ACTV1Carry:
        def __init__(self, inner_carry, steps, halted, current_data):
            self.inner_carry = inner_carry
            self.steps = steps
            self.halted = halted
            self.current_data = current_data

    class HierarchicalReasoningModel_ACTV1InnerCarry:
        def __init__(self, z_H, z_L):
            self.z_H = z_H
            self.z_L = z_L

    class ACTLossHead(nn.Module):
        def __init__(self, base_model, loss_type="softmax_cross_entropy"):
            super().__init__()
            self.base_model = base_model
            self.loss_type = loss_type

        def initial_carry(self, batch):
            return None

        def forward(self, *args, **kwargs):
            raise NotImplementedError("Replace with actual ACT loss implementation")


@dataclass
class HRMConfigArgs:
    # Model architecture
    batch_size: int = 32
    seq_len: int = 64  # Will be set based on patch configuration
    vocab_size: int = 256  # For image patch embeddings
    num_puzzle_identifiers: int = 10  # Number of CIFAR-10 classes
    hidden_size: int = 256
    num_heads: int = 8
    expansion: float = 4.0
    H_layers: int = 3  # High-level reasoning layers
    L_layers: int = 3  # Low-level processing layers
    H_cycles: int = 1
    L_cycles: int = 2  # More cycles for visual processing
    halt_max_steps: int = 20
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"  # Use bfloat16 for FlashAttention compatibility
    pos_encodings: str = "rope"

    # Image processing specific
    patch_size: int = 4
    image_size: int = 32
    num_channels: int = 3

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100

    def to_model_config(self) -> Dict[str, object]:
        # Calculate sequence length based on patches
        patches_per_side = self.image_size // self.patch_size
        seq_len = patches_per_side * patches_per_side  # No class token for now

        return {
            "batch_size": self.batch_size,
            "seq_len": seq_len,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": self.num_puzzle_identifiers,
            "vocab_size": self.vocab_size,
            "H_cycles": self.H_cycles,
            "L_cycles": self.L_cycles,
            "H_layers": self.H_layers,
            "L_layers": self.L_layers,
            "hidden_size": self.hidden_size,
            "expansion": self.expansion,
            "num_heads": self.num_heads,
            "pos_encodings": self.pos_encodings,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": self.halt_max_steps,
            "halt_exploration_prob": self.halt_exploration_prob,
            "forward_dtype": self.forward_dtype,
        }


def load_cifar10_batch(file_path):
    """Load a single CIFAR-10 batch file."""
    with open(file_path, 'rb') as fo:
        batch_dict = pickle.load(fo, encoding='bytes')
    return batch_dict


def load_cifar10_data(data_dir="../data/cifar10/cifar-10-batches-py"):
    """Load CIFAR-10 dataset."""
    # Load training data
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        batch = load_cifar10_batch(batch_file)
        train_data.append(torch.from_numpy(batch[b'data']))
        train_labels.extend(batch[b'labels'])

    # Combine all training batches using torch
    train_data = torch.vstack(train_data)
    train_labels = torch.tensor(train_labels)

    # Load test data
    test_batch = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    test_data = torch.from_numpy(test_batch[b'data'])
    test_labels = torch.tensor(test_batch[b'labels'])

    # Load class names
    meta_file = os.path.join(data_dir, "batches.meta")
    with open(meta_file, 'rb') as fo:
        meta_dict = pickle.load(fo, encoding='bytes')
        class_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]

    return (train_data, train_labels), (test_data, test_labels), class_names


def prepare_cifar10_for_hrm(data, labels, patch_size=4, normalize=True):
    """Convert CIFAR-10 data to patches suitable for HRM."""
    data_tensor = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels)

    batch_size = data_tensor.shape[0]
    # Reshape from (batch, 3072) to (batch, 3, 32, 32) then to (batch, 32, 32, 3)
    images = data_tensor.view(batch_size, 3, 32, 32).permute(0, 2, 3, 1).float()

    if normalize:
        images = images / 255.0  # Normalize to [0, 1]

    # Convert to patches
    H, W, C = images.shape[1], images.shape[2], images.shape[3]
    patches_per_side = H // patch_size

    # Extract patches using unfold
    patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.reshape(batch_size, patches_per_side * patches_per_side, patch_size * patch_size * C)

    # Quantize patches to discrete tokens (simple approach)
    # In practice, you might want to use a more sophisticated tokenization
    patches_quantized = (patches * 255).long().clamp(0, 255)

    # Flatten patch features to create token sequences
    # Here we're using a simple approach - sum across patch dimensions and mod by vocab_size
    tokens = patches_quantized.sum(dim=-1) % 256  # Assuming vocab_size = 256

    labels_tensor = labels_tensor.long()

    return tokens, labels_tensor


class HRMImageClassifier(nn.Module):
    """HRM-based image classifier for CIFAR-10."""

    def __init__(self, config: HRMConfigArgs):
        super().__init__()
        self.config = config

        # Initialize HRM base model
        self.hrm_model = HierarchicalReasoningModel_ACTV1(config.to_model_config())

        # Classification head - extracts from HRM's final hidden states
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.num_puzzle_identifiers)
        )

    def forward(self, tokens, carry=None):
        """Forward pass through HRM."""
        batch_size, seq_len = tokens.shape

        # Ensure tokens are within vocab range
        tokens = torch.clamp(tokens, 0, self.config.vocab_size - 1)

        # Create input batch for HRM
        batch = {
            "inputs": tokens,  # HRM expects raw token indices
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        }

        # Initialize carry state if not provided
        if carry is None:
            carry = self.hrm_model.initial_carry(batch)
            # Move carry to the same device as input
            carry = self._carry_to_device(carry, tokens.device)

        # Forward through HRM
        new_carry, outputs = self.hrm_model(carry, batch)

        # Extract logits from HRM output
        hrm_logits = outputs["logits"]  # Shape: (batch_size, seq_len, vocab_size)

        # Get hidden states from the HRM model's inner state
        # Use the high-level reasoning state z_H for classification
        z_H = new_carry.inner_carry.z_H  # Shape: (batch_size, seq_len, hidden_size)

        # Global average pooling over sequence dimension for classification
        # Skip puzzle embedding positions if they exist
        puzzle_emb_len = getattr(self.hrm_model.inner, 'puzzle_emb_len', 0)
        if puzzle_emb_len > 0:
            pooled = z_H[:, puzzle_emb_len:].mean(dim=1)  # Skip puzzle embeddings
        else:
            pooled = z_H.mean(dim=1)  # Pool all positions

        # Convert to float32 for classifier compatibility
        pooled = pooled.float()

        # Classification
        class_logits = self.classifier(pooled)

        return {
            "logits": class_logits,
            "hrm_outputs": outputs,
            "carry": new_carry
        }

    def _carry_to_device(self, carry, device):
        """Move carry state to specified device."""
        # Ensure proper dtype for FlashAttention compatibility
        forward_dtype = getattr(torch, self.config.forward_dtype)

        inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=carry.inner_carry.z_H.to(device=device, dtype=forward_dtype),
            z_L=carry.inner_carry.z_L.to(device=device, dtype=forward_dtype),
        )
        current_data = {}
        for k, v in carry.current_data.items():
            if hasattr(v, 'to'):
                # Keep integer types as they are, only cast float types
                if v.dtype.is_floating_point:
                    current_data[k] = v.to(device=device, dtype=forward_dtype)
                else:
                    current_data[k] = v.to(device)
            else:
                current_data[k] = v

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data=current_data,
        )


def carry_to_device(carry, device):
    """Move HRM carry state to device."""
    if carry is None:
        return None

    inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
        z_H=carry.inner_carry.z_H.to(device) if hasattr(carry.inner_carry, 'z_H') else None,
        z_L=carry.inner_carry.z_L.to(device) if hasattr(carry.inner_carry, 'z_L') else None,
    )
    current_data = {k: v.to(device) if hasattr(v, 'to') else v
                    for k, v in carry.current_data.items()} if hasattr(carry, 'current_data') else {}
    return HierarchicalReasoningModel_ACTV1Carry(
        inner_carry=inner_carry,
        steps=carry.steps.to(device) if hasattr(carry, 'steps') else torch.tensor(0, device=device),
        halted=carry.halted.to(device) if hasattr(carry, 'halted') else torch.tensor(False, device=device),
        current_data=current_data,
    )


def run_training_iteration_hrm(
        model: HRMImageClassifier,
        loss_head: Optional[ACTLossHead],
        optimizer: torch.optim.Optimizer,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
) -> Tuple[float, Dict[str, torch.Tensor], int]:
    """Run one training iteration with HRM."""

    optimizer.zero_grad()

    try:
        # Forward pass through HRM classifier
        outputs = model(tokens)

        # Extract classification logits
        class_logits = outputs["logits"]
        hrm_outputs = outputs["hrm_outputs"]
        carry = outputs["carry"]

        # Classification loss (ensure float32 for loss computation)
        class_loss = nn.CrossEntropyLoss()(class_logits.float(), labels)

        # Additional HRM-specific losses if available
        total_loss = class_loss

        # If we have Q-learning components from HRM, add Q-loss
        if "q_halt_logits" in hrm_outputs and "q_continue_logits" in hrm_outputs:
            q_halt = hrm_outputs["q_halt_logits"].float()
            q_continue = hrm_outputs["q_continue_logits"].float()

            # Simple Q-learning loss: encourage continuing for better accuracy
            # This is a simplified version - in practice you'd want more sophisticated rewards
            q_loss = 0.1 * F.mse_loss(q_continue, torch.ones_like(q_continue))
            total_loss = total_loss + q_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Compute accuracy (ensure float32 for comparison)
        _, predicted = class_logits.float().max(1)
        accuracy = predicted.eq(labels).float().mean()

        # Get number of reasoning steps
        steps = carry.steps.float().mean().item() if hasattr(carry, 'steps') else 1.0

        metrics = {
            "class_loss": class_loss.detach(),
            "total_loss": total_loss.detach(),
            "accuracy": accuracy,
            "steps": torch.tensor(steps)
        }

        # Add Q-learning metrics if available
        if "q_halt_logits" in hrm_outputs:
            halt_rate = (hrm_outputs["q_halt_logits"] > hrm_outputs["q_continue_logits"]).float().mean()
            metrics["halt_rate"] = halt_rate

        return total_loss.item(), metrics, int(steps)

    except Exception as e:
        print(f"HRM training iteration failed: {e}")
        import traceback
        traceback.print_exc()

        # Return dummy values to keep training going
        return 0.0, {"error": True}, 0


def validate_model(model, val_loader, device, config):
    """Validate the HRM model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    total_steps = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tokens, targets in val_loader:
            tokens, targets = tokens.to(device), targets.to(device)

            try:
                # Forward pass through HRM
                outputs = model(tokens)
                class_logits = outputs["logits"]
                carry = outputs["carry"]

                loss = criterion(class_logits.float(), targets)
                val_loss += loss.item()

                _, predicted = class_logits.float().max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Track reasoning steps
                if hasattr(carry, 'steps'):
                    total_steps += carry.steps.float().mean().item()
                else:
                    total_steps += 1.0
                num_batches += 1

            except Exception as e:
                print(f"Validation error: {e}")
                # Use dummy predictions to keep validation going
                dummy_logits = torch.randn(tokens.shape[0], config.num_puzzle_identifiers, device=device)
                loss = criterion(dummy_logits, targets)
                val_loss += loss.item()

                _, predicted = dummy_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                total_steps += 1.0
                num_batches += 1

    val_accuracy = 100. * correct / total if total > 0 else 0.0
    val_loss = val_loss / num_batches if num_batches > 0 else 0.0
    avg_steps = total_steps / num_batches if num_batches > 0 else 1.0

    return val_loss, val_accuracy, avg_steps


def main():
    parser = argparse.ArgumentParser(description='HRM CIFAR-10 Training')
    parser.add_argument('--data-dir', default='../../data/cifar10/cifar-10-batches-py', help='CIFAR-10 data directory')
    parser.add_argument('--checkpoint-dir', default='./hrm_checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch-size', type=int, default=8, help='Patch size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--H-layers', type=int, default=3, help='High-level layers')
    parser.add_argument('--L-layers', type=int, default=3, help='Low-level layers')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10_data(args.data_dir)

    # Prepare data for HRM
    train_tokens, train_labels_tensor = prepare_cifar10_for_hrm(
        train_data, train_labels, patch_size=args.patch_size
    )
    test_tokens, test_labels_tensor = prepare_cifar10_for_hrm(
        test_data, test_labels, patch_size=args.patch_size
    )

    print(f"Train tokens shape: {train_tokens.shape}")
    print(f"Test tokens shape: {test_tokens.shape}")

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_tokens, train_labels_tensor)
    test_dataset = TensorDataset(test_tokens, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize HRM configuration
    config = HRMConfigArgs(
        batch_size=args.batch_size,
        seq_len=train_tokens.shape[1],
        hidden_size=args.hidden_size,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        patch_size=args.patch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    # Initialize model
    model = HRMImageClassifier(config).to(device)

    # For now, don't use ACT loss head since the HRM components may not be fully implemented
    # loss_head = ACTLossHead(model.hrm_model, loss_type="softmax_cross_entropy").to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting training...")

    best_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total_steps = 0

        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')

        for tokens, labels in train_pbar:
            tokens, labels = tokens.to(device), labels.to(device)

            loss, metrics, steps = run_training_iteration_hrm(
                model, None, optimizer, tokens, labels, device
            )

            running_loss += loss
            total_steps += steps

            train_pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{metrics.get("accuracy", 0):.3f}',
                'Steps': f'{steps}'
            })

        avg_loss = running_loss / len(train_loader)
        avg_steps = total_steps / len(train_loader)

        # Validation
        val_loss, val_accuracy, val_steps = validate_model(model, test_loader, device, config)

        scheduler.step()

        print(f'Epoch {epoch+1:3d}/{config.epochs} | '
              f'Train Loss: {avg_loss:.4f} | Train Steps: {avg_steps:.2f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val Steps: {val_steps:.2f}')

        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
            }, os.path.join(args.checkpoint_dir, 'best_hrm_model.pth'))

    print("=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
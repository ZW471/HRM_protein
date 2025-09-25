#!/usr/bin/env python3

import os
import sys
import pickle
import math
import argparse
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from experiments.model.vit import patch_image
from experiments.model.transformer import TransformerEncoder


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


def prepare_cifar10_for_vit(data, labels, normalize=True):
    """Convert CIFAR-10 data to torch tensors suitable for ViT."""

    # Ensure inputs are torch tensors
    data_tensor = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels)

    # Convert data to images and normalize using torch operations
    batch_size = data_tensor.shape[0]
    # Reshape from (batch, 3072) to (batch, 3, 32, 32) then to (batch, 32, 32, 3)
    images = data_tensor.view(batch_size, 3, 32, 32).permute(0, 2, 3, 1).float()

    if normalize:
        images = images / 255.0  # Normalize to [0, 1]

    labels_tensor = labels_tensor.long()

    return images, labels_tensor


class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10, max_seq_len=100, patch_size=4):
        super().__init__()
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size

        # Calculate patch dimension
        CHANNEL_NUM = 3
        dim_in = CHANNEL_NUM * patch_size * patch_size

        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_in // 4),
            nn.GELU(),
            nn.Linear(dim_in // 4, num_classes),
        )

        # Initialize positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, dim_in) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, dim_in))

    def forward(self, x):
        # x shape: (batch_size, H, W, C)
        batch_size, H, W, C = x.shape

        # Convert to patches using unfold
        patch_size = self.patch_size
        dim_in = C * patch_size * patch_size

        # Extract patches
        x_patches = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        x_patches = x_patches.reshape(batch_size, (H // patch_size) * (W // patch_size), dim_in)

        # Add CLS token (using mean of patches as CLS token)
        cls_token = self.cls_token.expand((batch_size, -1, -1))
        x_with_cls = torch.concat([cls_token, x_patches], dim=1)  # (batch_size, seq_len+1, patch_dim)

        # Add positional encoding
        seq_len_with_cls = x_with_cls.shape[1]
        if seq_len_with_cls <= self.max_seq_len:
            pos_enc = self.pos_encoding[:seq_len_with_cls].unsqueeze(0)  # (1, seq_len+1, patch_dim)
            x_with_pos = x_with_cls + pos_enc  # (batch_size, seq_len+1, patch_dim)
        else:
            # Handle sequences longer than max_seq_len
            pos_enc = self.pos_encoding.unsqueeze(0)  # (1, max_seq_len, patch_dim)
            x_with_pos = x_with_cls[:, :self.max_seq_len] + pos_enc

        # Pass through transformer encoder
        encoded = self.encoder(x_with_pos)  # (batch_size, seq_len+1, patch_dim)

        # Use CLS token for classification
        return self.classifier(encoded[:, 0])


def validate_model(model, val_loader, criterion, device, rank):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Aggregate across all processes
    if dist.is_initialized():
        metrics = torch.tensor([val_loss, correct, total], device=device)
        dist.all_reduce(metrics)
        val_loss, correct, total = metrics.tolist()

    val_accuracy = 100. * correct / total if total > 0 else 0.0
    val_loss = val_loss / (len(val_loader) * dist.get_world_size() if dist.is_initialized() else len(val_loader))

    return val_loss, val_accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_path, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }

    # Save regular checkpoint
    torch.save(state, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth'))

    # Save best checkpoint
    if is_best:
        torch.save(state, os.path.join(checkpoint_path, 'best_model.pth'))


def main():
    parser = argparse.ArgumentParser(description='ViT Training with Distributed Support')
    parser.add_argument('--data-dir', default='./data/cifar10/cifar-10-batches-py', help='CIFAR-10 data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=1024 // 8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--hidden-dim', type=int, default=768, help='Hidden dimension')

    args = parser.parse_args()

    # Initialize distributed training if available
    rank = 0
    world_size = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10 dataset
    if rank == 0:
        print("Loading CIFAR-10 dataset...")

    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10_data(args.data_dir)
    train_images, train_labels_tensor = prepare_cifar10_for_vit(train_data, train_labels)
    test_images, test_labels_tensor = prepare_cifar10_for_vit(test_data, test_labels)

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_images, train_labels_tensor)
    test_dataset = TensorDataset(test_images, test_labels_tensor)

    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model setup
    CHANNEL_NUM = 3
    H, W = 32, 32
    dim_in = CHANNEL_NUM * args.patch_size * args.patch_size

    encoder = TransformerEncoder(
        dim_in=dim_in,
        dim_qk=args.hidden_dim,
        dim_v=dim_in,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )

    model = ViTClassifier(encoder, num_classes=10, patch_size=args.patch_size, max_seq_len=(32 // args.patch_size) ** 2)
    model = model.to(device)

    # Wrap model with DDP if distributed
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    best_acc = 0.0

    if rank == 0:
        print(f"Training on device: {device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"World size: {world_size}")
        print("Starting training...")
        print("=" * 60)

    # Training loop
    for epoch in range(args.epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if rank == 0:
            train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        else:
            train_pbar = train_loader

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if rank == 0 and hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        # Aggregate training metrics across all processes
        if dist.is_initialized():
            metrics = torch.tensor([running_loss, correct, total], device=device)
            dist.all_reduce(metrics)
            running_loss, correct, total = metrics.tolist()

        train_loss = running_loss / (len(train_loader) * world_size)
        train_accuracy = 100. * correct / total

        # Validation
        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device, rank)

        # Update learning rate
        scheduler.step()

        # Save best model
        is_best = val_accuracy > best_acc
        if is_best:
            best_acc = val_accuracy

        if rank == 0:
            print(f'Epoch {epoch+1:2d}/{args.epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1, best_acc, args.checkpoint_dir, is_best)

    if rank == 0:
        print("=" * 60)
        print("Training completed!")
        print(f"Best validation accuracy: {best_acc:.2f}%")

    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
"""Minimal training demo for the Hierarchical Reasoning Model (HRM)."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from models.losses import ACTLossHead


@dataclass
class HRMConfigArgs:
    batch_size: int = 2
    seq_len: int = 8
    vocab_size: int = 32
    num_puzzle_identifiers: int = 4
    hidden_size: int = 64
    num_heads: int = 4
    expansion: float = 4.0
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 1
    L_cycles: int = 1
    halt_max_steps: int = 3
    halt_exploration_prob: float = 0.0
    forward_dtype: str = "bfloat16"
    pos_encodings: str = "rope"

    def to_model_config(self) -> Dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
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


def build_dummy_batch(config: HRMConfigArgs, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.seq_len),
        dtype=torch.int64,
        device=device,
    )
    labels = torch.roll(inputs, shifts=-1, dims=1)
    puzzle_identifiers = torch.zeros(
        config.batch_size,
        dtype=torch.int64,
        device=device,
    )
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
    }


def carry_to_device(
    carry: HierarchicalReasoningModel_ACTV1Carry,
    device: torch.device,
) -> HierarchicalReasoningModel_ACTV1Carry:
    inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
        z_H=carry.inner_carry.z_H.to(device),
        z_L=carry.inner_carry.z_L.to(device),
    )
    current_data = {k: v.to(device) for k, v in carry.current_data.items()}
    return HierarchicalReasoningModel_ACTV1Carry(
        inner_carry=inner_carry,
        steps=carry.steps.to(device),
        halted=carry.halted.to(device),
        current_data=current_data,
    )


def run_training_iteration(
    loss_head: ACTLossHead,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[HierarchicalReasoningModel_ACTV1Carry, float, Dict[str, torch.Tensor], int]:
    carry = carry_to_device(loss_head.initial_carry(batch), device)
    optimizer.zero_grad(set_to_none=True)

    finished = False
    total_loss = 0.0
    inner_steps = 0
    metrics: Dict[str, torch.Tensor] = {}

    while not finished:
        carry, loss, metrics, _, finished = loss_head(return_keys=[], carry=carry, batch=batch)
        (loss / batch_size).backward()
        total_loss += loss.item() / batch_size
        inner_steps += 1

    optimizer.step()
    return carry, total_loss, metrics, inner_steps


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA-capable device for FlashAttention.")

    torch.manual_seed(0)

    config = HRMConfigArgs()
    device = torch.device("cuda")

    base_model = HierarchicalReasoningModel_ACTV1(config.to_model_config())
    loss_head = ACTLossHead(base_model, loss_type="softmax_cross_entropy")
    loss_head.to(device)
    loss_head.train()

    optimizer = torch.optim.Adam(loss_head.parameters(), lr=1e-3)

    for update_idx in range(3):
        batch = build_dummy_batch(config, device)
        carry, loss_value, metrics, act_steps = run_training_iteration(
            loss_head=loss_head,
            optimizer=optimizer,
            batch=batch,
            device=device,
            batch_size=config.batch_size,
        )

        count = max(metrics.get("count", torch.tensor(0, device=device)).item(), 1)
        mean_steps = metrics.get("steps", torch.tensor(act_steps * count, device=device)).item() / count

        print(f"Update {update_idx}: loss_per_sample={loss_value:.4f}, mean_steps={mean_steps:.2f}")
        print("  final steps:", carry.steps.detach().cpu().tolist())
        print("  halted flags:", carry.halted.detach().cpu().tolist())


if __name__ == "__main__":
    main()

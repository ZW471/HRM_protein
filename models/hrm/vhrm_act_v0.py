from typing import Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class VisualHierarchicalReasoningModel_ACTV0InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class VisualHierarchicalReasoningModel_ACTV0Carry:
    inner_carry: VisualHierarchicalReasoningModel_ACTV0InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class VisualHierarchicalReasoningModel_ACTV0Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    patch_dim: int = 0  # Will be auto-calculated from input data
    num_classes: int  # Number of output classes (e.g., 10 for CIFAR-10, 100 for CIFAR-100)

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class VisualHierarchicalReasoningModel_ACTV0Block(nn.Module):
    def __init__(self, config: VisualHierarchicalReasoningModel_ACTV0Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class VisualHierarchicalReasoningModel_ACTV0ReasoningModule(nn.Module):
    def __init__(self, layers: List[VisualHierarchicalReasoningModel_ACTV0Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class VisualHierarchicalReasoningModel_ACTV0_Inner(nn.Module):
    def __init__(self, config: VisualHierarchicalReasoningModel_ACTV0Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)

        # Patch projection layer: will be created lazily when we know input dimensions
        self.patch_projection = None

        self.lm_head = CastedLinear(self.config.hidden_size, self.config.num_classes, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            embed_init_std = 1.0 / self.embed_scale
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = VisualHierarchicalReasoningModel_ACTV0ReasoningModule(layers=[VisualHierarchicalReasoningModel_ACTV0Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = VisualHierarchicalReasoningModel_ACTV0ReasoningModule(layers=[VisualHierarchicalReasoningModel_ACTV0Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Input shape: (batch_size, seq_len, patch_dim)

        # Initialize patch projection layer on first use
        if self.patch_projection is None:
            patch_dim = input.shape[-1]
            self.config.patch_dim = patch_dim
            self.patch_projection = CastedLinear(patch_dim, self.config.hidden_size, bias=False)
            # Move to same device as input
            self.patch_projection = self.patch_projection.to(input.device)

        # Project patches to hidden dimension
        embedding = self.patch_projection(input.to(self.forward_dtype))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return VisualHierarchicalReasoningModel_ACTV0InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: VisualHierarchicalReasoningModel_ACTV0InnerCarry):
        return VisualHierarchicalReasoningModel_ACTV0InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: VisualHierarchicalReasoningModel_ACTV0InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[VisualHierarchicalReasoningModel_ACTV0InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad updates
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        new_carry = VisualHierarchicalReasoningModel_ACTV0InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # Heads: use high-level states so downstream logic matches ACTv1 while staying visual-specific
        puzzle_offset = self.puzzle_emb_len
        token_states = z_H[:, puzzle_offset:]
        lm_logits = self.lm_head(token_states)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]

        return new_carry, lm_logits, (q_halt_logits, q_continue_logits)


class VisualHierarchicalReasoningModel_ACTV0(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = VisualHierarchicalReasoningModel_ACTV0Config(**config)
        self.inner = VisualHierarchicalReasoningModel_ACTV0_Inner(self.config)

        # For compatibility with HRM interface
        self.puzzle_emb = self.inner.puzzle_emb if hasattr(self.inner, 'puzzle_emb') else None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return VisualHierarchicalReasoningModel_ACTV0Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: VisualHierarchicalReasoningModel_ACTV0Carry, batch: Dict[str, torch.Tensor], **kwargs) -> Tuple[VisualHierarchicalReasoningModel_ACTV0Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward through inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Enforce exploration with random minimum step requirement like ACTv1
                if self.config.halt_exploration_prob > 0:
                    min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q-values for training stability (no replay buffer)
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Updated carry
        new_carry = VisualHierarchicalReasoningModel_ACTV0Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )

        return new_carry, outputs

"""Bias-free SwiGLU packing and explicitly hindsight-only masking."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def dimensions(mlp: nn.Module) -> tuple[int, int]:
    """Reject incompatible tensors instead of silently approximating a new architecture."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(mlp, name, None)
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise ValueError(f"Expected a bias-free Linear {name}")
    width, hidden = mlp.gate_proj.weight.shape
    if mlp.up_proj.weight.shape != (width, hidden):
        raise ValueError("Mismatched gate/up dimensions")
    if mlp.down_proj.weight.shape != (hidden, width):
        raise ValueError("Mismatched down dimensions")
    if not callable(getattr(mlp, "act_fn", None)):
        raise ValueError("Missing activation function")
    return hidden, width


def validate_order(order: torch.Tensor, width: int) -> None:
    if order.dtype != torch.long or order.ndim != 1 or order.numel() != width:
        raise ValueError("Order must be an int64 permutation of every neuron")
    if not torch.equal(order.sort().values.cpu(), torch.arange(width)):
        raise ValueError("Order must contain each neuron exactly once")


@torch.no_grad()
def repack_(mlp: nn.Module, order: torch.Tensor) -> None:
    """Physically reorder paired projections; applying argsort(order) restores them."""
    _, width = dimensions(mlp)
    validate_order(order, width)
    order = order.to(mlp.gate_proj.weight.device)
    for name, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        weight = getattr(mlp, name).weight
        weight.copy_(weight.index_select(axis, order))


def group_bytes(hidden: int, neurons: int, element_size: int) -> int:
    if min(hidden, neurons, element_size) <= 0:
        raise ValueError("Dimensions and element size must be positive")
    return 3 * hidden * neurons * element_size


@torch.inference_mode()
def grouped_forward(mlp: nn.Module, x: torch.Tensor, group_width: int) -> torch.Tensor:
    """Evaluate every physical group and accumulate locally before output commitment.

    This is a correctness reference. It has no cache, transfers, or fast sparse kernel.
    FP32 accumulation avoids accumulating rounding error across hundreds of BF16 sums.
    """
    hidden, width = dimensions(mlp)
    if group_width <= 0:
        raise ValueError("group_width must be positive")
    result = torch.zeros((*x.shape[:-1], hidden), device=x.device, dtype=torch.float32)
    for start in range(0, width, group_width):
        stop = min(start + group_width, width)
        z = mlp.act_fn(F.linear(x, mlp.gate_proj.weight[start:stop]))
        z = z * F.linear(x, mlp.up_proj.weight[start:stop])
        result.add_(F.linear(z, mlp.down_proj.weight[:, start:stop]).float())
    return result.to(x.dtype)


def select_groups(scores: torch.Tensor, group_width: int, keep: float) -> torch.Tensor:
    """Top summed importance groups per token, including a short final group.

    Keeps ceil(keep * number_of_groups) groups, not an exact byte fraction.
    Stable sorting gives deterministic group-index tie breaking.
    """
    if group_width <= 0 or not math.isfinite(keep) or not 0 < keep <= 1:
        raise ValueError("Expected positive group_width and 0 < keep <= 1")
    width = scores.shape[-1]
    if width == 0:
        raise ValueError("Cannot select from zero neurons")
    count = math.ceil(width / group_width)
    padded = F.pad(scores, (0, count * group_width - width))
    grouped = padded.reshape(*scores.shape[:-1], count, group_width).sum(-1)
    selected = grouped.argsort(dim=-1, descending=True, stable=True)[..., :math.ceil(count * keep)]
    mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, selected, True)
    return mask.repeat_interleave(group_width, dim=-1)[..., :width]


class ImportanceCollector:
    """Calibration-only importance, computed AFTER both gate and up projections."""

    def __init__(self, mlp: nn.Module):
        _, width = dimensions(mlp)
        self.norms = mlp.down_proj.weight.detach().float().norm(dim=0)
        self.total = torch.zeros(width, device=self.norms.device, dtype=torch.float64)
        self.tokens = 0

    def __call__(self, module, args):
        z = args[0].detach().reshape(-1, self.norms.numel())
        self.total += (z.float().abs() * self.norms).double().sum(0)
        self.tokens += z.shape[0]

    def order(self) -> torch.Tensor:
        if not self.tokens:
            raise ValueError("No calibration tokens observed")
        return self.total.argsort(descending=True, stable=True).cpu()


class HindsightMask:
    """Dense gate/up, dense masked down: analytical quality probe, NEVER a speedup.

    The byte counter is hypothetical selected weight volume for all three projections,
    with no cache or reuse. It is not actual transfer traffic or reduced residency.
    """

    def __init__(self, mlp: nn.Module, group_width: int, keep: float):
        self.hidden, self.width = dimensions(mlp)
        if group_width <= 0 or not math.isfinite(keep) or not 0 < keep <= 1:
            raise ValueError("Invalid group selection settings")
        self.group_width, self.keep = group_width, keep
        self.norms = mlp.down_proj.weight.detach().float().norm(dim=0)
        self.bytes_per_neuron = group_bytes(self.hidden, 1, mlp.down_proj.weight.element_size())
        self.counts = torch.zeros(3, dtype=torch.float64, device=self.norms.device)

    def __call__(self, module, args):
        z = args[0]
        scores = z.float().abs() * self.norms
        mask = select_groups(scores, self.group_width, self.keep)
        self.counts[0] += mask.sum()
        self.counts[1] += z.numel() // self.width
        self.counts[2] += (scores * mask).double().sum()
        return (z * mask, *args[1:])

    def accounting(self) -> dict:
        selected, tokens, importance = self.counts.cpu().tolist()
        return {
            "token_observations": int(tokens),
            "hypothetical_selected_weight_bytes": int(selected) * self.bytes_per_neuron,
            "full_weight_bytes_for_same_observations": int(tokens) * self.width * self.bytes_per_neuron,
            "selected_importance_sum": importance,
        }

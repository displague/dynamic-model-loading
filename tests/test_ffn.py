import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dynamic_model_loading.ffn import (
    HindsightMask, ImportanceCollector, dimensions, group_bytes, grouped_forward, repack_, select_groups,
)
from dynamic_model_loading.experiment import hooks, layout


class ToyMLP(nn.Module):
    def __init__(self, hidden=7, width=13):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("group_width", [1, 4, 13, 20])
def test_packing_and_group_reconstruction_preserve_function(group_width):
    torch.manual_seed(42)
    mlp = ToyMLP()
    x = torch.randn(2, 3, 7)
    before = {name: value.clone() for name, value in mlp.state_dict().items()}
    expected = mlp(x)
    order = torch.randperm(13)
    with layout([mlp], [order]):
        torch.testing.assert_close(mlp(x), expected, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(grouped_forward(mlp, x, group_width), expected, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(mlp.gate_proj.weight, before["gate_proj.weight"][order], rtol=0, atol=0)
    for name, value in mlp.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_layout_and_hooks_restore_after_error():
    mlp = ToyMLP()
    before = mlp.gate_proj.weight.clone()
    with pytest.raises(RuntimeError):
        with layout([mlp], [torch.randperm(13)]), hooks([mlp], [HindsightMask(mlp, 4, 0.5)]):
            raise RuntimeError("test interruption")
    assert not mlp.down_proj._forward_pre_hooks
    torch.testing.assert_close(mlp.gate_proj.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("order", [torch.zeros(13, dtype=torch.long), torch.arange(12), torch.arange(13).float()])
def test_malformed_permutation_rejected_without_modification(order):
    mlp = ToyMLP()
    before = mlp.gate_proj.weight.clone()
    with pytest.raises(ValueError):
        repack_(mlp, order)
    torch.testing.assert_close(mlp.gate_proj.weight, before, rtol=0, atol=0)


def test_short_group_and_deterministic_ties():
    scores = torch.tensor([[1., 1., 4., 0., 10.], [1., 1., 1., 1., 2.]])
    mask = select_groups(scores, 2, 1/3)
    assert mask.tolist() == [[False, False, False, False, True], [True, True, False, False, False]]
    assert select_groups(scores, 2, 1).all()
    assert group_bytes(3584, 32, 2) == 672 * 1024


def test_mask_identity_and_accounting():
    mlp = ToyMLP()
    x = torch.randn(1, 5, 7)
    expected = mlp(x)
    mask = HindsightMask(mlp, 4, 1)
    with hooks([mlp], [mask]):
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)
    account = mask.accounting()
    assert account["token_observations"] == 5
    assert account["hypothetical_selected_weight_bytes"] == 5 * sum(p.numel() * p.element_size() for p in mlp.parameters())


def test_collector_sees_dense_intermediates():
    mlp = ToyMLP()
    x = torch.randn(1, 5, 7)
    collector = ImportanceCollector(mlp)
    with hooks([mlp], [collector]):
        mlp(x)
    z = F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    expected = (z.float().abs() * mlp.down_proj.weight.float().norm(dim=0)).double().sum((0, 1))
    torch.testing.assert_close(collector.total, expected)
    assert collector.tokens == 5


def test_unsupported_bias_fails():
    mlp = ToyMLP()
    mlp.down_proj = nn.Linear(13, 7, bias=True)
    with pytest.raises(ValueError):
        dimensions(mlp)


def test_individual_selection_is_permutation_equivariant_without_ties():
    torch.manual_seed(99)
    mlp = ToyMLP()
    x = torch.randn(1, 5, 7)
    with hooks([mlp], [HindsightMask(mlp, 1, 0.5)]):
        expected = mlp(x)
    with layout([mlp], [torch.randperm(13)]):
        with hooks([mlp], [HindsightMask(mlp, 1, 0.5)]):
            actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)

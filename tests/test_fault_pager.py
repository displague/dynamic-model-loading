import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dynamic_model_loading.fault_pager import PageCache, PageCatalog, PageKey, PagedDraft, SideIndex, paged_swiglu


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(3, 5, bias=False)
        self.up_proj = nn.Linear(3, 5, bias=False)
        self.down_proj = nn.Linear(5, 3, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_catalog_preserves_paired_short_tail_and_cache_is_lru():
    mlp = ToyMLP()
    catalog = PageCatalog(mlp, layer=2, width=2)
    assert catalog.pages == 3
    assert catalog.payload(2).gate.shape == (1, 3)
    cache = PageCache(catalog.payload(0).bytes * 2, "cpu")
    cache.get(PageKey(2, 0), catalog.payload(0), request="demand")
    cache.get(PageKey(2, 1), catalog.payload(1), request="demand")
    cache.get(PageKey(2, 0), catalog.payload(0), request="prefetch")
    cache.get(PageKey(2, 2), catalog.payload(2), request="demand")
    assert cache.events[2]["outcome"] == "hit"
    assert cache.events[-1]["evicted"] == [PageKey(2, 1)]
    assert cache.used_bytes <= cache.capacity_bytes


def test_side_index_is_deterministic_and_ties_choose_lowest_page():
    inputs = torch.tensor([[0., 0.], [3., 0.], [0., 4.], [2., 2.]])
    scores = torch.tensor([[1., 1., 0.], [0., 1., 2.], [2., 1., 0.], [0., 2., 1.]])
    index = SideIndex.build(inputs, scores, medoids=2)
    assert index.centroids.tolist() == [[0., 0.], [0., 4.]]
    assert index.select(torch.tensor([[0.1, 0.1], [0., 3.9]]), 2).tolist() == [[0, 1], [0, 1]]
    assert index.bytes > 0


def test_all_pages_reconstruct_toy_mlp_and_subset_is_physical_approximation():
    torch.manual_seed(3)
    mlp, x = ToyMLP(), torch.randn(1, 2, 3)
    catalog = PageCatalog(mlp, layer=0, width=2)
    all_pages = torch.tensor([[0, 1, 2], [0, 1, 2]])
    full = paged_swiglu(x, catalog, PageCache(4096, "cpu"), all_pages, F.silu)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(full, expected, rtol=1e-5, atol=1e-6)
    subset = paged_swiglu(x, catalog, PageCache(4096, "cpu"), torch.tensor([[0], [0]]), F.silu)
    assert not torch.equal(subset, expected)


def test_paged_draft_replaces_and_restores_ffn_forward():
    torch.manual_seed(7)
    mlp, x = ToyMLP(), torch.randn(1, 2, 3)
    indexes = [SideIndex(torch.zeros(1, 3), torch.tensor([[0, 1, 2]]))]
    draft = PagedDraft([mlp], indexes, page_width=2, selected_pages=3, capacity_bytes=4096, device="cpu")
    expected = mlp(x)
    draft.restore()
    dense = mlp(x)
    torch.testing.assert_close(expected, dense, rtol=1e-5, atol=1e-6)
    assert any(event["outcome"] == "selection_copy" for event in draft.cache.events)
    assert draft.controller_bytes > 0


def test_paged_draft_rejects_every_index_before_mutating_a_module():
    first, second = ToyMLP(), ToyMLP()
    original = first.forward
    valid = SideIndex(torch.zeros(1, 3), torch.tensor([[0, 1, 2]]))
    invalid = SideIndex(torch.zeros(1, 3), torch.tensor([[0, 1]]))
    with pytest.raises(ValueError):
        PagedDraft([first, second], [valid, invalid], page_width=2, selected_pages=3,
                   capacity_bytes=4096, device="cpu")
    assert first.forward == original

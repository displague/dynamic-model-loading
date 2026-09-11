import torch

from dynamic_model_loading.corpus import articles
from dynamic_model_loading.experiment import hooks
from dynamic_model_loading.packing import CoactivationCollector, balanced_coactivation_order
from test_ffn import ToyMLP


def test_bisection_recovers_disjoint_coactivation_groups():
    # Four groups have identical signatures within a group and disjoint support.
    signatures = torch.eye(4).repeat_interleave(4, dim=1)
    shuffled = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
    samples = signatures[:, shuffled]
    order = balanced_coactivation_order(samples, group_width=4, sketch_dim=64, seed=7)
    groups = (shuffled[order] // 4).reshape(4, 4)
    assert all(len(set(group.tolist())) == 1 for group in groups)
    assert torch.equal(order, balanced_coactivation_order(samples, 4, 64, seed=7))


def test_zero_profiles_and_partial_groups_preserve_every_neuron():
    order = balanced_coactivation_order(torch.zeros(8, 13), group_width=4)
    assert torch.equal(order, torch.arange(13))


def test_reservoir_selects_exact_highest_random_priorities():
    torch.manual_seed(4)
    mlp = ToyMLP()
    x = torch.randn(1, 21, 7)
    collector = CoactivationCollector(mlp, capacity=5, seed=9)
    with hooks([mlp], [collector]):
        for chunk in x.split([3, 7, 11], dim=1):
            mlp(chunk)
    expected_ordinals = torch.rand(21, generator=torch.Generator().manual_seed(9), dtype=torch.float64).argsort(descending=True)[:5]
    assert set(collector.ordinals.tolist()) == set(expected_ordinals.tolist())
    z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    expected = z[0, collector.ordinals].float().abs() * collector.norms
    torch.testing.assert_close(collector.samples, expected)
    assert collector.tokens == 21


def test_articles_keep_subsections_inside_document():
    found = list(articles(["", " = First = ", "", "paragraph", " = = Subheading = = ",
                           "more", "", " = Second = ", "", "last"]))
    assert found == [("First", "paragraph\n= = Subheading = =\nmore"), ("Second", "last")]


def test_articles_do_not_split_inline_equations_or_table_legends():
    lines = ["", " = First = ", "", "Note : GP", " = Games Played ; TOI = ",
             "Time On Ice", "", " = and = ", "a continuation", "",
             " = Second = ", "", "last"]
    found = list(articles(lines))
    assert found == [("First", "Note : GP\n= Games Played ; TOI =\nTime On Ice\n= and =\na continuation"),
                     ("Second", "last")]

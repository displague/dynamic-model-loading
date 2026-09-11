import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dynamic_model_loading.cache import dense_static_frontier, group_arrays, hot_mask, replay
from dynamic_model_loading.cache_analysis import validate_gates
from dynamic_model_loading.ffn import select_groups
from dynamic_model_loading.trace import TraceRecorder, load_trace, save_trace


def test_applied_zero_tie_masks_and_tail_geometry(tmp_path):
    scores = torch.zeros(1, 2, 5)
    observer = TraceRecorder(2, 0.5)
    observer(scores, select_groups(scores, 2, 0.5))
    receipt = save_trace(tmp_path / "mask.npz", [observer], {"document": "d"})
    mask = load_trace(tmp_path / "mask.npz", receipt)
    assert mask.tolist() == [[[True, True, False]], [[True, True, False]]]
    geometry = receipt["geometry_per_layer"][0]
    assert geometry == dict(top_individual_neurons=6, top_individual_neurons_in_applied_mask=6,
                            neurons_in_individual_cover_groups=8, applied_neurons=8)
    with pytest.raises(ValueError, match="Missing"):
        observer.take()
    with (tmp_path / "mask.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_trace(tmp_path / "mask.npz", receipt)


def test_tail_bytes_and_static_preload_charge_unused_pages():
    sizes, importance = group_arrays(np.array([[1., 1., 9.]]), np.array([[0, 1, 2]]), 2, 10)
    assert sizes.tolist() == [[20, 10]]
    trace = np.array([[[True, False]], [[True, False]]])
    cold, warm = replay(trace, sizes, importance, 40, "static_global")
    assert cold["preload_bytes"] == 10  # The never-requested hot tail still costs a preload.
    assert cold["total_bytes"] == 50
    assert warm["total_bytes"] == 40
    assert cold["incoming_reserve_bytes"] == 20
    assert cold["retained_slots"] == 1


def test_equal_layer_hot_set_uses_stable_quotas():
    sizes = np.ones((3, 4), dtype=np.int64)
    importance = np.array([[10, 10, 10, 10], [1, 1, 1, 1], [0, 0, 0, 0]])
    mask = hot_mask(importance, sizes, 5, "static_equal_layer")
    assert mask.tolist() == [[True, True, False, False], [True, True, False, False], [True, False, False, False]]


@pytest.mark.parametrize("seed", range(8))
def test_lru_bound_agrees_with_explicit_cold_and_wraparound(seed):
    rng = np.random.default_rng(seed)
    trace = np.zeros((5, 4, 7), dtype=bool)
    for token in trace:
        for layer in token:
            layer[rng.choice(7, 4, replace=False)] = True
    sizes = np.full((4, 7), 10, dtype=np.int64)
    fast = replay(trace, sizes, sizes, 90, "lru")  # eight retained pages, >=12 distinct intervening pages
    exact = replay(trace, sizes, sizes, 90, "lru", force_explicit=True)
    assert fast[0]["method"] == "distinct_other_layer_bound"
    for a, b in zip(fast, exact):
        assert a["total_bytes"] == b["total_bytes"] == 800
        assert a["demand_hits"] == b["demand_hits"] == 0


def test_lru_fallback_retains_warm_hits_and_consumes_hits_before_misses():
    trace = np.array([[[False, True, True]], [[True, False, True]]])
    sizes = np.ones((1, 3), dtype=np.int64)
    cold, warm = replay(trace, sizes, sizes, 3, "lru")
    assert cold["method"] == "explicit_lru"
    assert cold["demand_hits"] == 1
    assert warm["demand_hits"] == 2
    assert cold["total_bytes"] == 3
    assert warm["total_bytes"] == 2


def test_gate_ledger_requires_coverage_and_recomputes_pass_flags():
    cfg = dict(layouts=["native"], reconstruction_relative_l2_max=0.01,
               logit_relative_l2_max=0.01, logit_mean_kl_max=0.001)
    checks = [dict(kind="group_reconstruction", layer=0, layout="native", relative_l2=0.0, passed=True),
              dict(kind="layout_correctness", document="d", layout="native", logit_relative_l2=0.0,
                   mean_kl_dense_to_candidate=0.0, passed=True)]
    validate_gates(checks, cfg, ["d"], 1)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_gates(checks[:1], cfg, ["d"], 1)
    checks[1]["mean_kl_dense_to_candidate"] = 0.1
    with pytest.raises(ValueError, match="tolerance"):
        validate_gates(checks, cfg, ["d"], 1)


def test_dense_frontier_can_choose_smaller_transfer_group_than_sparse_condition():
    means = np.ones((2, 8))
    frontier = dense_static_frontier(means, {"native": [list(range(8))]*2}, [1, 4], [9], 1, [3, 2])
    warm = frontier[(9, "warm")]
    assert warm["group_width"] == 1
    assert warm["total_bytes"] == 40  # 8 retained bytes; width 4 could retain only 4.
    assert frontier[(9, "cold")]["total_bytes"] == 56

import math

import pytest
import torch

from dynamic_model_loading.metrics import aggregate, compare_logits


def test_metrics_use_shifted_targets_and_exclude_last_position():
    logits = torch.tensor([[[2., 0.], [0., 2.], [90., -90.]]])
    ids = torch.tensor([[1, 0, 1]])
    changed = logits.clone()
    changed[:, -1] = -changed[:, -1]
    result = compare_logits(logits, changed, ids)
    assert result["predicted_tokens"] == 2
    assert result["mean_kl_dense_to_candidate"] == 0
    assert result["top1_agreement"] == 1
    assert result["relative_perplexity"] == 1
    assert result["dense_nll"] == pytest.approx(math.log1p(math.exp(-2)), rel=1e-6)


def test_aggregate_weights_by_tokens_not_documents():
    rows = [{"predicted_tokens": n, "mean_kl_dense_to_candidate": x, "top1_agreement": 1,
             "dense_nll": x, "candidate_nll": x + 0.1} for n, x in [(1, 2.), (9, 4.)]]
    result = aggregate(rows)
    assert result["dense_nll"] == 3.8
    assert result["relative_perplexity"] == pytest.approx(math.exp(0.1))


def test_nonfinite_logits_fail():
    logits = torch.tensor([[[float("nan"), 0.], [0., 1.]]])
    with pytest.raises(ValueError, match="Non-finite"):
        compare_logits(logits, logits, torch.tensor([[0, 1]]))

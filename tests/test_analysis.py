import pytest

from dynamic_model_loading.analysis import paired_kl_interval


def test_paired_bootstrap_retains_document_pairing():
    rows = [dict(kind="hindsight_document", layout=layout, group_width=32, keep_fraction=0.75,
                 document=str(i), predicted_tokens=10 + i, mean_kl_dense_to_candidate=value + i)
            for layout, value in [("popularity", 1.), ("coactivation", 0.5)] for i in range(3)]
    result = paired_kl_interval(rows)
    assert result["delta"] == -0.5
    assert result["percentile_95_interval"] == [-0.5, -0.5]
    with pytest.raises(ValueError, match="identical"):
        paired_kl_interval(rows[:-1])

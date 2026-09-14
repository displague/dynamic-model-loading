import json
from pathlib import Path

import pytest
import torch

from dynamic_model_loading.fault_pager_study import (
    mib_to_bytes, shuffled_conditions, validate_config, verification_commit,
)


def config():
    return json.loads((Path(__file__).parents[1] / "configs" / "fault-pager.json").read_text())


def test_frozen_config_is_accepted_and_rejects_changed_page_choice():
    cfg = config()
    validate_config(cfg)
    cfg["selected_pages"] = 26
    with pytest.raises(ValueError, match="frozen"):
        validate_config(cfg)
    cfg = config()
    cfg["corpus_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen"):
        validate_config(cfg)


def test_verifier_commits_only_common_prefix_and_target_fallback():
    result = verification_commit(torch.tensor([4, 5, 6, 7]), torch.tensor([4, 9, 8, 7]), eos_token_id=0)
    assert result["accepted"] == 1
    assert result["attempted"] == 4
    assert result["fallback"] == 9
    assert result["committed"].tolist() == [4, 9]
    assert result["stop_reason"] == "rejected"
    accepted = verification_commit(torch.tensor([4, 5, 6, 7]), torch.tensor([4, 5, 6, 7]), eos_token_id=0)
    assert accepted["accepted"] == 4 and accepted["attempted"] == 4 and accepted["fallback"] is None
    assert accepted["committed"].tolist() == [4, 5, 6, 7]


def test_verifier_stops_on_target_eos_and_rejects_non_token_predictions():
    result = verification_commit(torch.tensor([4, 0, 7, 8]), torch.tensor([4, 0, 7, 8]), eos_token_id=0)
    assert result["committed"].tolist() == [4, 0]
    assert result["accepted"] == 2 and result["stop_reason"] == "eos"
    with pytest.raises(ValueError, match="torch.long"):
        verification_commit(torch.tensor([4, 5, 6, 7]), torch.tensor([4., 5., 6., 7.]), eos_token_id=0)
    with pytest.raises(ValueError, match="frozen-length"):
        verification_commit(torch.tensor([4, 5]), torch.tensor([4, 5]), eos_token_id=0)


def test_condition_order_is_fixed_complete_and_mib_accounting_is_exact():
    first = shuffled_conditions(20260914, 0)
    assert first == shuffled_conditions(20260914, 0)
    assert set(first) == {(mode, budget) for mode in ("eager", "lru", "prefetch") for budget in (128, 512)}
    assert mib_to_bytes(128) == 128 * 1024 * 1024

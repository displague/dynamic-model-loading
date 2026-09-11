import copy
import json
from pathlib import Path
import shutil

import pytest
import torch
from safetensors.torch import save_file

from dynamic_model_loading.environment_analysis import load_ordinary_logits, validate_pair, validate_rows
from dynamic_model_loading.experiment import digest


def test_archived_arithmetic_ledger_is_complete_and_fail_closed():
    root = Path("results/packing-pilot-20260911/precision-receipted")
    read = lambda name: json.loads((root / name).read_text())
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    cfg, tokens, summary = read("config.json"), read("token-ids.json"), read("summary.json")
    validate_rows(rows, summary, cfg, tokens)
    with pytest.raises(ValueError, match="original numerical limits"):
        validate_rows(rows, summary, {**cfg, "logit_mean_kl_max": 0.9}, tokens)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_rows(rows[:-1], summary, cfg, tokens)
    altered = copy.deepcopy(rows)
    altered[0]["within_original_limits"] = not altered[0]["within_original_limits"]
    with pytest.raises(ValueError, match="flag"):
        validate_rows(altered, summary, cfg, tokens)
    altered = copy.deepcopy(rows)
    altered[0]["predicted_tokens"] += 1
    with pytest.raises(ValueError, match="count"):
        validate_rows(altered, summary, cfg, tokens)
    altered_summary = copy.deepcopy(summary)
    altered_summary[0]["max_mean_kl"] += 1
    with pytest.raises(ValueError, match="aggregate"):
        validate_rows(rows, altered_summary, cfg, tokens)


def test_control_pair_rejects_config_and_execution_setting_drift():
    cfg = {"cpu_threads": 4, "seed": 1729, "expected_versions_sha256": "a"}
    other = {**cfg, "expected_versions_sha256": "b"}
    env = dict(python="3.14", platform="windows", device="cuda", cuda_build="13", versions={},
               tf32_matmul=False, tf32_cudnn=False, bf16_reduced_precision_reduction=True,
               fp16_reduced_precision_reduction=True, cpu_threads=4, gpu={"name": "same"})
    validate_pair(cfg, other, env, env)
    with pytest.raises(ValueError, match="config drift"):
        validate_pair(cfg, {**other, "seed": 1}, env, env)
    with pytest.raises(ValueError, match="settings drift"):
        validate_pair(cfg, other, env, {**env, "cpu_threads": 8})


def test_dense_logits_bundle_cannot_be_swapped_between_environments(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    tokens = {"d": [[1, 2]]}
    for root in (a, b):
        root.mkdir()
        (root / "manifest.json").write_text(json.dumps({"environment": root.name}))
        save_file({"d": torch.zeros(1, 2, 4)}, root / "ordinary-logits.safetensors",
                  metadata={"manifest_sha256": digest(root / "manifest.json"), "comparison": "ordinary_native_bf16"})
        (root / "ordinary-logits-receipt.json").write_text(json.dumps({"documents": ["d"],
                  "sha256": digest(root / "ordinary-logits.safetensors")}))
        load_ordinary_logits(root, tokens)
    for name in ("ordinary-logits.safetensors", "ordinary-logits-receipt.json"):
        shutil.copyfile(a / name, b / name)
    with pytest.raises(ValueError, match="provenance"):
        load_ordinary_logits(b, tokens)

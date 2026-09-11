import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.experiment import layout
from dynamic_model_loading.precision import arithmetic, save_inputs, validate_control


def test_precision_inputs_preserve_exact_token_and_layout_receipts(tmp_path):
    tokens = {"document": torch.tensor([[3, 8, 2]])}
    manifest = save_inputs(tmp_path, tokens, [torch.arange(3)], [torch.tensor([2, 0, 1])])
    assert json.loads((tmp_path / "token-ids.json").read_text()) == {"document": [[3, 8, 2]]}
    assert json.loads((tmp_path / "layouts.json").read_text()) == {"native": [[0, 1, 2]], "random": [[2, 0, 1]]}
    assert manifest["layouts_sha256"] == hashlib.sha256((tmp_path / "layouts.json").read_bytes()).hexdigest()
    assert manifest["token_ids_file_sha256"] == hashlib.sha256((tmp_path / "token-ids.json").read_bytes()).hexdigest()


def test_canonical_down_preserves_order_and_restores_forward():
    torch.manual_seed(5)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    mlps = extract_ffns(model)
    orders = [torch.randperm(32) for m in mlps]
    x = torch.randn(1, 4, 16)
    expected = mlps[0](x)
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with pytest.raises(RuntimeError):
        with layout(mlps, orders), arithmetic(mlps, "canonical_down", orders):
            torch.testing.assert_close(mlps[0](x), expected, rtol=0, atol=0)
            raise RuntimeError("interrupt")
    assert "forward" not in mlps[0].down_proj.__dict__
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
    torch.testing.assert_close(mlps[0](x), expected, rtol=0, atol=0)


def test_environment_control_rejects_unrelated_dependency_drift(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("fixture")
    versions = {"torch": "2.12.0+cu130", "transformers": "5.13.1"}
    cfg = {"environment_control": True, "expected_python": "same interpreter",
           "logit_relative_l2_max": 0.01, "logit_mean_kl_max": 0.001,
           "expected_versions_sha256": hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest(),
           "expected_corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()}
    values = {"python": "same interpreter", "versions": versions}
    validate_control(cfg, values, corpus)
    versions["transformers"] = "changed"
    with pytest.raises(ValueError, match="inventory"):
        validate_control(cfg, values, corpus)


def test_precision_interrupt_keeps_failure_receipt(tmp_path, monkeypatch):
    from dynamic_model_loading import precision
    output = tmp_path / "interrupted"
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(precision, "_run_precision", interrupted)
    with pytest.raises(KeyboardInterrupt):
        precision.run_precision("unused", "unused", output)
    assert json.loads((output / "failure.json").read_text())["error_type"] == "KeyboardInterrupt"
    before = (output / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        precision.run_precision("unused", "unused", output)
    assert (output / "failure.json").read_bytes() == before


def test_missing_cuda_preserves_preflight_failure(tmp_path, monkeypatch):
    from dynamic_model_loading import precision
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "no-cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        precision.run_precision("configs/smoke.json", "data/smoke.jsonl", output)
    assert json.loads((output / "failure.json").read_text())["error_type"] == "RuntimeError"
import hashlib
import json

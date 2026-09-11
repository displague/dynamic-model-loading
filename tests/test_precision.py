import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.experiment import layout
from dynamic_model_loading.precision import arithmetic, save_inputs


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
import hashlib
import json

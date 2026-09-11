import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM, Lfm2Config, Lfm2ForCausalLM, Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.experiment import layout
from dynamic_model_loading.ffn import dimensions, grouped_forward


@pytest.mark.parametrize("kind", ["qwen2", "granite", "lfm2"])
def test_tiny_real_architecture_repacking(kind):
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    torch.manual_seed(12)
    if kind == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**common))
    elif kind == "granite":
        model = GraniteForCausalLM(GraniteConfig(**common))
    else:
        model = Lfm2ForCausalLM(Lfm2Config(**common, block_auto_adjust_ff_dim=False,
                                          layer_types=["conv", "full_attention"]))
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mlps = extract_ffns(model)
    with torch.inference_mode():
        expected = model(input_ids=ids, use_cache=False).logits
        with layout(mlps, [torch.randperm(dimensions(m)[1]) for m in mlps]):
            torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, expected, rtol=2e-5, atol=1e-6)
            for mlp in mlps:
                x = torch.randn(1, 2, 32)
                torch.testing.assert_close(grouped_forward(mlp, x, 7), mlp(x), rtol=2e-5, atol=1e-6)

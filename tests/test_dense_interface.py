from types import SimpleNamespace
import json

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from safetensors.torch import save_file, load_file

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.causal_study import packed
from dynamic_model_loading.dense_interface import aligned_metrics, decoding, full_retention, incremental_generate, prepare, score


def test_scoring_keeps_correctness_and_format_separate():
    task = {'domain': 'arithmetic', 'answer': '312', 'syntax': 'integer'}
    assert score('312\nHere is why.', task) == dict(first_line='312', answer_correct=True, format_compliant=False, success=False)
    assert not score('The answer is 312', task)['answer_correct']
    assert score(' 312\n', task)['success']
    assert score('99', task)['format_compliant']
    assert not score('99', task)['success']
    assert not score('```312```', task)['answer_correct']
    task = {'domain': 'copying', 'answer': 'Q_a', 'syntax': 'identifier'}
    assert not score('Q_a\nExplanation', task)['answer_correct']
    assert score('Q_a', task)['success']


class FakeModel:
    def __init__(self): self.calls = 0
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.calls += 1
        # Emit EOS immediately, exposing accidental execution beyond the stop.
        return SimpleNamespace(logits=torch.tensor([[[0., 1., 9.]]]), past_key_values=self.calls)


def test_incremental_stops_at_eos_and_rejects_post_eos_replay():
    model = FakeModel(); config = GenerationConfig(max_new_tokens=64, eos_token_id=[2], pad_token_id=0)
    ids, logits = incremental_generate(model, torch.tensor([[0, 1]]), config)
    assert ids == [2] and model.calls == 2 and logits.shape == (1, 3)
    with pytest.raises(ValueError, match='post-EOS'):
        incremental_generate(model, torch.tensor([[0]]), config, forced=[2, 1])


def test_incremental_compares_same_prefix_not_diverged_generations():
    config = GenerationConfig(max_new_tokens=3, eos_token_id=[2], pad_token_id=0)
    ids, logits = incremental_generate(FakeModel(), torch.tensor([[0]]), config, forced=[0, 1, 2])
    assert ids == [0, 1, 2] and logits.argmax(-1).tolist() == [2, 2, 2]


def test_native_incremental_and_full_retention_tiny_model():
    torch.manual_seed(1729)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, attention_dropout=0.)).eval()
    config = GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=[23], pad_token_id=0)
    prompt = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        ref = model.generate(prompt, attention_mask=torch.ones_like(prompt), generation_config=config,
            output_logits=True, return_dict_in_generate=True)
        ref_ids = ref.sequences[0, 3:].tolist()
        native = torch.cat(list(ref.logits), 0)
        ids, values = incremental_generate(model, prompt, config)
        assert ids == ref_ids and aligned_metrics(native, values)['relative_l2'] < .00001
        mlps = extract_ffns(model)
        originals = [m.down_proj.weight.clone() for m in mlps]
        with packed(mlps, [list(reversed(range(32)))]*2):
            with full_retention(mlps) as counts:
                ids, values = incremental_generate(model, prompt, config, forced=ref_ids)
                assert ids == ref_ids and aligned_metrics(native, values)['relative_l2'] < .00001
            assert counts == [(3+len(ids)-1)*4]*2
        assert all(torch.equal(m.down_proj.weight, original) for m,original in zip(mlps,originals))
        with pytest.raises(RuntimeError, match='fixture'):
            with full_retention(mlps): raise RuntimeError('fixture')
        assert all(not m.down_proj._forward_pre_hooks for m in mlps)


def test_nonfinite_is_explicit_failure():
    reference = torch.ones(2, 3)
    bad = reference.clone(); bad[-1, -1] = float('nan')
    assert aligned_metrics(reference, bad) is None
    with pytest.raises(ValueError, match='shape'):
        aligned_metrics(reference, reference[:1])
    extreme = torch.tensor([[3e38, -3e38]], dtype=torch.float32)
    assert aligned_metrics(extreme, extreme.clone())['mean_kl'] == 0.


def test_chat_wrapper_and_explicit_generation_defaults(tmp_path):
    backend = Tokenizer(models.WordLevel({'[UNK]':0, 'synthetic':1}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}")
    assert prepare(tokenizer, {'task':'synthetic'}, 'chat').ndim == 2
    (tmp_path/'generation_config.json').write_text(json.dumps(dict(eos_token_id=[23],pad_token_id=0,
        bos_token_id=1,repetition_penalty=1.1,do_sample=True,temperature=.7,top_p=.8,top_k=20)))
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    model.generation_config = GenerationConfig.from_dict(json.loads((tmp_path/'generation_config.json').read_text()))
    requested = decoding(tmp_path)
    effective, _ = model._prepare_generation_config(requested)
    assert effective.repetition_penalty == 1. and not effective.do_sample
    assert effective.to_dict() == requested.to_dict()
    logits = torch.ones(2, 3)
    save_file({'own_logits':logits, 'aligned_logits':logits.clone()}, tmp_path/'native.safetensors')
    saved = load_file(tmp_path/'native.safetensors')
    assert torch.equal(saved['own_logits'], saved['aligned_logits'])

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

from dynamic_model_loading import relu, relu_study
from dynamic_model_loading.experiment import digest


def layer_fixture():
    torch.manual_seed(23)
    layer = nn.Module()
    layer.fc1, layer.fc2 = nn.Linear(3, 5), nn.Linear(5, 3)
    layer.fc2.bias.data.copy_(torch.tensor([2., -3., 4.]))
    return layer


def test_bias_permutation_and_tail_reconstruction_restore_on_error():
    layer = layer_fixture()
    x = torch.randn(2, 3)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    expected = relu.dense_forward(layer, x)
    order = torch.tensor([4, 2, 0, 3, 1])
    with pytest.raises(RuntimeError):
        with relu.layout([layer], [order]):
            torch.testing.assert_close(layer.fc1.bias, before['fc1.bias'][order])
            torch.testing.assert_close(layer.fc2.bias, before['fc2.bias'])
            torch.testing.assert_close(relu.dense_forward(layer, x), expected)
            torch.testing.assert_close(relu.grouped_forward(layer, x, 2), expected)
            raise RuntimeError("interrupt")
    for k, v in layer.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_exact_zero_tail_accounting_always_charges_output_bias():
    layer = layer_fixture()
    probe = relu.Probe(layer, 2, "exact_zero")
    z = torch.tensor([[0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.]])
    masked = probe(layer.fc2, (z,))[0]
    torch.testing.assert_close(masked, z, rtol=0, atol=0)
    account = probe.take()
    assert account['selected_neurons'] == 1
    assert account['selected_group_weight_bytes'] == (2*3+1)*4
    assert account['fixed_output_bias_bytes'] == 2*3*4
    assert account['hypothetical_selected_ffn_bytes'] == 52
    assert account['zero_groups'] == 5
    assert account['zero_neurons'] == 9
    torch.testing.assert_close(layer.fc2(masked)[1], layer.fc2.bias)


@pytest.mark.parametrize('fault', ['allocation', 'after_mutation'])
def test_partial_repacking_failure_keeps_all_original_tensors(monkeypatch, fault):
    layer = layer_fixture()
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    fired = False
    if fault == 'allocation':
        original = torch.Tensor.index_select
        def fail(self, dim, index):
            nonlocal fired
            if not fired and dim == 1 and tuple(self.shape) == (3, 5):
                fired = True
                raise RuntimeError('allocation failure')
            return original(self, dim, index)
        monkeypatch.setattr(torch.Tensor, 'index_select', fail)
    else:
        original = torch.Tensor.copy_
        def fail(self, source, *args, **kwargs):
            nonlocal fired
            result = original(self, source, *args, **kwargs)
            if not fired and self.data_ptr() == layer.fc2.weight.data_ptr():
                fired = True
                raise KeyboardInterrupt()
            return result
        monkeypatch.setattr(torch.Tensor, 'copy_', fail)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        with relu.layout([layer], [torch.tensor([4, 3, 2, 1, 0])]):
            pass
    assert fired
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('when', ['before_copy', 'after_copy'])
def test_cleanup_interruption_restores_every_layer_before_propagating(monkeypatch, when):
    layers = [layer_fixture(), layer_fixture()]
    before = [{k: v.clone() for k, v in layer.state_dict().items()} for layer in layers]
    original = torch.Tensor.copy_
    armed = fired = False
    def interrupt(self, source, *args, **kwargs):
        nonlocal fired
        fail = armed and not fired and self.data_ptr() == layers[1].fc2.weight.data_ptr()
        if fail and when == 'before_copy':
            fired = True
            raise KeyboardInterrupt()
        result = original(self, source, *args, **kwargs)
        if fail:
            fired = True
            raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(torch.Tensor, 'copy_', interrupt)
    with pytest.raises(KeyboardInterrupt):
        with relu.layout(layers, [torch.tensor([4, 2, 0, 3, 1])]*2):
            armed = True
    assert fired
    for layer, snapshot in zip(layers, before):
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


@pytest.fixture
def tiny_relu(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = OPTForCausalLM(OPTConfig(vocab_size=16, hidden_size=16, ffn_dim=24,
        num_hidden_layers=2, num_attention_heads=2, word_embed_proj_dim=16, dropout=0.0))
    model.config.save_pretrained(checkpoint)
    torch.save(model.state_dict(), checkpoint / "pytorch_model.bin")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(checkpoint)
    monkeypatch.setattr(relu_study, "snapshot_download", lambda *a, **kw: str(checkpoint))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('\n'.join(json.dumps(dict(id=name, split=split, domain="fixture", text=text)) for name, split, text in
        [("cal", "calibration", "a b c d a"), ("diag", "diagnostic", "d c b a")]))
    cfg = json.loads(Path("configs/smoke.json").read_text())
    cfg.update(model="facebook/opt-1.3b",dtype="float32",expected_torch=str(torch.__version__),
        expected_corpus_sha256=digest(corpus),expected_weights_sha256=digest(checkpoint/'pytorch_model.bin'),
        expected_shapes=[[16,24],[16,24]],group_widths=[1,7],keep_fractions=[.75],
        reconstruction_group_width=7,protocol="docs/relu-control-protocol.md")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    return config, corpus, tmp_path / "run"


def test_real_opt_fixture_exact_zero_and_approximate_receipts(tiny_relu):
    result = relu_study.run(*tiny_relu, device="cpu")
    assert result['status'] == 'relu_control_completed'
    assert result['exact_zero_passed']
    assert len(result['probes']) == 12
    zero = [r for r in result['probes'] if r['mode']=='exact_zero']
    assert all(r['relative_perplexity'] == pytest.approx(1.0, abs=1e-6) for r in zero)
    for r in result['probes']:
        a=r['accounting']
        assert a['fixed_output_bias_bytes']==4*16*4*2
        assert a['hypothetical_selected_ffn_bytes']==a['selected_group_weight_bytes']+a['fixed_output_bias_bytes']
    with pytest.raises(FileExistsError):
        relu_study.run(*tiny_relu, device="cpu")


def test_relu_gate_failure_blocks_every_probe(tiny_relu, monkeypatch):
    monkeypatch.setattr(relu, 'grouped_forward', lambda layer,x,width: relu.dense_forward(layer,x)*0)
    result=relu_study.run(*tiny_relu,device='cpu')
    assert result['status']=='correctness_gate_failed'
    assert result['probes']==[]


def test_exact_zero_failure_blocks_approximate_omission(tiny_relu, monkeypatch):
    original=relu.Probe.__call__
    def corrupt(self,module,args):
        result=original(self,module,args)
        return (torch.ones_like(result[0])*100,*result[1:]) if self.mode=='exact_zero' else result
    monkeypatch.setattr(relu.Probe,'__call__',corrupt)
    result=relu_study.run(*tiny_relu,device='cpu')
    assert result['status']=='exact_zero_gate_failed'
    assert {r['mode'] for r in result['probes']}=={'exact_zero'}

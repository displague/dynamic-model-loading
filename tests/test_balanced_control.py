import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dynamic_model_loading.balanced_control import PREFIX, Probe, ffn_volume, masks, prepare_ids, save_masks, score_generation, validate_config
from dynamic_model_loading.balanced_analysis import checked_masks
from dynamic_model_loading.balanced_analysis import generated_tokens
from dynamic_model_loading.balanced_control import reconstruction_metric
from dynamic_model_loading.balanced_control import generate_checked
from dynamic_model_loading.balanced_analysis import generation_valid
from dynamic_model_loading.balanced_control import quality_metrics, quality_aggregate


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_local_reconstruction_is_an_explicit_failed_measurement(bad):
    assert reconstruction_metric(torch.ones(2,3),torch.full((2,3),bad))==(None,'nonfinite_reconstruction')
    assert reconstruction_metric(torch.ones(2,3),torch.ones(2,3))==(0.,None)


def test_finite_zero_reference_with_nonzero_residual_is_an_explicit_failed_metric():
    assert reconstruction_metric(torch.zeros(2,3),torch.ones(2,3))==(None,'nonfinite_reconstruction_metric')


@pytest.mark.parametrize('position',[0,2])
def test_nonfinite_teacher_logits_are_retained_as_invalid_metrics_without_drop(position):
    good=torch.ones(1,3,5);bad=good.clone();bad[0,position,0]=float('nan');ids=torch.tensor([[1,2,3]])
    assert quality_metrics(good,bad,ids) is None
    assert quality_metrics(bad,good,ids) is None
    metric=quality_metrics(good,good,ids)
    assert quality_aggregate([metric,None]) is None
    assert quality_aggregate([metric,metric])['predicted_tokens']==4


def test_nonfinite_generation_preserves_ids_and_auditable_failure_tensors(tmp_path):
    from types import SimpleNamespace
    class Model(torch.nn.Module):
        def forward(self,input_ids,**kwargs):
            logits=torch.zeros(1,1,8);logits[:,:,3]=float('nan')
            return SimpleNamespace(logits=logits,past_key_values=None)
    model=Model();(tmp_path/'references').mkdir()
    ids,receipt=generate_checked(model,torch.tensor([[1,2]]),tmp_path,'fixture')
    assert ids==[1,2]+[3]*32 and not model._forward_hooks
    assert len(receipt['finite_steps'])==33 and not any(receipt['finite_steps'])
    assert not generation_valid(tmp_path,receipt,33,8)
    receipt['nonfinite'].pop()
    with pytest.raises(ValueError,match='inventory'):generation_valid(tmp_path,receipt,33,8)


@pytest.mark.parametrize('mutation',['empty','truncated','prefix','float','vocabulary'])
def test_failed_layout_generations_still_require_complete_valid_ids(mutation):
    prompt=torch.tensor([[1,2]]);ids=[1,2]+[3]*32
    if mutation=='empty':ids=[]
    elif mutation=='truncated':ids.pop()
    elif mutation=='prefix':ids[0]=4
    elif mutation=='float':ids[-1]=3.0
    else:ids[-1]=10
    with pytest.raises(ValueError,match='grid'):generated_tokens(ids,prompt,10)


class Tokenizer:
    eos_token_id=0
    def __call__(self,text,**kwargs):return {'input_ids':torch.tensor([[ord(x) for x in text]])}
    def decode(self,tokens,**kwargs):return ''.join(chr(x) for x in tokens if x)


def test_answer_boundary_uses_separate_tokenization_and_no_truncation():
    task={'task':'Compute 2+2.','answer':'4'}
    prompt,ids=prepare_ids(Tokenizer(),task)
    assert prompt[0].tolist()==list(map(ord,PREFIX+task['task']+'\nAnswer:'))
    assert ids[0,-1].item()==ord('4') and ids.shape[1]==prompt.shape[1]+1
    with pytest.raises(ValueError,match='bound'):prepare_ids(Tokenizer(),{'task':'x'*256,'answer':'4'})


@pytest.mark.parametrize('text,expected',[(' 4\n',True),('4\nextra',False),('4\x00extra',True),('',False),('Four',False)])
def test_exact_answer_scoring_preserves_format_errors_and_stops_at_eos(text,expected):
    assert score_generation(Tokenizer(),list(map(ord,text)),'4')['exact_target_match'] is expected


@pytest.mark.parametrize('biased',[False,True])
def test_masked_projection_keeps_output_bias_and_restores_hooks(biased,tmp_path):
    torch.manual_seed(13);down=torch.nn.Linear(160,3,bias=biased);z=torch.randn(1,160)
    full=down(z)
    with masks([down],'static') as observers:
        actual=down(z);receipt=save_masks(tmp_path/'mask.npz',observers)
    expected=torch.nn.functional.linear(torch.cat((z[:,:144],torch.zeros_like(z[:,144:])),1),down.weight,down.bias)
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(down(z),full)
    (tmp_path/'traces').mkdir();(tmp_path/'mask.npz').rename(tmp_path/'traces'/'mask.npz')
    assert checked_masks(tmp_path,receipt,1,1,20,'static')==18


def test_hindsight_rank_uses_contribution_scores_and_preserves_stable_ties():
    down=torch.nn.Linear(160,2,bias=False)
    with torch.no_grad():down.weight.fill_(1)
    probe=Probe(down,'hindsight');z=torch.ones(1,160);z[:,:16]=.01
    result=probe(down,(z,))[0]
    assert result[:,:16].eq(0).all() and result[:,16:].eq(1).all()
    equal=Probe(down,'hindsight');equal(down,(torch.ones(1,160),))
    assert equal.masks[0].tolist()==[True]*18+[False]*2
    with pytest.raises(ValueError,match='single-token'):probe(down,(torch.ones(2,160),))


def test_partial_hook_installation_is_cleaned(monkeypatch):
    one=torch.nn.Linear(16,2);two=torch.nn.Linear(16,2)
    def fail(*args):raise RuntimeError('installation failed')
    monkeypatch.setattr(two,'register_forward_pre_hook',fail)
    with pytest.raises(RuntimeError,match='installation'):
        with masks([one,two],'static'):pass
    assert not one._forward_pre_hooks


def test_opt_bias_is_charged_on_selected_neurons_and_every_output_visit():
    full,selected=ffn_volume('opt',2,4,16,3,6)
    assert full==3*(2*16*9*4+2*4*4)
    assert selected==6*8*9*4+3*2*4*4
    assert ffn_volume('qwen',2,4,16,3,6)==(3*2*16*3*4*4,6*8*3*4*4)
    with pytest.raises(ValueError):ffn_volume('other',2,4,16,3,6)


def test_configuration_is_the_fixed_small_pair():
    cfg=json.loads((Path(__file__).parents[1]/'configs/balanced-development.json').read_text())
    validate_config(cfg);cfg['models']['opt']['revision']='changed'
    with pytest.raises(ValueError,match='pair'):validate_config(cfg)


def test_changed_static_masks_fail_even_with_updated_self_reported_digest(tmp_path):
    from dynamic_model_loading.experiment import digest
    (tmp_path/'traces').mkdir();path=tmp_path/'traces/mask.npz'
    mask=np.ones((2,1,20),dtype=bool);mask[:,:,:2]=False
    np.savez_compressed(path,bits=np.packbits(mask,axis=-1,bitorder='little'),shape=np.asarray(mask.shape))
    receipt={'file':'mask.npz','sha256':digest(path),'shape':[2,1,20],'selected_groups':36}
    with pytest.raises(ValueError,match='Static'):checked_masks(tmp_path,receipt,2,1,20,'static')

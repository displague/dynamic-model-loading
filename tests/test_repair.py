import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading import repair
from dynamic_model_loading.causal_study import incremental
from dynamic_model_loading.metrics import compare_logits
from dynamic_model_loading.repair_analysis import validate_repair_trace, safe_path
from dynamic_model_loading.repair_study import run_repair, validate_config, save_selector_control


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4,20,bias=False)
        self.up_proj = nn.Linear(4,20,bias=False)
        self.down_proj = nn.Linear(20,4,bias=False)
        self.act_fn = nn.functional.silu
    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


@pytest.mark.parametrize('schedule',repair.SCHEDULES)
@pytest.mark.parametrize('limit',[0,1,2,10])
def test_privileged_additions_match_independent_ranking_and_residency(schedule,limit):
    initial = torch.tensor([True,False,False,True,False])
    hot = torch.tensor([False,True,False,True,False])
    scores = torch.tensor([100.,2.,7.,100.,7.])
    actual = repair.additions(initial,hot,scores,schedule,limit)
    expected = repair.expected_additions(initial.numpy(),hot.numpy(),scores[~initial].numpy(),schedule,limit)
    assert np.array_equal(actual.numpy(),expected)
    assert not (actual & initial).any()
    if schedule=='resident_first':assert actual[1]
    if limit==1:assert actual[2] and not actual[4]  # Group-index tie break.
    if limit>=3:assert torch.equal(actual,~initial)


@pytest.mark.parametrize('scores',[[1.,float('nan')],[1.,-1.],[float('inf'),1.]])
def test_invalid_scores_cannot_create_a_repair_receipt(scores):
    with pytest.raises(ValueError,match='Finite'):
        repair.additions(torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),torch.tensor(scores),'hindsight',1)


@pytest.mark.parametrize('schedule',repair.SCHEDULES)
def test_additive_full_repair_equals_dense_and_initial_output_is_distinct(schedule):
    torch.manual_seed(91)
    mlp = TinyMLP(); x = torch.randn(1,1,4)
    initial = torch.tensor([[True]*7+[False]*3]); hot = torch.tensor([False]*8+[True,False])
    dense = mlp(x)
    probe = repair.RepairProbe(mlp,initial,hot,schedule,10,width=2)
    with repair.repair_probes([mlp],[probe]):
        corrected = mlp(x)
    torch.testing.assert_close(corrected,dense,rtol=1e-5,atol=1e-7)
    added,scores,audits = probe.take()
    assert np.array_equal(added,~initial.numpy())
    assert scores.shape==(1,3) and audits[0,1]<1e-5
    assert audits[0,0]>audits[0,1]
    assert not mlp._forward_hooks and not mlp._forward_pre_hooks and not mlp.down_proj._forward_pre_hooks


def test_repair_is_committed_inside_actual_incremental_qwen():
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=20,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).eval()
    ids = torch.tensor([[1,7,3,6,2]])
    mlps = extract_ffns(model)
    initial = np.zeros((5,2,10),dtype=bool); initial[:,:,:7] = True
    hot = np.zeros((2,10),dtype=bool); hot[:,-1] = True
    with torch.inference_mode():
        dense = incremental(model,ids)
        corrected,added,scores,audits = run_repair(model,mlps,ids,initial,hot,'resident_first',10,2)
        zero,_,_,_ = run_repair(model,mlps,ids,initial,hot,'hindsight',0,2)
    assert compare_logits(dense,corrected,ids)['logit_relative_l2']<1e-5
    assert compare_logits(dense,zero,ids)['logit_relative_l2']>1e-5
    assert np.array_equal(added,~initial) and scores.shape==(5,2,3) and audits.shape==(5,2,5)


def test_repair_cleans_hooks_and_retained_inputs_when_execution_raises():
    mlp = TinyMLP(); initial = torch.ones((1,10),dtype=torch.bool)
    observer = repair.RepairProbe(mlp,initial,torch.zeros(10,dtype=torch.bool),'hindsight',0,2)
    with pytest.raises(RuntimeError,match='interrupt'):
        with repair.repair_probes([mlp],[observer]):
            observer.before(mlp,(torch.ones(1,1,4),))
            raise RuntimeError('interrupt')
    assert observer.x is None and observer.z is None
    assert not mlp._forward_hooks and not mlp._forward_pre_hooks and not mlp.down_proj._forward_pre_hooks
    with pytest.raises(ValueError,match='Incomplete'):observer.take()


@pytest.fixture
def repair_receipt(tmp_path):
    initial = np.array([[[True,False,True,False,False]]],dtype=bool)
    hot = np.array([[False,True,False,False,False]],dtype=bool)
    scores = np.array([[[2.,7.,7.]]],dtype=np.float32)
    added = np.array([[[False,True,False,True,False]]],dtype=bool)
    audits = np.array([[[.4,.2,1.,.8,.3]]],dtype=np.float32)
    path = tmp_path/'trace.npz'
    repair.save_receipt(path,initial,added,scores,audits)
    return path,initial,hot,added


def test_valid_receipt_rebuilds_ranking_and_acquisition_counts(repair_receipt):
    path,initial,hot,added = repair_receipt
    actual,final,audits = validate_repair_trace(path,initial,hot,'resident_first',1)
    assert np.array_equal(actual,added) and np.array_equal(final,initial|added)
    metrics = repair.trace_metrics(initial,added,hot,np.full((1,5),100))
    assert metrics == {'added_groups':2,'resident_added_groups':1,'cold_added_groups':1,'added_cold_bytes':100,
                       'initial_selected_bytes':200,'added_selected_bytes':200,'final_selected_bytes':400}


@pytest.mark.parametrize('mutation',['initial','overlap','union','ranking','nan','negative','padding','shape'])
def test_corrupt_repair_receipts_fail_closed(repair_receipt,mutation):
    path,initial,hot,_ = repair_receipt
    with np.load(path,allow_pickle=False) as source:data={k:source[k].copy() for k in source.files}
    if mutation=='initial':data['initial_bits'][0,0,0]^=1
    elif mutation=='overlap':data['repair_bits'][0,0,0]|=1;data['final_bits'][0,0,0]|=1
    elif mutation=='union':data['final_bits'][0,0,0]^=2
    elif mutation=='ranking':data['omitted_scores'][0,0,2]=8
    elif mutation=='nan':data['omitted_scores'][0,0,0]=float('nan')
    elif mutation=='negative':data['audits'][0,0,0]=-1
    elif mutation=='padding':data['repair_bits'][0,0,0]|=128
    else:data['shape'][0]=2
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError):validate_repair_trace(path,initial,hot,'resident_first',1)


@pytest.mark.parametrize('path',['../escape','C:/outside','/outside'])
def test_artifact_paths_cannot_escape_archive(tmp_path,path):
    with pytest.raises(ValueError):safe_path(tmp_path,path)


def test_changed_repair_operating_point_is_rejected():
    with pytest.raises(ValueError,match='prospective'):validate_config({'ppl_limit':1.2})


def test_failed_selector_reproduction_preserves_actual_masks_and_logits(tmp_path):
    from safetensors.torch import load_file
    (tmp_path/'references').mkdir();(tmp_path/'traces').mkdir()
    expected=np.array([[[True,False]]]);actual=~expected
    logits=torch.randn(1,2,7)
    receipt=save_selector_control(tmp_path,'static',0,'doc',logits,actual,expected,100,100)
    assert receipt['passed'] is False and receipt['masks_equal'] is False
    torch.testing.assert_close(load_file(tmp_path/receipt['reference_file'])['logits'],logits,rtol=0,atol=0)
    with np.load(tmp_path/receipt['trace_file']) as data:
        unpacked=np.unpackbits(data['bits'],axis=-1,count=2,bitorder='little').astype(bool)
    assert np.array_equal(unpacked,actual)

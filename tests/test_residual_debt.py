import copy

import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.fault_generation import generate
from dynamic_model_loading.residual_debt import Packed2,ResidualDraft,choose_page


def unpack(packed):
    n,c = packed.codes.shape
    return packed.unpack_into(torch.empty(n,c*4),torch.empty(n,c,dtype=torch.uint8),torch.empty(n,c))


def test_two_bit_packing_order_constant_groups_and_exact_grid():
    w = torch.tensor([[0.,1.,2.,3.,-2.,-2.,-2.,-2.],[3.,2.,1.,0.,1.,2.,3.,4.]])
    q = Packed2.pack(w,4)
    assert q.codes[0,0].item()==228
    assert q.codes[1,0].item()==27
    assert q.scale[0,1].item()==0
    assert torch.equal(unpack(q),w)
    assert q.bytes==4+4*4+4*4


def test_quantization_error_and_workspace_reuse():
    torch.manual_seed(1)
    w = torch.randn(5,16)
    q = Packed2.pack(w,8)
    result = unpack(q)
    assert torch.all((result-w).reshape(5,2,8).abs()<=q.scale[...,None]/2+1e-6)
    scratch1 = torch.empty(5,4,dtype=torch.uint8)
    scratch2 = torch.empty(5,4)
    out = torch.empty_like(w)
    assert q.unpack_into(out,scratch1,scratch2).data_ptr()==out.data_ptr()
    assert torch.equal(out,result)


@pytest.mark.parametrize('group',[0,3,6,True,32])
def test_bad_quant_groups_are_rejected(group):
    with pytest.raises(ValueError,match='two-bit'):
        Packed2.pack(torch.ones(2,16),group)


def test_nonfinite_weights_and_wrong_scratch_rejected():
    with pytest.raises(ValueError,match='two-bit'):
        Packed2.pack(torch.full((2,8),float('nan')),4)
    q = Packed2.pack(torch.ones(2,8),4)
    with pytest.raises(ValueError,match='workspace'):
        q.unpack_into(torch.empty(2,8),torch.empty(2,2),torch.empty(2,2))


def test_error_cancellation_beats_largest_norm_and_can_stop():
    sketches = [[10.],[-9.],[2.]]
    assert choose_page(sketches,[3.],set())==(2,8.)
    assert choose_page(sketches,[1.],{2})[0] is None


def test_actual_correction_feedback_changes_the_second_acquisition():
    sketches = [[2.,0.],[1.,1.],[0.,2.]]
    assert choose_page(sketches,[3.,3.],set())[0]==1
    assert choose_page(sketches,[2.,2.],{1})[0]==0  # Hypothetical predicted update.
    assert choose_page(sketches,[-1.,3.],{1})[0]==2  # Actual correction was [4,0].
    assert choose_page([[1.],[1.]],[2.],set())[0]==0
    assert choose_page([[1.]],[1.],{0})[0] is None
    with pytest.raises(ValueError,match='Nonfinite'):
        choose_page([[float('inf')]],[1.],set())


def tiny_model():
    torch.manual_seed(312)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=128,eos_token_id=30)).eval()


def wrap(model,**kwargs):
    centroids = [torch.eye(16)[:2] for _ in range(2)]
    return ResidualDraft(extract_ffns(model),centroids,page_width=8,group=4,rank=2,
                         max_pages=2,device='cpu',**kwargs)


def test_complete_correction_reconstructs_all_neurons_without_weight_mutation():
    model = tiny_model()
    x = torch.randn(1,2,16)
    mlps = extract_ffns(model)
    expected = [mlp(x).detach() for mlp in mlps]
    originals = [mlp.gate_proj.weight.clone() for mlp in mlps]
    decisions, pages = [],[]
    pager = wrap(model,page_sink=pages.append,policy_sink=decisions.append)
    try:
        pager.mode='complete'
        for mlp,ref in zip(mlps,expected,strict=True):
            assert torch.allclose(mlp(x),ref,atol=1e-7,rtol=1e-5)
        assert sum(e['outcome']=='load' for e in pages)==16
        assert sum(e['outcome']=='release' for e in pages)==16
        assert all([r['page'] for r in e['choices']]==list(range(4)) for e in decisions)
        assert pager.cache.used_bytes==0 and len(pager.layers)==2
        assert pager.resident_bytes>0 and pager.workspace_bytes>0
        for mlp,catalog,old in zip(mlps,pager.catalogs,originals,strict=True):
            assert torch.equal(mlp.gate_proj.weight,old)
            assert mlp.gate_proj.weight.untyped_storage().data_ptr()==catalog._gate.untyped_storage().data_ptr()
    finally:
        pager.restore()
    assert all(torch.equal(mlp(x),ref) for mlp,ref in zip(mlps,expected,strict=True))


@pytest.mark.parametrize('mode',['dense_stream','base','fixed','debt'])
def test_actual_tiny_transformer_commits_only_dense_reference(mode):
    target = tiny_model()
    draft = copy.deepcopy(target)
    rows,pages = [],[]
    pager = wrap(draft,policy_sink=rows.append,page_sink=pages.append)
    pager.mode = mode
    prefix = torch.tensor([[1,3]])
    try:
        ref = generate(target,prefix,(30,),cap=7)
        alt = generate(target,prefix,(30,),cap=7,draft=draft,pager=pager)
        assert alt['ids']==ref['ids'] and alt['stop_reason']==ref['stop_reason']
        assert alt['attempted']%4==0 and alt['accepted']<=alt['attempted']
        assert all(len(r['choices'])<=({'dense_stream':4,'base':0,'fixed':2,'debt':2}[mode]) for r in rows)
        if mode=='base':
            assert not pages and all(not r['sketches'] for r in rows)
        assert pager.cache.used_bytes==0
    finally:
        pager.restore()


def test_invalid_build_restores_original_forward():
    model = tiny_model()
    module = model.model.layers[0].mlp
    original = module.forward
    with pytest.raises(ValueError):
        ResidualDraft(extract_ffns(model),[torch.zeros(2,16)]*2,page_width=8,group=3,rank=2,
                      max_pages=2,device='cpu')
    assert module.forward==original


def test_forward_feedback_uses_actual_projected_correction(monkeypatch):
    import dynamic_model_loading.residual_debt as module
    model=tiny_model()
    rows,calls=[],[]
    pager=wrap(model,policy_sink=rows.append)
    def observe_selector(sketches,debt,used):
        calls.append(list(debt))
        return min(set(range(len(sketches)))-used),1.0
    monkeypatch.setattr(module,'choose_page',observe_selector)
    pager.mode='debt'
    try:
        extract_ffns(model)[0](torch.randn(1,1,16))
        row=rows[0]
        expected=[a-b for a,b in zip(calls[0],row['choices'][0]['observed'],strict=True)]
        predicted=[a-b for a,b in zip(calls[0],row['sketches'][0],strict=True)]
        assert calls[1]==expected and calls[1]!=predicted
    finally:
        pager.restore()


@pytest.mark.parametrize('position',[0,1,2,3])
def test_forced_partial_acceptance_rolls_back_actual_residual_draft(position):
    target=tiny_model()
    draft=copy.deepcopy(target)
    pager=wrap(draft)
    pager.mode='debt'
    original=draft.forward
    calls=0
    def changed(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs)
        calls+=1
        if calls==2+position:
            wrong=(int(result.logits[0,-1].argmax())+1)%31
            result.logits=result.logits.clone()
            result.logits[0,-1,wrong]=1e4
        return result
    draft.forward=changed
    rows=[]
    prefix=torch.tensor([[1,3]])
    try:
        reference=generate(target,prefix,(30,),cap=7)
        result=generate(target,prefix,(30,),cap=7,draft=draft,pager=pager,record=rows.append)
        assert result['ids']==reference['ids'] and result['stop_reason']==reference['stop_reason']
        assert rows[0]['accepted']==position and rows[0]['fallback'] is not None
        assert rows[0]['target_cache']==rows[0]['draft_cache']==2+position+1
    finally:
        pager.restore()

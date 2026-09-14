import copy

import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.fault_generation import generate
from dynamic_model_loading.progressive_precision import EmbeddedWeight,IncrementCache,ProgressiveDraft


def decode(q,stage):
    n,c=q.base.shape
    out=torch.empty(n,c*2)
    b=torch.empty(n,c,dtype=torch.uint8)
    f=torch.empty(n,c)
    delta=torch.empty_like(out)
    q.base_into(out,b,f)
    for s in range(stage):
        q.add_into(out,s,q.increments[s],delta,b,f)
    return out


def test_embedded_levels_match_independent_grid_and_constant_groups():
    torch.manual_seed(15)
    w=torch.randn(3,16)
    w[0,:8]=3
    q=EmbeddedWeight.encode(w,8)
    group=w.reshape(3,2,8)
    lo=group.amin(-1,keepdim=True)
    scale=(group.amax(-1,keepdim=True)-lo)/255
    codes=((group-lo)/scale.clamp_min(1e-30)).round().clamp(0,255)
    for stage,step in enumerate((16,4,1)):
        expected=(codes.div(step,rounding_mode='floor')*step+(step-1)/2)*scale+lo
        assert torch.allclose(decode(q,stage),expected.reshape_as(w),rtol=1e-6,atol=1e-6)
        assert torch.all((decode(q,stage)-w).reshape_as(group).abs()<=scale*(step/2)+1e-6)
        assert torch.equal(decode(q,stage)[0,:8],w[0,:8])
    assert q.base.numel()==w.numel()/2
    assert all(p.numel()==w.numel()/4 for p in q.increments)


@pytest.mark.parametrize('group',[0,3,6,True,32])
def test_invalid_quantizer(group):
    with pytest.raises(ValueError):
        EmbeddedWeight.encode(torch.ones(2,16),group)


def test_cache_cold_bypass_admission_hit_and_eviction():
    rows=[]
    cache=IncrementCache((3,2,4),1,'cpu',rows.append)
    host=torch.arange(24,dtype=torch.uint8).reshape(3,2,4)
    cache.begin_token(True)
    assert torch.equal(cache.get((0,0),host),host)
    assert not cache.entries
    cache.observe((0,0),2.)
    cache.begin_token(True)
    cache.get((0,0),host)
    cache.get((1,0),host)
    cache.begin_token(True)
    assert torch.equal(cache.get((0,0),host),host)
    cache.observe((1,0),10.)
    cache.begin_token(True)
    assert not cache.entries
    cache.get((1,0),host)
    assert cache.stats==dict(h2d_bytes=0,hits=1,loads=4,evictions=1)
    assert [r['kind'] for r in rows]==['load','load','load','hit','evict','load']
    assert cache.pool.numel()==48 and cache.staging.numel()==24


def tiny_model():
    torch.manual_seed(312)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=128,eos_token_id=30)).eval()


@pytest.mark.parametrize('mode',ProgressiveDraft.MODES)
def test_tiny_speculation_commits_dense_ids(mode):
    target=tiny_model()
    draft=copy.deepcopy(target)
    rows=[]
    pager=ProgressiveDraft(extract_ffns(draft),device='cpu',group=4,slots=1,policy_sink=rows.append)
    pager.mode=mode
    try:
        prefix=torch.tensor([[1,3]])
        ref=generate(target,prefix,(30,),cap=7)
        result=generate(target,prefix,(30,),cap=7,draft=draft,pager=pager)
        assert ref['ids']==result['ids'] and ref['stop_reason']==result['stop_reason']
        assert pager.cache.token>2
        assert all(r['stages']=={'q4':0,'q6':1,'q8':2}.get(mode,1+(r['values'][0]/4>.02))
                   for r in rows) if mode!='q4' else all(r['stages']==0 for r in rows)
    finally:
        pager.restore()


def test_retaining_rejected_work_can_only_change_placement_not_outputs():
    target=tiny_model()
    draft=copy.deepcopy(target)
    pager=ProgressiveDraft(extract_ffns(draft),device='cpu',group=4,slots=1,threshold=0)
    original=draft.forward
    calls=0
    def force_rejection(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs)
        calls+=1
        if calls==2:
            wrong=(int(result.logits[0,-1].argmax())+1)%31
            result.logits=result.logits.clone()
            result.logits[0,-1,wrong]=1e4
        return result
    draft.forward=force_rejection
    outputs=[]
    ledgers=[]
    rounds=[]
    for mode in ('adaptive','retained','static'):
        calls=0
        rows=[]
        rr=[]
        pager.policy_sink=rows.append
        pager.reset()
        pager.mode=mode
        outputs.append(generate(target,torch.tensor([[1,3]]),(30,),cap=11,draft=draft,pager=pager,record=rr.append))
        rounds.append(rr)
        ledgers.append([{k:v for k,v in r.items() if k!='mode'} for r in rows])
    assert outputs[0]['ids']==outputs[1]['ids']
    assert outputs[0]['accepted']==outputs[1]['accepted']
    assert outputs[2]['ids']==outputs[0]['ids'] and ledgers[2]==ledgers[0]
    assert rounds[0]==rounds[1] and rounds[0][0]['accepted']==0
    assert rounds[0][0]['target_cache']==rounds[0][0]['draft_cache']==3
    assert ledgers[0]==ledgers[1]
    assert pager.cache.stats['hits']>0
    pager.restore()

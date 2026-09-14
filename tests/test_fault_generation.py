import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.fault_generation import generate
from dynamic_model_loading.fault_pager import PageCache, PageCatalog, PageKey, PagedDraft, SideIndex


def tiny_model():
    torch.manual_seed(1729)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,max_position_embeddings=128,eos_token_id=30)).eval()


@pytest.mark.parametrize('selected',[3,4])
@pytest.mark.parametrize('mode',['eager','lru','prefetch'])
def test_kv_rollback_commits_dense_reference_with_real_tiny_transformer(selected,mode):
    target=tiny_model()
    draft=copy.deepcopy(target)
    indexes=[SideIndex(torch.zeros(1,16),torch.tensor([[3,1,2,0]])) for _ in range(2)]
    pager=PagedDraft(extract_ffns(draft),indexes,page_width=8,selected_pages=selected,
                     capacity_bytes=3*8*16*4*2,device='cpu',mode=mode)
    prefix=torch.tensor([[1,2,3]])
    rows=[]
    reference=generate(target,prefix,(99,),cap=9)
    result=generate(target,prefix,(99,),cap=9,draft=draft,pager=pager,record=rows.append)
    assert result['ids']==reference['ids']
    assert result['attempted']%4==0 and len(result['ids'])==9
    assert [v for r in rows for v in r['committed']]==reference['ids']
    assert all(r['target_cache']==r['draft_cache'] for r in rows)
    if mode=='eager':
        assert pager.cache.used_bytes==0 and not pager.cache._entries
    assert pager.cache.stats['peak_payload_bytes']<=pager.cache.capacity_bytes
    pager.restore()


def test_medoid_duplicate_vectors_still_select_distinct_source_positions():
    index=SideIndex.build(torch.zeros(4,2),torch.ones(4,3),3)
    assert index.source_positions==[0,1,2]


def test_forced_rejections_crop_both_caches_and_consume_fallback_context():
    target=tiny_model()
    draft=copy.deepcopy(target)
    with torch.no_grad():
        draft.lm_head.weight.neg_()
    indexes=[SideIndex(torch.zeros(1,16),torch.tensor([[3,1,2,0]])) for _ in range(2)]
    pager=PagedDraft(extract_ffns(draft),indexes,page_width=8,selected_pages=3,
                     capacity_bytes=4096,device='cpu',mode='prefetch')
    prefix=torch.tensor([[1,2,3]])
    expected=generate(target,prefix,(99,),cap=9)
    rows=[]
    actual=generate(target,prefix,(99,),cap=9,draft=draft,pager=pager,record=rows.append)
    assert actual['ids']==expected['ids']
    assert any(r['fallback'] is not None for r in rows)
    assert all(r['target_cache']==r['draft_cache'] for r in rows)
    assert [r['base'] for r in rows]==[3+sum(len(p['committed']) for p in rows[:i]) for i in range(len(rows))]


def test_page_cache_does_not_alias_cpu_source_or_mutate_live_payloads():
    model=tiny_model()
    catalog=PageCatalog(extract_ffns(model)[0],0,8)
    cache=PageCache(4096,'cpu')
    value=cache.get(PageKey(0,0),catalog.payload(0),request='demand')
    original=value.gate.clone()
    cache.get(PageKey(0,1),catalog.payload(1),request='demand')
    torch.testing.assert_close(value.gate,original,rtol=0,atol=0)
    assert value.gate.untyped_storage().data_ptr()!=catalog._gate.untyped_storage().data_ptr()


@pytest.mark.parametrize('rejected',[False,True])
@pytest.mark.parametrize('cap',[1,9])
def test_accepted_and_fallback_eos_pay_all_four_proposals(rejected,cap):
    target=tiny_model()
    prefix=torch.tensor([[1,2,3]])
    eos=generate(target,prefix,(99,),cap=1)['ids'][0]
    draft=copy.deepcopy(target)
    if rejected:
        with torch.no_grad():
            draft.lm_head.weight.neg_()
    indexes=[SideIndex(torch.zeros(1,16),torch.tensor([[3,1,2,0]])) for _ in range(2)]
    pager=PagedDraft(extract_ffns(draft),indexes,page_width=8,selected_pages=4,
                     capacity_bytes=4096,device='cpu',mode='prefetch')
    rows=[]
    result=generate(target,prefix,(eos,),cap=cap,draft=draft,pager=pager,record=rows.append)
    assert result['ids']==[eos] and result['stop_reason']=='eos'
    assert result['attempted']==4 and result['accepted']==(0 if rejected else 1)
    assert len(rows)==1 and len(rows[0]['proposed'])==4
    assert rows[0]['emitted_accepted']==(0 if rejected else 1)
    assert rows[0]['target_cache']==rows[0]['draft_cache']==(3 if rejected else 7)
    assert rows[0]['fallback']==(eos if rejected else None)
    if rejected:
        assert not pager.pending

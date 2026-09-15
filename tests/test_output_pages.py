import copy

import pytest
import torch
from transformers.cache_utils import DynamicCache

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.decision_field import fingerprint
from dynamic_model_loading.fault_generation import step
from dynamic_model_loading.output_pages import OutputPageDraft,split_four_base,projected_margin
from dynamic_model_loading.progressive_precision import ProgressiveDraft,pack_codes
from test_progressive_precision import tiny_model


def unpack(bits):
    return torch.stack([(bits>>(2*i))&3 for i in range(4)],-1).reshape(bits.shape[0],-1)


def test_two_bit_extension_preserves_integer_grid():
    codes = torch.arange(256,dtype=torch.uint8).reshape(4,64)
    upper = codes>>4
    low,next_ = split_four_base(pack_codes(upper,4))
    assert torch.equal((unpack(low)<<2)+unpack(next_),upper)


def test_projection_keeps_pair_order_but_not_third_token_certificate():
    h = torch.tensor([1.,0.,1.])
    delta = torch.tensor([0.,0.,10.])
    gamma = torch.tensor([2.,3.,4.])
    head = torch.eye(3)
    basis = (head[0]-head[1])*gamma
    corrected = h+delta
    logits = head@(gamma*corrected/torch.sqrt(corrected.square().mean()+1e-6))
    assert projected_margin(basis,h,delta,1e-6)==pytest.approx(float(logits[0]-logits[1]),abs=1e-6)
    assert logits[0]>logits[1] and int(logits.argmax())==2


def test_local_page_sum_matches_full_grid_and_kv_replay():
    model = tiny_model()
    other = copy.deepcopy(model)
    parent = ProgressiveDraft(extract_ffns(other),device='cpu',group=4,slots=1)
    parent.mode='q8'
    pager = OutputPageDraft(extract_ffns(model),device='cpu',group=4,slots=1,page_width=8)
    x = torch.randn(1,1,16)
    pager.prefill=True
    pager.begin_token(); parent.begin_token()
    old = extract_ffns(other)[-1](x)
    new = extract_ffns(model)[-1](x)
    assert torch.allclose(new,old,rtol=1e-4,atol=1e-6)
    capture={}
    handle=model.model.norm.register_forward_pre_hook(lambda m,a:capture.update(h=a[0].clone()))
    cache = DynamicCache(config=model.config)
    for token in (1,3):
        step(model,torch.tensor([[token]]),cache,pager)
    pager.prefill=False
    ids=torch.tensor([[5]])
    base=step(model,ids,cache,pager)
    original_h=capture['h'].clone()
    original_y=pager.base_y.clone()
    before,_=fingerprint(cache,3)
    events=[]
    pager.page_cache.sink=events.append
    deltas=[pager.acquire(i) for i in range(4)]
    full_y=pager._compute()
    assert torch.allclose(original_y+sum(deltas),full_y,rtol=1e-4,atol=1e-6)
    predicted=model.lm_head(model.model.norm.forward(original_h+(full_y-original_y)))
    assert len(events)==12 and all(r['kind']=='load' for r in events)
    with pytest.raises(ValueError):
        pager.acquire(0)
    partial=pager.partial([0,2])
    assert torch.allclose(partial,original_y+deltas[0]+deltas[2],rtol=1e-4,atol=1e-6)
    assert fingerprint(cache,3)[0]==before
    cache.crop(2)
    pager.force_full=True
    actual=step(model,ids,cache,pager)
    assert torch.allclose(actual,predicted,rtol=1e-4,atol=1e-6)
    assert fingerprint(cache,3)[0]==before
    assert base.shape==actual.shape
    handle.remove(); pager.restore(); parent.restore()


def test_invalid_page_geometry_does_not_mutate_model():
    model=tiny_model()
    original=model.model.layers[0].mlp.forward
    with pytest.raises(ValueError):
        OutputPageDraft(extract_ffns(model),device='cpu',group=4,slots=1,page_width=7)
    assert model.model.layers[0].mlp.forward==original

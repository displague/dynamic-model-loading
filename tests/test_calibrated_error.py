import copy
import json
import math

import numpy as np
import pytest
import torch

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.bayes_analysis import audit_kv,audit_policy,fit_replay,decision,statistical_metrics
from dynamic_model_loading.bayes_screen import CONFIG,FROZEN,validate_config
from dynamic_model_loading.calibrated_error import ErrorGP,CalibratedDraft,block_quantile
from dynamic_model_loading.fault_generation import generate
from test_progressive_precision import tiny_model


def test_frozen_protocol():
    validate_config(json.loads(CONFIG.read_text(encoding='utf-8')))
    for k,v in [('threshold',.02),('fit_documents',4),('calibration_documents',8),('prefix_tokens',4)]:
        with pytest.raises(ValueError):
            validate_config(dict(FROZEN,**{k:v}))


def test_document_quantile_does_not_pretend_small_sample_coverage():
    assert block_quantile(list(range(9)))==8
    assert math.isinf(block_quantile(list(range(8))))
    assert block_quantile([-3.]*9)==0
    with pytest.raises(ValueError):
        block_quantile([float('nan')])


def test_gp_uses_covariance_and_uncertainty_and_independent_replay():
    generator=torch.Generator().manual_seed(43)
    x=torch.randn(32,18,generator=generator,dtype=torch.float64)
    z=-3+.4*torch.sin(x[:,0])+.2*x[:,1]
    gp=ErrorGP(x,z)
    blocks=[]
    for _ in range(9):
        bx=torch.randn(4,18,generator=generator,dtype=torch.float64)
        blocks.append((bx,-3+.4*torch.sin(bx[:,0])+.2*bx[:,1]))
    gp.calibrate(blocks)
    reconstructed,predict=fit_replay(x.numpy(),z.numpy(),[(a.numpy(),b.numpy()) for a,b in blocks])
    for k,v in gp.tensors().items():
        np.testing.assert_allclose(v.numpy(),reconstructed[k],rtol=1e-7,atol=1e-8)
    for point in [x[0],x[1],torch.ones(18)*100]:
        a,b=gp.bounds(point),predict(point.numpy())
        for k in a:
            assert a[k]==pytest.approx(b[k],rel=1e-7,abs=1e-8)
        assert a['upper']>=a['mean']
    assert gp.predict(torch.ones(18)*100)[1]>gp.predict(x[0])[1]
    assert abs(gp.predict(x[0])[0]-float(z[0]))<abs(float(gp.zmean-z[0]))
    rows=[('document',dict(layer=0,feature=x[i].tolist(),observed_log=float(z[i]))) for i in range(4)]
    summary=statistical_metrics(rows,[predict],[float(gp.zmean)])
    json.dumps(summary,allow_nan=False)
    with pytest.raises(ValueError):
        gp.predict([float('nan')]*18)


@pytest.mark.parametrize('rejected',[0,1,2,3])
def test_explicit_prefill_and_forced_rollback_preserve_actual_prefix_kv(rejected):
    target=tiny_model()
    draft=copy.deepcopy(target)
    prefix=torch.tensor([[1,3]])
    ref=generate(target,prefix,(99,),cap=9)
    pager=CalibratedDraft(extract_ffns(draft),device='cpu',group=4,slots=1,threshold=.06)
    pager.mode='p8d6'; pager.prefill_tokens=2
    policy,kv,rr=[],[],[]
    pager.policy_sink=policy.append; pager.kv_sink=kv.append
    original=draft.forward
    calls=0
    def force(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs)
        calls+=1
        j=calls-2
        if 0<=j<=rejected:
            token=ref['ids'][j] if j<rejected else (ref['ids'][j]+1)%31
            result.logits=result.logits.clone()
            result.logits[0,-1].fill_(-1e4)
            result.logits[0,-1,token]=1e4
        return result
    draft.forward=force
    try:
        actual=generate(target,prefix,(99,),cap=9,draft=draft,pager=pager,record=rr.append)
        assert actual['ids']==ref['ids'] and rr[0]['accepted']==rejected
        assert rr[0]['target_cache']==rr[0]['draft_cache']==2+rejected+1
        assert all(r['stages']==(2 if r['token']<=2 else 1) for r in policy)
        assert len({r['sha256'] for r in kv})==1
        # Replay CPU fingerprints with zero D2H cost but actual tensor hashing.
        audit_kv(kv,2,pager.cache.token,rr,bytes_per_token=0)
        changed=copy.deepcopy(kv); changed[-1]['sha256']='a'*64
        with pytest.raises(ValueError):
            audit_kv(changed,2,pager.cache.token,rr,bytes_per_token=0)
    finally:
        pager.restore()


def test_shadow_counterfactual_is_label_only_not_returned_output():
    model=tiny_model()
    adapter=extract_ffns(model)[0]
    pager=CalibratedDraft(extract_ffns(model),device='cpu',group=4,slots=1)
    pager.prefill_tokens=1
    rows=[]
    pager.policy_sink=rows.append
    x=torch.ones(1,1,16)
    outputs={}
    try:
        for mode in ('q6','p8d8','shadow'):
            pager.reset(); pager.mode=mode
            pager.begin_token(); pager.begin_token()
            outputs[mode]=adapter(x)
        torch.testing.assert_close(outputs['shadow'],outputs['q6'],rtol=0,atol=0)
        assert not torch.equal(outputs['shadow'],outputs['p8d8'])
        assert rows[-1]['observed_log'] is not None and len(rows[-1]['feature'])==18
    finally:
        pager.restore()


@pytest.mark.parametrize('mode', ['p8mean','p8gp','p8constant'])
def test_adaptive_modes_really_omit_unrequested_payloads(mode):
    target=tiny_model()
    draft=copy.deepcopy(target)
    pager=CalibratedDraft(extract_ffns(draft),device='cpu',group=4,slots=1,threshold=.06)
    pager.mode=mode; pager.prefill_tokens=2
    # A real fitted GP with constant, tiny errors: every calibrated rule skips.
    x=torch.arange(32*18,dtype=torch.float64).reshape(32,18)/100
    models=[ErrorGP(x,torch.full((32,),-8.,dtype=torch.float64)) for _ in range(2)]
    for m in models:
        m.calibrate([(x[:4],torch.full((4,),-8.)) for _ in range(9)])
    pager.models=models
    rows=[]; pages=[]
    pager.policy_sink=rows.append; pager.cache.sink=pages.append
    before=[{k:v.clone() for k,v in m.tensors().items()} for m in models]
    try:
        reference=generate(target,torch.tensor([[1,3]]),(99,),cap=5)
        actual=generate(target,torch.tensor([[1,3]]),(99,),cap=5,draft=draft,pager=pager)
        assert reference['ids']==actual['ids']
        assert all(r['stages']==1 and r['observed_log'] is None for r in rows if r['phase']=='decode')
        assert all(r['key'][1]==0 for r in pages if r['token']>2)
        for m,b in zip(models,before,strict=True):
            assert all(torch.equal(v,b[k]) for k,v in m.tensors().items())
    finally:
        pager.restore()


def test_physical_and_phase_receipts_fail_closed():
    policy=[dict(token=t,layer=0,mode='p8d6',phase='prefill' if t==1 else 'decode',
        stages=2 if t==1 else 1,feature=None,estimate=None,observed_log=None,d2h_bytes=0) for t in (1,2)]
    pages=[dict(kind='load',token=1,key=[0,0],slot=0,bytes=12),
        dict(kind='load',token=1,key=[0,1],slot=1,bytes=12),dict(kind='hit',token=2,key=[0,0],slot=0,bytes=0)]
    assert audit_policy(policy,pages,'p8d6',2,1,layers=1,slots=1,slab_bytes=12)['cache']['h2d_bytes']==24
    for key,value in [('phase','prefill'),('stages',2),('observed_log',-3),('d2h_bytes',4)]:
        altered=copy.deepcopy(policy); altered[-1][key]=value
        with pytest.raises(ValueError):
            audit_policy(altered,pages,'p8d6',2,1,layers=1,slots=1,slab_bytes=12)
    with pytest.raises(ValueError):
        audit_policy(policy,pages[:-1],'p8d6',2,1,layers=1,slots=1,slab_bytes=12)


def test_prefill_success_does_not_become_bayesian_success():
    c={m:dict(acceptance=.5,h2d_per_committed=100) for m in FROZEN['conditions']}
    c['q6']['acceptance']=.2; c['p8d6']['h2d_per_committed']=80
    s=dict(constant_mean_mae=1,predictions={k:dict(mae=1,block_coverage=.8,false_safe=1,safe=1) for k in ('mean','upper')})
    verdict=decision(c,s)
    assert verdict['hypotheses']==dict(Hcal=False,Hunc=False,Hprefill=True,Hpolicy=False)
    assert verdict['decision']=='advance_to_new_protocol' and not verdict['full_suite_launched']

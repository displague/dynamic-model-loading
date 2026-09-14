import copy
import json

import pytest

from dynamic_model_loading.refinement_analysis import audit_policy,decision
from dynamic_model_loading.refinement_screen import CONFIG,FROZEN,validate_config


def test_frozen_subset():
    validate_config(json.loads(CONFIG.read_text(encoding='utf-8')))
    for k,v in [('threshold',.1),('cache_slots',16),('extra_cuda_mib',2048),('prefix_tokens',32)]:
        with pytest.raises(ValueError):
            validate_config(dict(FROZEN,**{k:v}))


def example():
    policy=[dict(token=t,layer=l,mode='retained',stages=1,values=[.04],d2h_bytes=4)
            for t in (1,2,3) for l in (0,1)]
    pages=[dict(kind='load',token=1,key=[l,0],slot=1,bytes=12) for l in (0,1)]
    pages +=[dict(kind='load',token=2,key=[l,0],slot=l,bytes=12) for l in (0,1)]
    pages +=[dict(kind='hit',token=3,key=[0,0],slot=0,bytes=0),dict(kind='load',token=3,key=[1,0],slot=1,bytes=12)]
    return policy,pages


def test_independent_replay_and_tampering():
    policy,pages=example()
    report=audit_policy(policy,pages,'retained',3,layers=2,slots=1,slab_bytes=12)
    assert report['cache']==dict(h2d_bytes=60,hits=1,loads=5,evictions=0)
    for target,field,value in [('policy','stages',2),('policy','values',[.2]),('pages','slot',0),('pages','bytes',0)]:
        p,e=copy.deepcopy(policy),copy.deepcopy(pages)
        (p if target=='policy' else e)[0][field]=value
        with pytest.raises(ValueError):
            audit_policy(p,e,'retained',3,layers=2,slots=1,slab_bytes=12)
    with pytest.raises(ValueError):
        audit_policy(policy,pages[:-1],'retained',3,layers=2,slots=1,slab_bytes=12)


def test_gates_keep_speed_separate_and_do_not_erase_representation_failure():
    c={m:dict(acceptance=.5,h2d_per_consumed=100,h2d_bytes=1000,wall_seconds=10) for m in FROZEN['conditions']}
    c['q4']['acceptance']=.2
    c['retained']['h2d_bytes']=800
    report=decision(c)
    assert report['hypotheses']==dict(H1_representation=True,H2_acquisition=False,H3_persistence=True)
    assert report['decision']=='advance_to_new_protocol' and not report['retained_5pct_faster']
    c['q6']['acceptance']=.1
    assert decision(c)['decision']=='stop'


def test_richardson_transfers_when_error_expansion_holds_and_fails_when_it_does_not():
    import torch
    from dynamic_model_loading.refinement_analysis import extrapolation_diagnostic
    data={f'{m}.0':torch.tensor([[[v]]]) for m,v in [('fp32',10.),('q4',14.),('q6',11.),('q8',10.25)]}
    report=extrapolation_diagnostic(data,1)
    assert report['passed'] and report['mean_richardson_error']==0
    assert report['rows'][0]['increment_ratio']==.25
    assert report['rows'][0]['increment_cosine']==1
    data['q6.0']=torch.tensor([[[9.]]])
    assert not extrapolation_diagnostic(data,1)['passed']

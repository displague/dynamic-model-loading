import numpy as np
import pytest
import torch
from dynamic_model_loading.vocabulary_risk import common_normals, pick_action, acquire, audit_trace


def test_common_draws_are_fixed_and_antithetic():
    a=common_normals(913,5)
    assert a.shape==(2,16,5) and a.dtype==np.float32
    assert np.array_equal(a,common_normals(913,5)) and np.array_equal(a[:,:8],-a[:,8:])


def test_full_vocabulary_score_sees_third_rival_and_ties_use_page_order():
    pair=np.zeros((2,16),dtype=np.int64); third=pair.copy(); third[1,:5]=2
    rows=[dict(page=0,ids=third.tolist(),mismatches=5),dict(page=1,ids=pair.tolist(),mismatches=0)]
    assert pick_action(rows)==1
    rows[0].update(ids=pair.tolist(),mismatches=0)
    assert pick_action(rows)==0


def test_sample_score_tampering_rejected():
    with pytest.raises(ValueError): pick_action([dict(page=0,ids=np.zeros((2,16),int).tolist(),mismatches=1)])


def test_controller_only_observes_selected_pages_and_scores_entire_head():
    index=dict(vector_prior=np.zeros((3,4)),vector_covariance=np.eye(3),feature_variance=np.ones(4)*.01)
    calls=[]; scored=[]
    weight=torch.tensor([[1.,0,0,0],[0,1,0,0],[0,0,2,0],[0,0,0,1.]])
    def head(h): scored.append(h.shape); return h@weight.T
    def observe(page): calls.append(page); return torch.full((4,),.01*(page+1))
    result=acquire(index,torch.tensor([1.,.1,.2,.3]),torch.ones(4),head,observe,seed=77,budget=2)
    assert calls==result['pages'] and len(set(calls))==2 and len(scored)==3+2
    assert all(shape==(32,4) for shape in scored)
    assert result['h2d_bytes']==result['d2h_bytes']==0
    for step in result['steps']: assert step['page']==pick_action(step['actions'])


def test_positive_common_rms_scale_does_not_change_full_argmax():
    h=torch.tensor([[1.,-2.,3.],[2.,1.,-.5]])
    scale=torch.sqrt(h.square().mean(-1,keepdim=True)+1e-6)
    norm=torch.tensor([.5,1.,2.]); weight=torch.tensor([[1.,0,1.],[0,2,0],[-1.,1.,3.]])
    assert torch.equal(((h*norm)@weight.T).argmax(-1),((h/scale*norm)@weight.T).argmax(-1))


def test_trace_replay_and_paid_vector_tamper():
    import copy
    index=dict(vector_prior=np.zeros((3,4)),vector_covariance=np.eye(3),feature_variance=np.ones(4)*.01)
    observe=lambda page:torch.full((4,),.01*(page+1))
    raw=acquire(index,torch.tensor([1.,.1,.2,.3]),torch.ones(4),lambda h:h,observe,seed=20260919,budget=2)
    scalar=[float(observe(p).sum()) for p in raw['pages']]
    assert audit_trace(raw,index,np.ones(4),scalar,1,width=4,pages=3,budget=2,cuda=False)==(0,0)
    changed=copy.deepcopy(raw); changed['steps'][0]['observed'][0]+=1
    with pytest.raises(ValueError): audit_trace(changed,index,np.ones(4),scalar,1,width=4,pages=3,budget=2,cuda=False)


def test_new_gate_requires_real_prefix_savings_against_both_controls():
    from dynamic_model_loading.vocabulary_screen import screen_decision
    base=dict(accepted=8,attempted=8,acceptance=1.,charged_h2d_bytes=100,charged_wall_seconds=1.)
    rows={name:dict(base) for name in ('risk','fullrisk','all35')}
    assert not screen_decision(rows,.1)['gates']['Hacquisition']
    rows['fullrisk']['charged_h2d_bytes']=99
    assert all(screen_decision(rows,.1)['gates'].values())
    rows['all35']['charged_h2d_bytes']=99
    assert not screen_decision(rows,.1)['gates']['Hacquisition']


def test_new_config_freezes_score_and_new_workload():
    import json
    from dynamic_model_loading.vocabulary_screen import CONFIG,validate_config
    cfg=json.loads(CONFIG.read_text()); validate_config(cfg)
    cfg['monte_carlo_draws']=8
    with pytest.raises(ValueError): validate_config(cfg)

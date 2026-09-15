import copy,hashlib,json
import numpy as np
import pytest
import torch
from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.output_pages import OutputPageDraft
from dynamic_model_loading.fault_generation import generate
from dynamic_model_loading.risk_runtime import RiskReadout,VocabularyReadout,KVJournal
from dynamic_model_loading.risk_screen import CONFIG,FROZEN,validate_config
from dynamic_model_loading.risk_analysis import screen_decision
from test_progressive_precision import tiny_model


def test_config_matches_frozen_protocol():
    validate_config(json.loads(CONFIG.read_text()))
    changed=dict(FROZEN,page_budget=18)
    with pytest.raises(ValueError): validate_config(changed)


def test_journal_rejects_stale_rejected_suffix():
    journal=KVJournal(); empty=hashlib.sha256(b'').hexdigest()
    journal.before(0,empty); journal.append(1,'a'*64); journal.append(2,'b'*64)
    journal.crop(1,'a'*64)
    with pytest.raises(ValueError): journal.before(2,'b'*64)
    journal.append(2,'c'*64); journal.before(2,'c'*64)
    with pytest.raises(ValueError): journal.before(1,'c'*64)


@pytest.mark.parametrize('condition',['fixed','risk','all35','fullrisk'])
@pytest.mark.parametrize('force_reject',[False,True])
def test_physical_readout_generator_with_real_tiny_model(condition,force_reject):
    target=tiny_model(); draft=copy.deepcopy(target)
    if force_reject:
        with torch.no_grad(): draft.lm_head.weight.neg_()
    pager=OutputPageDraft(extract_ffns(draft),device='cpu',group=4,slots=1,page_width=8)
    index=dict(vector_prior=np.zeros((4,16)),vector_covariance=np.eye(4),feature_variance=np.ones(16))
    rows=[]; rounds=[]; events=[]
    pager.page_cache.sink=events.append
    runtime=(VocabularyReadout if condition=='fullrisk' else RiskReadout)(draft,pager,index,rows.append,{},budget=2)
    runtime.reset(condition)
    prefix=torch.tensor([[1,2,3,4]])
    try:
        reference=generate(target,prefix,(999,),cap=5)
        actual=generate(target,prefix,(999,),cap=5,draft=draft,pager=runtime,record=rounds.append,copy_receipt=True)
        assert actual['ids']==reference['ids']
        assert actual['generator_copies']==dict(h2d_bytes=0,d2h_bytes=0,operations={})
        calls=[r for r in rows if r['kind']=='call']
        assert all(r['prefill']==(i<4) for i,r in enumerate(calls))
        assert all(r['prior_sha']==r['prior_after_sha'] and r['post_sha']==r['post_after_sha'] for r in calls)
        assert len(events)==3*(4*4+(len(calls)-4)*(4 if condition=='all35' else 2))
        assert all(r['draft_cache']==r['target_cache'] for r in rounds)
        if force_reject: assert any(r['kind']=='crop' for r in rows)
    finally:
        runtime.close(); pager.restore()


def test_gate_does_not_turn_equal_acceptance_and_equal_traffic_into_win():
    common=dict(accepted=8,attempted=16,acceptance=.5,h2d_per_accepted=100.,charged_wall_seconds=1.)
    conditions={name:dict(common) for name in ('fixed','risk','all35')}
    verdict=screen_decision(conditions,.1)
    assert not verdict['gates']['Hacquisition'] and verdict['gates']['Hruntime']
    assert not verdict['beats_resident_target_clock']
    conditions['risk']['accepted']=0; conditions['risk']['h2d_per_accepted']=None
    assert not screen_decision(conditions,.1)['gates']['Hacquisition']

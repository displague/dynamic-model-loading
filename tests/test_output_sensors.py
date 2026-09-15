import hashlib
import json

import numpy as np
import pytest
import torch

from dynamic_model_loading.output_pages import split_four_base
from dynamic_model_loading.output_sensors import CONFIG,FROZEN,validate_config
from dynamic_model_loading.output_sensors_analysis import decision,integer_identity,validate_rows,validate_pages,metrics,PAGE_BYTES,LAYER_BYTES


def test_sensor_protocol_is_frozen():
    validate_config(json.loads(CONFIG.read_text(encoding='utf-8')))
    with pytest.raises(ValueError):
        validate_config(dict(FROZEN,oracle_pages=18))


def test_integer_planes_cannot_bless_changed_parent():
    base=torch.arange(32,dtype=torch.uint8).reshape(4,8)
    lo,next_=split_four_base(base)
    data=base.numpy()
    expected=hashlib.sha256(data.tobytes()).hexdigest()
    integer_identity(data,lo.numpy(),next_.numpy(),expected)
    changed=data.copy(); changed[0,0]+=1
    with pytest.raises(ValueError,match='frozen parent'):
        integer_identity(changed,lo.numpy(),next_.numpy(),expected)
    broken=lo.numpy().copy(); broken[0,0]^=1
    with pytest.raises(ValueError,match='integer identity'):
        integer_identity(data,broken,next_.numpy(),expected)


def test_observation_speed_is_not_an_action_frontier():
    d=dict(all_projection_checks_pass=True,base_wrong=0,oracle_repair=0,oracle_new_error=0,fixed_new_error=0)
    stats=dict(fit=dict(d),diagnostic=dict(d))
    result=decision(stats,.01,.2)
    assert result['gates']==dict(Hobservation=True,Haction=False) and result['decision']=='stop'
    stats['diagnostic'].update(base_wrong=4,oracle_repair=2)
    assert decision(stats,.01,.2)['decision']=='eligible_for_short_estimation'
    stats['diagnostic']['oracle_new_error']=1
    assert not decision(stats,.01,.2)['gates']['Haction']
    assert not decision(stats,.3,.2)['gates']['Hobservation']


def frames():
    return [dict(document='d',split='fit',position=p,consumed_token=p+4,base_length=p+4,end_length=p+5,
        token_counter=p+6,prior_sha=('a' if p==0 else 'b')*64,prior_after_sha=('a' if p==0 else 'b')*64,post_sha='b'*64,post_after_sha='b'*64,
        kv_fingerprint_d2h_bytes=2*(9+2*p)*57344,kv_logical_bytes=2*(5+p)*57344,
        kv_storage_bytes=2*(5+p)*57344,full_logit_d2h_bytes=5*151936*4,auxiliary_d2h_bytes=12*1536*4+4,
        delta_d2h_bytes=35*1536*4,correction_scalar_d2h_bytes=108*(1+int(p==0)),base_seconds=.1,target_seconds=.1,
        high_readout_seconds=.01,oracle_selection_seconds=.01,fixed_seconds=.01,oracle_seconds=.01,
        observations=[dict(page=i,h2d_bytes=3*PAGE_BYTES,d2h_bytes=1536*4+12,wall_seconds=.001) for i in range(35)],
        oracle_pages=list(range(17)),replay=dict(kv_sha='b'*64,kv_d2h_bytes=5*57344,
            logit_d2h_bytes=151936*4,wall_seconds=.1) if p==0 else None) for p in range(16)]


@pytest.mark.parametrize('key,value',[('token_counter',999),('auxiliary_d2h_bytes',0),('kv_storage_bytes',0),
    ('prior_after_sha','z'*64),('full_logit_d2h_bytes',0),('base_seconds',float('nan')),
    ('fixed_seconds',0),('oracle_seconds',0),('oracle_selection_seconds',0),('high_readout_seconds',0)])
def test_sensor_accounting_rejects_missing_or_wrong_receipts(key,value):
    rows=frames()
    validate_rows(rows,[('d','fit')],{'d':list(range(20))})
    rows[0][key]=value
    with pytest.raises(ValueError):
        validate_rows(rows,[('d','fit')],{'d':list(range(20))})
    del rows[0][key]
    with pytest.raises(KeyError):
        validate_rows(rows,[('d','fit')],{'d':list(range(20))})


def test_matching_within_frame_hashes_do_not_hide_disconnected_history():
    rows=frames()
    rows[1]['prior_sha']=rows[1]['prior_after_sha']='c'*64
    with pytest.raises(ValueError,match='history chain'):
        validate_rows(rows,[('d','fit')],{'d':list(range(20))})


def test_separate_page_and_layer_cache_ledger():
    events=[dict(phase='mechanics',document=None,position=None,action=None,cache='layer',token=i+1,
        key=[i,s],bytes=LAYER_BYTES,kind='load',slot=1) for i in range(27) for s in range(2)]
    events += [dict(phase='mechanics',document=None,position=None,action=None,cache='page',token=28,
        key=[i,s],bytes=PAGE_BYTES,kind='load',slot=1) for i in range(35) for s in range(3)]
    receipt=validate_pages(events,[],[])
    assert receipt['page']['h2d_bytes']==105*PAGE_BYTES
    assert receipt['layer']['h2d_bytes']==54*LAYER_BYTES
    events[-1]['kind']='hit'
    with pytest.raises(ValueError):
        validate_pages(events,[],[])


def test_independent_sensor_geometry_and_third_token_receipts():
    zeros=lambda *shape:np.zeros(shape,dtype=np.float32)
    data={name:zeros(151936) for name in ('base','high','fixed','oracle','target')}
    h=np.ones(1536,dtype=np.float32)
    gain=np.ones_like(h)
    rows=zeros(2,1536); rows[0,0]=2; rows[1,0]=1
    normal=h/np.sqrt(np.mean(h*h)+1e-6)
    for value in data.values():
        value[0],value[1]=rows@normal
    data.update(h=h,residual=h.copy(),gain=gain,headrows=rows,basis=rows[0]-rows[1],base_y=zeros(1536),canonical_y=zeros(1536),
        fixed_y=zeros(1536),oracle_y=zeros(1536),deltas=zeros(35,1536),oracle_basis=zeros(1536),
        oracle_row=rows[0].copy(),oracle_utilities=zeros(35))
    row=frames()[1]
    row.update(base_ids=[0,1],high_id=0,epsilon=1e-6,grouped_relative_l2=0.)
    for obs in row['observations']:
        obs.update(predicted=float(data['base'][0]-data['base'][1]),actual=float(data['base'][0]-data['base'][1]),utility=0.)
    result=metrics(data,row)
    assert result['projection_pass'] and not result['base_wrong']
    data['fixed'][0]+=1
    with pytest.raises(ValueError,match='Full-logit pair'):
        metrics(data,row)

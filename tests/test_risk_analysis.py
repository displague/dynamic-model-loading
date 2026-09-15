import copy,hashlib
import pytest
from dynamic_model_loading.risk_analysis import audit_calls,validate_execution,validate_generation_timing,generator_copies


def rejection_fixture():
    rows=[]; previous=hashlib.sha256(b'').hexdigest()
    for i in range(8):
        prefill=i<4; digest=f'{i+1:064x}'
        policy=None if prefill else dict(pages=list(range(17)),observations=[0.]*17,axis=None,trace=None,
            base_margin=1.,base_pair=1.,final_pair=1.,predicted_pair=1.,base_rms=1.,final_rms=1.,
            base_ids=[1,2],acquisition_seconds=.01,observation_seconds=.005,nonprobe_acquisition_seconds=.005)
        rows.append(dict(kind='call',call=i+1,condition='fixed',prefill=prefill,consumed_id=i+1,next_id=i+2,
            base_length=i,end_length=i+1,prior_sha=previous,prior_after_sha=previous,post_sha=digest,post_after_sha=digest,
            kv_fingerprint_d2h_bytes=(4*i+2)*57344,draft_kv_logical_bytes=(i+1)*57344,draft_kv_storage_bytes=(i+1)*57344,
            axis_d2h_bytes=0,observation_d2h_bytes=0 if prefill else 68,readout_scalar_d2h_bytes=16 if prefill else 52,
            inherited_correction_d2h_bytes=216 if prefill else 108,policy=policy,wall_seconds=.02))
        previous=digest
    rows.append(dict(kind='crop',after_call=8,length=4,sha256=f'{4:064x}',kv_fingerprint_d2h_bytes=4*57344,
        draft_kv_logical_bytes=4*57344,draft_kv_storage_bytes=8*57344))
    rounds=[dict(proposed=[5,6,7,8],base=4,accepted=0,fallback=9,stop_reason='length')]
    return rows,rounds


def test_rejection_receipts_link_readouts_calls_crops_and_costs():
    rows,rounds=rejection_fixture(); result=audit_calls(rows,rounds,[1,2,3,4],'fixed',{})
    assert result['calls']==8 and result['crops']==1 and result['max_pair_error']==0
    assert len(result['expected_events'])==4*(54+105)+4*(27+51)


@pytest.mark.parametrize('case',['readout','history','crop_storage','observation_bytes','selected_page','margin','time','token_type'])
def test_runtime_receipt_tampering_fails_closed(case):
    rows,rounds=rejection_fixture(); rows=copy.deepcopy(rows)
    if case=='readout': rows[4]['next_id']=777
    if case=='history': rows[6]['prior_sha']='f'*64
    if case=='crop_storage': rows[-1]['draft_kv_storage_bytes']=4*57344
    if case=='observation_bytes': rows[4]['observation_d2h_bytes']=4
    if case=='selected_page': rows[4]['policy']['pages'][0]=17
    if case=='margin': rows[4]['policy']['final_pair']=2.
    if case=='time': rows[4]['policy']['observation_seconds']=.02
    if case=='token_type': rows[4]['next_id']=True
    with pytest.raises(ValueError): audit_calls(rows,rounds,[1,2,3,4],'fixed',{})


@pytest.mark.parametrize('key,value',[('device','cpu'),('cpu_threads',8),('cpu_threads',True),('tf32_matmul',True),('tf32_cudnn',None)])
def test_execution_settings_fail_closed(key,value):
    env=dict(device='cuda:0',cpu_threads=4,tf32_matmul=False,tf32_cudnn=False)
    validate_execution(env); env[key]=value
    with pytest.raises(ValueError): validate_execution(env)


@pytest.mark.parametrize('key,value',[('wall_seconds',99.),('finished_monotonic',30.),('decode_seconds',8.),('target_seconds',10.)])
def test_reference_timing_is_contained(key,value):
    row=dict(started_monotonic=1.,finished_monotonic=3.,wall_seconds=2.,prefill_seconds=.5,decode_seconds=1.5,target_seconds=1.)
    phase=dict(started_monotonic=0.,finished_monotonic=4.)
    validate_generation_timing(row,phase); row[key]=value
    with pytest.raises(ValueError): validate_generation_timing(row,phase)


def test_generator_copy_counts_charge_untruncated_commit_and_fallback():
    rows=[dict(accepted=3,fallback=9,stop_reason='rejected'),dict(accepted=4,fallback=None,stop_reason='length')]
    copy=generator_copies(rows,5)
    assert copy['d2h_bytes']==2*(128+32)
    assert copy['h2d_bytes']==2*(96+32)+8
    reference=generator_copies([],8,target_only=True)
    assert reference['h2d_bytes']==56 and reference['d2h_bytes']==64

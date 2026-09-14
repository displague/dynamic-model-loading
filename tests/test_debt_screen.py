import copy
import json
from unittest.mock import Mock

import pytest
import torch

from dynamic_model_loading.debt_analysis import policy_choices,audit_acquisition,decision,PAGE_BYTES
from dynamic_model_loading.debt_screen import CONFIG,FROZEN,validate_config
from dynamic_model_loading.fault_screen import supervise


def test_frozen_debt_subset_and_representation():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    for key,value in [('bits',4),('max_corrections',8),('extra_cuda_mib',768.0),('generation_tokens',64),('worker_timeout_seconds',301)]:
        with pytest.raises(ValueError,match='frozen'):
            validate_config(dict(cfg,**{key:value}))


def test_shared_supervisor_runs_only_named_worker_and_its_analyzer(tmp_path,monkeypatch):
    import dynamic_model_loading.fault_screen as supervisor
    process=Mock()
    process.wait.return_value=0
    launch=Mock(return_value=process)
    analyze=Mock(return_value=dict(decision='stop'))
    monkeypatch.setattr(supervisor.subprocess,'Popen',launch)
    result=supervise(tmp_path/'run',module='dynamic_model_loading.debt_screen',analyzer=analyze)
    assert result['decision']=='stop'
    assert launch.call_count==1 and 'dynamic_model_loading.debt_screen' in launch.call_args.args[0]
    assert analyze.call_count==1 and process.wait.call_args.kwargs=={'timeout':300}


def feedback_row():
    return dict(mode='debt',sketches=[[2.,0.],[1.,1.],[0.,2.]],
        choices=[dict(page=1,predicted_gain=10.,observed=[4.,0.]),
                 dict(page=2,predicted_gain=8.,observed=[0.,2.])],stop_gain=None,d2h_bytes=4*(3*2+2*2))


def test_independent_policy_replay_uses_actual_feedback():
    row=feedback_row()
    assert policy_choices(row,pages=3,rank=2,maximum=2)==[1,2]
    for field,value in [('page',0),('predicted_gain',4.)]:
        altered=copy.deepcopy(row)
        altered['choices'][1][field]=value
        with pytest.raises(ValueError):
            policy_choices(altered,pages=3,rank=2,maximum=2)
    bad=copy.deepcopy(row)
    bad['choices'][0]['observed']=[1.,1.]
    with pytest.raises(ValueError,match='acquisition'):
        policy_choices(bad,pages=3,rank=2,maximum=2)


def test_stop_and_fixed_controls_are_not_interchangeable():
    row=dict(mode='debt',sketches=[[10.],[-9.],[2.]],
        choices=[dict(page=2,predicted_gain=8.,observed=[2.])],stop_gain=-80.,d2h_bytes=16)
    assert policy_choices(row,pages=3,rank=1,maximum=2)==[2]
    with pytest.raises(ValueError,match='stopping'):
        policy_choices(dict(row,stop_gain=0),pages=3,rank=1,maximum=2)
    row=dict(mode='fixed',sketches=[[2.],[2.],[1.]],
        choices=[dict(page=0,predicted_gain=None,observed=[0.]),dict(page=1,predicted_gain=None,observed=[0.])],
        stop_gain=None,d2h_bytes=20)
    assert policy_choices(row,pages=3,rank=1,maximum=2)==[0,1]
    with pytest.raises(ValueError,match='early'):
        policy_choices(dict(row,choices=row['choices'][:1]),pages=3,rank=1,maximum=2)


def test_base_cannot_hide_physical_work(tmp_path):
    policy=tmp_path/'policy.jsonl'
    pages=tmp_path/'pages.jsonl'
    policy.write_text(json.dumps(dict(mode='base',layer=0,sketches=[],choices=[],stop_gain=None,d2h_bytes=0))+'\n')
    pages.write_text('')
    stats,counts=audit_acquisition(policy,pages,[('base',0)])
    assert not stats and counts['pages']==0
    key=dict(layer=0,page=1)
    events=[dict(key=key,outcome='load',request='demand',bytes=PAGE_BYTES,h2d_bytes=PAGE_BYTES,
                 used_bytes=PAGE_BYTES,evicted=[],wall_ms=1,cuda_ms=.5),
            dict(key=key,outcome='release',request='eager_release',bytes=PAGE_BYTES,used_bytes=0)]
    pages.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError,match='mismatch'):
        audit_acquisition(policy,pages,[('base',0)])


def test_real_cpu_runtime_policy_replays_independently():
    from dynamic_model_loading.residual_debt import ResidualDraft
    from dynamic_model_loading.adapters import extract_ffns
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(12)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    rows=[]
    pager=ResidualDraft(extract_ffns(model),[torch.eye(16)[:2]],page_width=8,group=4,rank=2,
                        max_pages=2,device='cpu',policy_sink=rows.append)
    try:
        for mode in ('base','fixed','debt','complete','dense_stream'):
            pager.mode=mode
            extract_ffns(model)[0](torch.randn(1,1,16))
            assert policy_choices(rows[-1],pages=4,rank=2,maximum=2,cuda=False)==[c['page'] for c in rows[-1]['choices']]
    finally:
        pager.restore()


def good_conditions():
    return dict(debt=dict(accepted=60,attempted=100,tokens=100,wall_seconds=90.,h2d_bytes=800),
        base=dict(accepted=40,attempted=100,tokens=100,wall_seconds=100.,h2d_bytes=0),
        fixed=dict(accepted=50,attempted=100,tokens=100,wall_seconds=100.,h2d_bytes=900),
        dense_stream=dict(accepted=100,attempted=100,tokens=100,wall_seconds=200.,h2d_bytes=1000))


def test_acquisition_must_add_value_beyond_compression_and_static_fetch():
    conditions=good_conditions()
    assert decision(conditions)['decision']=='eligible_for_expanded_protocol'
    for key,value in [('accepted',49),('wall_seconds',96),('h2d_bytes',901)]:
        altered=copy.deepcopy(conditions)
        altered['debt'][key]=value
        assert decision(altered)['decision']=='stop'
    conditions['base']['accepted']=60
    assert decision(conditions)['decision']=='stop'
    assert decision(conditions)['full_suite_launched'] is False
    assert decision(conditions)['native_admission_evaluated'] is False


def test_gates_use_aggregate_per_token_denominators():
    conditions=good_conditions()
    conditions['debt'].update(tokens=50,wall_seconds=45,h2d_bytes=400)
    assert decision(conditions)['decision']=='eligible_for_expanded_protocol'
    conditions['debt']['tokens']=0
    with pytest.raises(ValueError,match='denominator'):
        decision(conditions)


def test_charged_interval_includes_cleanup_and_rejects_understated_wall(monkeypatch):
    from dynamic_model_loading.debt_screen import charged_generate
    from dynamic_model_loading.debt_analysis import audit_timing
    import dynamic_model_loading.debt_screen as runner
    ticks=iter([10.,14.,15.])
    monkeypatch.setattr(runner.time,'perf_counter',lambda:next(ticks))
    inner=dict(started_monotonic=10.5,finished_monotonic=13.75,wall_seconds=3.,
               prefill_seconds=1.,decode_seconds=2.,target_seconds=.5)
    cleanup=Mock()
    row=charged_generate(lambda:inner,cleanup)
    cleanup.assert_called_once()
    audit_timing(row)
    assert row['wall_seconds']==5 and row['cleanup_seconds']==1 and row['wrapper_seconds']==1
    for key,value in [('wall_seconds',3.),('cleanup_seconds',0.),('prefill_seconds',0.),
                      ('target_seconds',4.),('wrapper_seconds',0.),('finished_monotonic',16.)]:
        with pytest.raises(ValueError):
            audit_timing(dict(row,**{key:value}))


def memory_fixture():
    baseline=6174857216+1550637056+4096
    resident,workspace=428343296,182353920
    def cuda(a,peak=None):
        return dict(allocated_bytes=a,reserved_bytes=a+2**20,peak_allocated_bytes=peak or a,
                    peak_reserved_bytes=a+4*2**20)
    account=dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        charged_baseline_bytes=6174857216+1550637056,
        non_ffn_baseline_allocated=baseline,baseline_cuda=cuda(baseline),resident_bytes=resident,
        workspace_bytes=workspace,construction_cuda=cuda(baseline+resident+workspace),construction_wall_seconds=1.)
    current=baseline+resident+workspace
    extra=resident+workspace+2*57344*12
    row=dict(mode='base',warmup=False,staging_bytes=PAGE_BYTES,cache={},
        target_kv_peak_bytes=57344*12,draft_kv_peak_bytes=57344*12,
        cuda=cuda(current,baseline+extra),extra_cuda_peak_bytes=extra+4096)
    return account,row,[dict(base=8)]


def test_physical_memory_lower_bounds_allocator_order_and_exact_kv():
    from dynamic_model_loading.debt_analysis import audit_allocation,audit_episode_memory
    account,row,rounds=memory_fixture()
    audit_allocation(account)
    audit_episode_memory(row,account,rounds)
    for key in ('target_kv_peak_bytes','draft_kv_peak_bytes','extra_cuda_peak_bytes'):
        with pytest.raises(ValueError):
            audit_episode_memory(dict(row,**{key:1}),account,rounds)
    for key,value in [('peak_allocated_bytes',account['non_ffn_baseline_allocated']+1),
                      ('peak_reserved_bytes',1),('allocated_bytes',account['non_ffn_baseline_allocated'])]:
        bad=copy.deepcopy(row)
        bad['cuda'][key]=value
        with pytest.raises(ValueError):
            audit_episode_memory(bad,account,rounds)
    for key,value in [('non_ffn_baseline_allocated',9000000000),
                      ('construction_cuda',account['baseline_cuda']),('construction_wall_seconds',float('nan'))]:
        with pytest.raises(ValueError):
            audit_allocation(dict(account,**{key:value}))


def test_baseline_overhead_is_fully_charged_not_waived_by_larger_slack():
    from dynamic_model_loading.debt_analysis import audit_allocation,audit_episode_memory
    account,row,rounds=memory_fixture()
    old_extra=row['extra_cuda_peak_bytes']
    overhead=24*2**20
    account['non_ffn_baseline_allocated']+=overhead
    for name in ('baseline_cuda','construction_cuda'):
        account[name]={k:v+overhead for k,v in account[name].items()}
    row['cuda']={k:v+overhead for k,v in row['cuda'].items()}
    row['extra_cuda_peak_bytes']+=overhead
    audit_allocation(account)
    audit_episode_memory(row,account,rounds)
    with pytest.raises(ValueError,match='budget'):
        audit_episode_memory(dict(row,extra_cuda_peak_bytes=old_extra),account,rounds)
    with pytest.raises(ValueError,match='parameter baseline'):
        audit_allocation(dict(account,charged_baseline_bytes=account['non_ffn_baseline_allocated']))
    bad=copy.deepcopy(row)
    bad['cuda']['peak_allocated_bytes']=bad['cuda']['allocated_bytes']
    bad['extra_cuda_peak_bytes']=bad['cuda']['peak_allocated_bytes']-account['charged_baseline_bytes']
    with pytest.raises(ValueError,match='budget'):
        audit_episode_memory(bad,account,rounds)

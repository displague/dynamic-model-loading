import json
import subprocess
from unittest.mock import Mock

import pytest

from dynamic_model_loading import fault_screen as screen
from dynamic_model_loading.fault_pager_analysis import audit_rounds


def test_frozen_subset_cannot_be_silently_enlarged():
    cfg = json.loads(screen.CONFIG.read_text(encoding='utf-8'))
    screen.validate_config(cfg)
    for key, value in [('generation_tokens',64),('worker_timeout_seconds',301),
                       ('repetitions',3),('budget_mib',512.0),('diagnostic_indices',[0,2])]:
        with pytest.raises(ValueError,match='frozen subset'):
            screen.validate_config(dict(cfg,**{key:value}))


@pytest.mark.parametrize('accepted,bytes_,decision',[(50,90,'eligible_for_expanded_protocol'),
    (49,90,'stop'),(50,91,'stop'),(30,105,'stop')])
def test_screen_uses_fixed_acceptance_and_byte_predicates(accepted,bytes_,decision):
    report = screen.screen_decision(dict(eager=dict(attempted=100,accepted=50,h2d_bytes=100),
        prefetch=dict(attempted=100,accepted=accepted,h2d_bytes=bytes_)))
    assert report['decision']==decision
    assert report['full_suite_launched'] is False
    assert report['native_admission_evaluated'] is False


def test_bytes_are_charged_per_all_proposals_not_just_accepted():
    conditions = dict(eager=dict(attempted=100,accepted=30,h2d_bytes=100),
                      prefetch=dict(attempted=200,accepted=100,h2d_bytes=180))
    assert screen.screen_decision(conditions)['decision']=='eligible_for_expanded_protocol'
    conditions['prefetch']['attempted']=0
    with pytest.raises(ValueError,match='denominator'):
        screen.screen_decision(conditions)


def mock_worker(monkeypatch, waits):
    process = Mock()
    process.wait.side_effect = waits
    launcher = Mock(return_value=process)
    monkeypatch.setattr(screen.subprocess,'Popen',launcher)
    analyzer = Mock(return_value=dict(decision='stop',full_suite_launched=False))
    monkeypatch.setattr(screen,'analyze',analyzer)
    return process, launcher, analyzer


def test_timeout_kills_only_owned_worker_and_never_analyzes_partial_run(tmp_path,monkeypatch):
    proc, launch, audit = mock_worker(monkeypatch,[subprocess.TimeoutExpired('test',300),-9])
    result = screen.supervise(tmp_path/'run')
    assert result['status']=='timeout'
    proc.kill.assert_called_once_with()
    assert proc.wait.call_args_list[0].kwargs=={'timeout':300}
    assert launch.call_count==1
    audit.assert_not_called()
    assert json.loads((tmp_path/'run/decision.json').read_text())['decision']=='inconclusive'
    with pytest.raises(FileExistsError):
        screen.supervise(tmp_path/'run')


@pytest.mark.parametrize('code',[1,42])
def test_failed_worker_cannot_pass_or_run_full_suite(tmp_path,monkeypatch,code):
    proc, launch, audit = mock_worker(monkeypatch,[code])
    assert screen.supervise(tmp_path/'run')['status']=='error'
    proc.kill.assert_not_called()
    audit.assert_not_called()
    assert launch.call_count==1


def test_complete_negative_screen_still_finishes_successfully(tmp_path,monkeypatch):
    proc, launch, audit = mock_worker(monkeypatch,[0])
    result = screen.supervise(tmp_path/'run')
    assert result['decision']=='stop'
    assert 'worker_wall_seconds' in result and 'analysis_wall_seconds' in result
    audit.assert_called_once()
    assert launch.call_count==1


def test_analysis_failure_is_never_a_pass(tmp_path,monkeypatch):
    _, _, audit = mock_worker(monkeypatch,[0])
    audit.side_effect = ValueError('tampered ledger')
    with pytest.raises(ValueError,match='tampered'):
        screen.supervise(tmp_path/'run')
    result = json.loads((tmp_path/'run/decision.json').read_text())
    assert result['decision']=='inconclusive' and result['reason']=='analysis_error'


def test_launch_failure_writes_inconclusive_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(screen.subprocess,'Popen',Mock(side_effect=OSError('no executable')))
    assert screen.supervise(tmp_path/'run')['reason']=='launch_error'
    assert json.loads((tmp_path/'run/decision.json').read_text())['decision']=='inconclusive'


def test_interruption_reaps_owned_worker_and_preserves_attempt(tmp_path,monkeypatch):
    proc, _, audit = mock_worker(monkeypatch,[KeyboardInterrupt(),-9])
    with pytest.raises(KeyboardInterrupt):
        screen.supervise(tmp_path/'run')
    proc.kill.assert_called_once_with()
    audit.assert_not_called()
    assert json.loads((tmp_path/'run/decision.json').read_text())['decision']=='inconclusive'


def test_round_replay_supports_short_cap_without_changing_historical_default(tmp_path):
    row = dict(round=1,base=4,proposed=[1,2,3,4],target_predictions=[1,2,3,4],
        accepted=4,emitted_accepted=2,fallback=None,committed=[1,2],
        target_cache=8,draft_cache=8,stop_reason='length')
    path = tmp_path/'rounds.jsonl'
    path.write_text(json.dumps(row)+'\n')
    episode = dict(ids=[1,2],attempted=4,accepted=2,stop_reason='length')
    assert audit_rounds(path,episode,prefix_tokens=4,generation_tokens=2)==28*8
    with pytest.raises(ValueError,match='prefix boundary'):
        audit_rounds(path,episode)
    with pytest.raises(ValueError,match='invalid round audit limits'):
        audit_rounds(path,episode,generation_tokens=True)


def test_cli_has_no_long_matrix_or_timeout_override():
    source = (screen.ROOT/'src/dynamic_model_loading/fault_screen.py').read_text()
    assert 'fault_pager_run --output' not in source
    assert "parser.add_argument('--timeout'" not in source


@pytest.mark.parametrize('patch',[dict(status='timeout'),dict(worker_exit_code=1),
    dict(timeout_seconds=600),dict(full_suite_launched=True),dict(worker_wall_seconds=302),
    dict(worker_wall_seconds=0)])
def test_replay_cannot_qualify_unbounded_or_unsupervised_run(tmp_path,patch):
    receipt = dict(status='complete',worker_exit_code=0,timeout_seconds=300,
                   full_suite_launched=False,worker_wall_seconds=120)
    path = tmp_path/'supervisor.json'
    path.write_text(json.dumps(receipt))
    assert screen.require_supervisor(tmp_path/'worker')==receipt
    path.write_text(json.dumps(dict(receipt,**patch)))
    with pytest.raises(ValueError,match='supervisor receipt'):
        screen.require_supervisor(tmp_path/'worker')


def test_both_logit_positions_are_checked_including_final_nonfinite():
    import torch
    ref = torch.ones(1,2,3)
    assert screen.full_logit_metrics(ref,ref)['logit_relative_l2']==0
    alt = ref.clone()
    alt[:,1,:] += 2
    assert screen.full_logit_metrics(ref,alt)['logit_relative_l2']>1
    alt[:,1,:] = float('nan')
    with pytest.raises(ValueError,match='Non-finite'):
        screen.full_logit_metrics(ref,alt)


def test_independent_allocation_audit_rejects_missing_charges():
    account = dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        draft_host_parameters_bytes=4624220160,controller_cuda_bytes=2877952,
        controller_host_bytes=2877952,shared_bytes=0,catalogue_aliases_host=True)
    episode = dict(staging_bytes=4718592,warmup=False,target_kv_peak_bytes=57344*40,
                   draft_kv_peak_bytes=57344*40,cuda=dict(peak_allocated_bytes=100,peak_reserved_bytes=101))
    screen.audit_allocation(account,[episode])
    for key in ('draft_host_parameters_bytes','controller_host_bytes','target_parameters_bytes'):
        with pytest.raises(ValueError,match='allocation'):
            screen.audit_allocation(dict(account,**{key:0}),[episode])
    with pytest.raises(ValueError,match='staging'):
        screen.audit_allocation(account,[dict(episode,staging_bytes=0)])
    with pytest.raises(ValueError,match='KV charge'):
        screen.audit_allocation(account,[dict(episode,target_kv_peak_bytes=0)])


def test_full_page_numerical_ledger_requires_all_35_rows(tmp_path):
    from dynamic_model_loading.fault_pager_analysis import audit_pages
    size = 4718592
    events = []
    for page in range(35):
        key = dict(layer=0,page=page)
        events += [dict(outcome='load',key=key,request='demand',bytes=size,h2d_bytes=size,
                        used_bytes=size,wall_ms=1,cuda_ms=1),
                   dict(outcome='release',key=key,request='eager_release',bytes=size,used_bytes=0)]
    events.append(dict(outcome='layer',layer=0,selected_pages=[list(range(35))],used_bytes=0))
    path = tmp_path/'pages.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    assert audit_pages(path,512*2**20,'eager',selected_pages=35)['load']==35
    with pytest.raises(ValueError,match='declared distinct'):
        audit_pages(path,512*2**20,'eager')


def test_numerical_hooks_use_adapter_source_and_preserve_output():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from dynamic_model_loading.adapters import extract_ffns
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    mlps = extract_ffns(model)
    cache = Mock()
    cache.record.return_value=None
    x = torch.zeros(1,2,16)
    before = [mlp(x) for mlp in mlps]
    handles = screen.full_page_hooks(mlps,cache)
    try:
        assert all(torch.equal(mlp(x),expected) for mlp,expected in zip(mlps,before,strict=True))
        assert [call.args[0]['layer'] for call in cache.record.call_args_list]==[0,1]
        assert all(call.args[0]['selected_pages']==[list(range(35))] for call in cache.record.call_args_list)
    finally:
        for handle in handles:
            handle.remove()

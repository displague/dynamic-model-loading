from pathlib import Path
import sys
from types import SimpleNamespace
import json

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import verification_offload as study


def test_child_settings_replace_case_insensitive_inherited_overrides_without_mutating_parent():
    parent = {'PATH': 'system-path', 'ggml_OP_OFFLOAD_MIN_BATCH': '999',
              'LLAMA_SPEC_FAKE_ACCEPT': '1', 'CUDA_VISIBLE_DEVICES': '7', 'unrelated': 'keep'}
    saved = parent.copy()
    child, effective, removed = study.child_environment(parent, 8, 2)
    assert parent == saved
    assert child == {'PATH': 'system-path', 'unrelated': 'keep', **effective}
    assert effective == {'LLAMA_TRACE': '1', 'GGML_OP_OFFLOAD_MIN_BATCH': '8', 'GGML_SCHED_DEBUG': '2'}
    assert set(removed) == set(parent) - {'PATH', 'unrelated'}
    with pytest.raises(ValueError):
        study.child_environment(parent, 4, 0)


def test_matrix_keeps_short_placement_fixed_and_has_separate_long_kv_control():
    configs = [study.configuration('short', k) for k in study.CONDITIONS if k != 'target']
    assert {(c['ngl'], c['threads'], c['context'], c['kv'], c['scheduler_debug']) for c in configs} == {
        (44, 24, 4096, 'f16', 0)}
    long = study.configuration('long', 'target')
    assert (long['k'], long['ngl'], long['context'], long['kv']) == (None, 38, 18432, 'q8_0')
    with pytest.raises(ValueError):
        study.configuration('mechanism', 'target')


def test_command_disables_offload_explicitly_and_changes_both_kv_allocations():
    checked = {'target': ['target.gguf'], 'draft05': ['draft.gguf']}
    disabled = study.command(Path('runtime'), checked, study.configuration('short', 'disabled-k16'), 8101)
    assert '--no-op-offload' in disabled
    long = study.command(Path('runtime'), checked, study.configuration('long', 'offload-k16'), 8101)
    for flag in ('-ctk', '-ctv', '--spec-draft-type-k', '--spec-draft-type-v'):
        assert long[long.index(flag)+1] == 'q8_0'
    assert long[long.index('-c')+1] == '18432'
    assert '--no-op-offload' not in long


def test_scheduler_evidence_is_not_a_copy_counter():
    raw = 'node # 12 (MUL_MAT): ffn_up-0 (100.0M) [CUDA0         ] use=1,c=0\n'
    raw += 'node # 13 (ADD): residual-0 (1.0K) [CPU         ]\n'
    parsed = study.scheduler_summary(raw)
    assert parsed['host_weight_offload_assignments'] is None
    assert not parsed['reason_tags_available']
    assert parsed['mul_mat_assignments_by_backend'] == {'CUDA0': 1}
    assert parsed['assignment_count'] == 2
    assert parsed['nodes'][0]['operation'] == 'MUL_MAT'
    assert 'not executed-operation counts' in parsed['limitation']


def test_resource_bounds_include_all_samples_and_fail_on_missing_telemetry():
    class R:
        rows = []
        errors = []
    assert not study.resource_check(R)['pass']
    R.rows = [{'gpu': {'used': 14000*2**20}, 'host': {'available': 3*2**30}},
              {'gpu': {'used': 15001*2**20}, 'host': {'available': 3*2**30}}]
    assert not study.resource_check(R)['pass']
    R.rows.pop()
    assert study.resource_check(R)['pass']
    R.errors = ['sample failed']
    assert not study.resource_check(R)['pass']


def test_setup_failure_is_retained_without_touching_an_older_attempt(tmp_path, monkeypatch):
    out = tmp_path/'fresh'
    def fail(args):
        args.output.mkdir()
        raise OSError('occupied port')
    monkeypatch.setattr(study, '_run', fail)
    with pytest.raises(OSError):
        study.run(SimpleNamespace(output=out))
    receipt = (out/'failure.json').read_bytes()
    assert json.loads(receipt)['message'] == 'occupied port'
    with pytest.raises(FileExistsError):
        study.run(SimpleNamespace(output=out))
    assert (out/'failure.json').read_bytes() == receipt

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import stock_study as study


def test_native_failure_is_not_an_oom_or_general_error_escape():
    failure = {'type': 'RuntimeError', 'message': 'server exited 3221226505'}
    log = 'ggml-cuda.cu:108: CUDA error'
    assert study.known_native_startup_failure(failure, log, [], 'draft32', 7, 0)
    assert not study.is_resource_failure(failure, log, [])
    assert not study.known_native_startup_failure(failure, log, [], 'draft32', 11, 0)
    assert not study.known_native_startup_failure(failure, log+'\nlaunch_slot_', [], 'draft32', 7, 0)
    assert not study.known_native_startup_failure(failure, log, [], 'draft32', 7, 1)
    assert not study.known_native_startup_failure(failure, log, [{'error': 'NVML', 'error_type': 'RuntimeError'}], 'draft32', 7, 0)


def test_continuation_preserves_bytes_and_rejects_changed_scores(tmp_path, monkeypatch):
    prior = tmp_path/'prior';prior.mkdir()
    study.write_json(prior/'environment.json', {})
    prior.with_suffix('.log').write_text('original driver log', encoding='utf-8')
    cases = []
    for name, draft, ngl, complete in [('target-ngl45-t8', None, 45, True), ('draft32-ngl7-t8', 'draft32', 7, False)]:
        directory = prior/name;directory.mkdir()
        manifest = {'head': 'current', 'mode': 'calibration', 'draft': draft, 'ngl': ngl, 'threads': 8, 'k': 16 if draft else None}
        study.write_json(directory/'manifest.json', manifest)
        (prior/(name+'.driver.log')).write_text('original case driver log', encoding='utf-8')
        if complete:
            study.write_json(directory/'completion.json', {})
            rows = [{'warmup': i == 0, 'resource_pass': True, 'sampled_gpu_peak': 100,
                     'response': {'timings': {'predicted_ms': 12}}} for i in range(3)]
            (directory/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        else:
            study.write_json(directory/'failure.json', {'type': 'RuntimeError', 'message': 'server exited 3221226505'})
            (directory/'server.log').write_text('ggml-cuda.cu:108: CUDA error', encoding='utf-8')
            (directory/'resources.jsonl').write_text('', encoding='utf-8')
        cases.append(study.summarize(directory)|{'name': name, 'draft': draft, 'ngl': ngl, 'threads': 8,
                                               'k': 16 if draft else None, 'returncode': 0 if complete else 1})
    ledger = prior/'driver-ledger.jsonl'
    ledger.write_text(''.join(json.dumps(c)+'\n' for c in cases), encoding='utf-8')
    original = ledger.read_bytes()
    policy = {'prior_ledger_sha256': study.digest(ledger), 'prior_cases': 2, 'prior_completed_cases': 1,
              'terminal_case': 'draft32-ngl7-t8', 'allowed_new_cases': ['draft32-ngl7-t16', 'draft32-ngl7-t24']}
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *a, **k: 'current')
    monkeypatch.setattr(study, 'check_measurement_source', lambda *a: None)
    out = tmp_path/'continued';out.mkdir()
    imported = study.import_calibration(prior, out, tmp_path, policy)
    assert ledger.read_bytes() == original == (out/'prior-driver-ledger.jsonl').read_bytes()
    assert (out/'target-ngl45-t8/rows.jsonl').read_bytes() == (prior/'target-ngl45-t8/rows.jsonl').read_bytes()
    assert imported[-1]['classification'] == 'native_startup_failure'
    receipt = json.loads((out/'continuation.json').read_text(encoding='utf-8'))
    assert {f['path'] for f in receipt['metadata_files']} == {
        'target-ngl45-t8.driver.log', 'draft32-ngl7-t8.driver.log', 'prior-driver-ledger.jsonl',
        'prior-environment.json', 'prior-driver.log'}
    cases[0]['median_decode_ms'] = 1
    ledger.write_text(''.join(json.dumps(c)+'\n' for c in cases), encoding='utf-8')
    invalid = tmp_path/'invalid';invalid.mkdir()
    with pytest.raises(ValueError, match='prior ledger digest'):
        study.import_calibration(prior, invalid, tmp_path, policy)
    # Keeping only the last failed case cannot cause earlier measurements to be rerun.
    ledger.write_text(json.dumps(cases[-1])+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='prior ledger digest'):
        study.import_calibration(prior, invalid, tmp_path, policy)
    ledger.write_bytes(original)
    target_rows = prior/'target-ngl45-t8/rows.jsonl'
    rows = [json.loads(s) for s in target_rows.read_text(encoding='utf-8').splitlines()]
    rows[-1]['response']['timings']['predicted_ms'] = 100
    target_rows.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='summary differs'):
        study.import_calibration(prior, invalid, tmp_path, policy)


def test_measurement_code_change_cannot_reuse_calibration(tmp_path):
    files = {'catalog_sha256': 'configs/stock-speculation-artifacts.json',
             'workloads_sha256': 'data/stock-speculation-workloads.json',
             'smoke_sha256': 'data/dense-interface-fresh.jsonl', 'runner_sha256': 'scripts/stock_benchmark.py'}
    manifest = {'head': 'current', 'mode': 'calibration'}
    for key, name in files.items():
        path = tmp_path/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text('{}', encoding='utf-8')
        manifest[key] = study.digest(path)
    study.check_measurement_source(manifest, tmp_path, 'current')
    (tmp_path/'scripts/stock_benchmark.py').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='substrate changed'):
        study.check_measurement_source(manifest, tmp_path, 'current')

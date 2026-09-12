import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_verification_profile import analyze


def test_cuda_bytes_are_windowed_and_boundary_copies_are_explicit(tmp_path):
    with sqlite3.connect(tmp_path/'trace.sqlite') as db:
        db.execute('CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)')
        db.execute('INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000000000)')
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY '
                   '(start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER, srcKind INTEGER, dstKind INTEGER)')
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?)',
                       [(200, 400, 10000, 1, 1, 3), (1100, 1200, 99, 1, 1, 3),
                        (1300, 1400, 7, 2, 3, 1), (950, 1050, 777, 1, 1, 3)])
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)')
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?)', [(1400, 1600), (3400, 3600)])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (3100,3200,99,1,1,3)')
    requests = [{'id': 'calibration-explanation', 'warmup': True,
        'begin_unix_ns': 1000001000, 'end_unix_ns': 1000002000, 'seconds': 0.000001},
        {'id': 'code-cache', 'warmup': False,
        'begin_unix_ns': 1000003000, 'end_unix_ns': 1000004000, 'seconds': 0.000001}]
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests), encoding='utf-8')
    (tmp_path/'resources.jsonl').write_text(json.dumps({'gpu': {'used': 100}, 'host': {'available': 3*2**30}})+'\n', encoding='utf-8')
    (tmp_path/'completion.json').write_text(json.dumps({'profiler_exit_code': 0, 'resources': {'pass': True}}), encoding='utf-8')
    (tmp_path/'profiler-stop.json').write_text('{"exit_code": 0}', encoding='utf-8')
    result = analyze(tmp_path)
    assert result['cuda_capture_verified']
    window = result['windows'][0]
    assert sum(c['bytes'] for c in window['copy_types']) == 106
    assert window['boundary_crossing_copies_excluded'] == 1
    assert window['kernel_count'] == 1
    (tmp_path/'rows.jsonl').write_text(json.dumps(requests[0])+'\n', encoding='utf-8')
    (tmp_path/'completion.json').write_text(json.dumps({'profiler_exit_code': 1, 'resources': {'pass': False}}), encoding='utf-8')
    partial = analyze(tmp_path)
    assert partial['aligned_cuda_activity_available']
    assert not partial['cuda_capture_verified']
    assert not partial['lifecycle_complete'] and not partial['expected_request_coverage']


def test_export_without_cuda_activity_is_not_zero_work(tmp_path):
    with sqlite3.connect(tmp_path/'trace.sqlite') as db:
        db.execute('CREATE TABLE metadata (name TEXT)')
    result = analyze(tmp_path)
    assert not result['cuda_capture_verified']
    assert 'missing' in result['reason']

"""Read actual CUDA activities within profiled HTTP windows from Nsight SQLite."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import stock_benchmark as stock
from analyze_verification_offload import audit_trace


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def analyze(path):
    result = {'run_label': path.name, 'cuda_capture_verified': False}
    if (path/'failure.json').exists():
        result['failure'] = json.loads((path/'failure.json').read_text(encoding='utf-8'))
    completion = (json.loads((path/'completion.json').read_text(encoding='utf-8'))
                  if (path/'completion.json').exists() else None)
    stopped = (json.loads((path/'profiler-stop.json').read_text(encoding='utf-8'))
               if (path/'profiler-stop.json').exists() else None)
    resource_audit = audit_trace(path/'resources.jsonl') if (path/'resources.jsonl').exists() else {'pass': False}
    requests = read_rows(path/'rows.jsonl') if (path/'rows.jsonl').exists() else []
    coverage = ([(r['id'], r['warmup']) for r in requests] ==
                [('calibration-explanation', True), ('code-cache', False)])
    lifecycle = bool('failure' not in result and completion and completion.get('profiler_exit_code') == 0
                     and completion.get('resources', {}).get('pass') and stopped and stopped.get('exit_code') == 0)
    result.update(lifecycle_complete=lifecycle, completion=completion, stop=stopped,
                  resource_audit=resource_audit, expected_request_coverage=coverage)
    files = list(path.glob('*.sqlite'))
    if len(files) != 1:
        result['reason'] = 'expected exactly one SQLite export'
        return result
    export = files[0]
    result['sqlite_sha256'] = stock.digest(export)
    # URI read-only mode also prevents accidentally creating an empty replacement DB.
    with sqlite3.connect(export.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result['tables'] = sorted(tables)
        needed = {'TARGET_INFO_SESSION_START_TIME', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_KERNEL'}
        if not needed <= tables:
            result['reason'] = 'CUDA activity or UTC alignment tables missing'
            return result
        epoch = int(db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0])
        result['session_utc_epoch_ns'] = epoch
        result['all_capture_copy_count'] = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]
        result['all_capture_kernel_count'] = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
        labels = {}
        for table in ('ENUM_CUDA_MEMCPY_OPER', 'ENUM_CUDA_MEM_KIND'):
            if table in tables:
                labels[table] = [dict(r) for r in db.execute('SELECT * FROM '+table)]
        result['enum_labels'] = labels
        windows = []
        for request in requests:
            lo, hi = request['begin_unix_ns']-epoch, request['end_unix_ns']-epoch
            wall_delta = (request['end_unix_ns']-request['begin_unix_ns'])/1e9
            copies = [dict(r) for r in db.execute(
                'SELECT start, end, bytes, copyKind, srcKind, dstKind FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE start >= ? AND end <= ? ORDER BY start', (lo, hi))]
            kernels = db.execute('SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL '
                                 'WHERE start >= ? AND end <= ?', (lo, hi)).fetchone()
            types = {}
            for copy in copies:
                key = (copy['copyKind'], copy['srcKind'], copy['dstKind'])
                item = types.setdefault(key, {'copy_kind': key[0], 'src_kind': key[1], 'dst_kind': key[2],
                                              'count': 0, 'bytes': 0, 'summed_duration_ns': 0})
                item['count'] += 1
                item['bytes'] += copy['bytes']
                item['summed_duration_ns'] += copy['end']-copy['start']
            crossing = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                                  'WHERE start < ? AND end > ? AND NOT (start >= ? AND end <= ?)',
                                  (hi, lo, lo, hi)).fetchone()[0]
            windows.append({'id': request['id'], 'warmup': request['warmup'],
                            'relative_start_ns': lo, 'relative_end_ns': hi,
                            'clock_duration_difference_seconds': wall_delta-request['seconds'],
                            'copy_types': list(types.values()), 'kernel_count': kernels[0],
                            'summed_kernel_duration_ns': kernels[1], 'boundary_crossing_copies_excluded': crossing,
                            'copy_size_histogram': dict(sorted(Counter(c['bytes'] for c in copies).items())),
                            'copies': copies})
        result['windows'] = windows
        result['aligned_cuda_activity_available'] = bool(windows) and all(
            w['relative_start_ns'] >= 0 and w['relative_end_ns'] > w['relative_start_ns'] and
            abs(w['clock_duration_difference_seconds']) < 0.01 and w['kernel_count'] > 0 and w['copies']
            for w in windows)
        result['cuda_capture_verified'] = bool(lifecycle and coverage and resource_audit['pass']
                                               and result['aligned_cuda_activity_available'])
        result['limits'] = ['Windows include prefill, drafting and target verification; startup is excluded.',
                            'Only fully contained copy/kernel activities are summed.',
                            'Summed activity durations may overlap and are not critical-path time.',
                            'Profiled times do not enter stock throughput rankings.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(analyze(args.run), indent=2)+'\n')


if __name__ == '__main__':
    main()

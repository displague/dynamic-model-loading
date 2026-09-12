"""Summarize retained stock requests without changing or rerunning measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_study import evaluation_schedule


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stop_state(response):
    return {key: response[key] for key in ['stop_type', 'stopping_word', 'truncated']}


def validate_schedule(schedule, selection, selection_digest):
    if schedule != evaluation_schedule(selection['selected'], selection_digest):
        raise ValueError('schedule differs from the matrix and ordering required by frozen selection')


def validate_reference(manifest, target, mode):
    expected = {'draft': None, 'ngl': target['ngl'], 'threads': target['threads'],
                'k': None, 'mode': mode, 'order_seed': None, 'inputs_sha256': None}
    if any(key not in manifest or manifest[key] != value for key, value in expected.items()):
        raise ValueError('reference is not the selected target-only configuration and workload')


def compare(row, reference):
    if row['request'] != reference['request']:
        raise ValueError('candidate request differs from the frozen reference')
    candidate = row['response']['tokens']
    expected = reference['response']['tokens']
    prefix = 0
    for a, b in zip(candidate, expected):
        if a != b:
            break
        prefix += 1
    token_match = candidate == expected
    return {'token_match': token_match,
            'stop_match': stop_state(row['response']) == stop_state(reference['response']),
            'common_generated_prefix': prefix,
            'reference_length': len(expected), 'candidate_length': len(candidate),
            'reference_next': expected[prefix] if prefix < len(expected) else None,
            'candidate_next': candidate[prefix] if prefix < len(candidate) else None}


def load_run(path, head):
    manifest = read(path/'manifest.json')
    if manifest['head'] != head:
        raise ValueError('measurement sources differ')
    rows_path = path/'rows.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()] if rows_path.exists() else []
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate case in measurement ledger')
    for row in rows:
        response = row['response']
        if (not isinstance(response['tokens'], list)
                or not all(type(t) is int and t >= 0 for t in response['tokens'])
                or len(response['tokens']) != response['timings']['predicted_n']):
            raise ValueError('generated IDs do not reconcile with native count')
        times = [row['seconds'], response['timings']['predicted_ms'], response['timings']['prompt_ms']]
        if not all(math.isfinite(t) and t >= 0 for t in times) or row['seconds'] == 0:
            raise ValueError('invalid timing')
    complete = (path/'completion.json').exists()
    if complete and [r['id'] for r in rows] != manifest['case_order']:
        raise ValueError('completed run does not cover its declared case order')
    startup = read(path/'startup.json') if (path/'startup.json').exists() else {}
    valid = complete and startup.get('resource_pass', False) and all(r['resource_pass'] for r in rows)
    return {'manifest': manifest, 'rows': rows, 'complete': complete,
            'resource_pass': valid, 'startup': startup,
            'failure': read(path/'failure.json') if (path/'failure.json').exists() else None,
            'ledger_sha256': sha(rows_path) if rows_path.exists() else None}


def aggregate(rows):
    emitted = sum(len(r['response']['tokens']) for r in rows)
    steps = sum(max(0, r['response']['timings']['predicted_n']-1) for r in rows)
    decode_s = sum(r['response']['timings']['predicted_ms']/1000 for r in rows)
    wall_s = sum(r['seconds'] for r in rows)
    events = [event for r in rows for event in r['acceptance']]
    consistent = all(r['acceptance_summary']['consistent'] for r in rows)
    cycles = [e for e in events if not e['checkpoint_replay']]
    maximum = max((e['attempted'] for e in cycles), default=0)
    divide = lambda a, b: a/b if b else None
    return {'requests': len(rows), 'emitted_tokens': emitted, 'native_decode_steps': steps,
            'prompt_tokens': sum(len(r['request']['prompt']) for r in rows),
            'prompt_length_range': [min(len(r['request']['prompt']) for r in rows),
                                    max(len(r['request']['prompt']) for r in rows)] if rows else None,
            'prompt_plus_output_length_range': [
                min(len(r['request']['prompt'])+len(r['response']['tokens']) for r in rows),
                max(len(r['request']['prompt'])+len(r['response']['tokens']) for r in rows)] if rows else None,
            'decode_seconds': decode_s, 'request_seconds': wall_s,
            'prefill_seconds': sum(r['response']['timings']['prompt_ms']/1000 for r in rows),
            'emitted_per_decode_second': divide(emitted, decode_s),
            'native_decode_steps_per_second': divide(steps, decode_s),
            'emitted_per_request_second': divide(emitted, wall_s),
            'seconds_per_emitted_token': divide(wall_s, emitted),
            'stop_types': {s: sum(r['response']['stop_type'] == s for r in rows)
                           for s in sorted({r['response']['stop_type'] for r in rows})},
            'sampled_request_gpu_peak': max((r['sampled_gpu_peak'] for r in rows), default=None),
            'sampled_request_host_available_min': min((r['sampled_host_available_min'] for r in rows), default=None),
            'acceptance_reconciled': consistent,
            'recorded_final_cycles': len(cycles),
            'checkpoint_replay_events': sum(e['checkpoint_replay'] for e in events),
            'attempted_draft_tokens': sum(r['response']['timings'].get('draft_n', 0) for r in rows),
            'accepted_draft_tokens': sum(r['response']['timings'].get('draft_n_accepted', 0) for r in rows),
            'accepted_prefix_survival': [sum(e['accepted'] >= i for e in cycles)/len(cycles)
                                         for i in range(1, maximum+1)] if consistent else None,
            'attempted_length_counts': {str(i): sum(e['attempted'] == i for e in cycles)
                                        for i in sorted({e['attempted'] for e in cycles})} if consistent else None,
            'mean_accepted_per_cycle': divide(sum(e['accepted'] for e in cycles), len(cycles)) if consistent else None,
            'decode_seconds_per_recorded_cycle': divide(decode_s, len(cycles)) if consistent else None,
            'decode_steps_minus_accepted_plus_recorded_cycles':
                steps-sum(e['accepted']+1 for e in cycles) if consistent and cycles else None,
            'emitted_decode_steps_per_cycle': divide(steps, len(cycles)) if consistent else None}


def analyze(directory):
    if not (directory/'driver-complete.json').exists():
        raise ValueError('evaluation is incomplete; do not publish a final frontier')
    selection = read(directory/'frozen-selection.json')
    head = selection['source']
    schedule = read(directory/'schedule.json')
    validate_schedule(schedule, selection, sha(directory/'frozen-selection.json'))
    references = {}
    reference_manifests = {}
    evidence_hashes = {name: sha(directory/name) for name in ['schedule.json', 'frozen-selection.json',
                                                            'driver-ledger.jsonl', 'driver-complete.json']}
    def bind_run(run_name):
        for name in ['manifest.json', 'rows.jsonl', 'inputs.json', 'startup.json', 'completion.json', 'failure.json']:
            relative = f'{run_name}/{name}'
            evidence_hashes[relative] = sha(directory/relative) if (directory/relative).exists() else None
    for mode in ['sustained', 'smoke']:
        run = load_run(directory/f'reference-{mode}', head)
        validate_reference(run['manifest'], selection['selected']['target'], mode)
        bind_run(f'reference-{mode}')
        if not run['resource_pass']:
            raise ValueError('reference is incomplete or exceeds its resource allowance')
        references[mode] = {r['id']: r for r in run['rows'] if not r['warmup']}
        reference_manifests[mode] = run['manifest']
    expected_runs = {f'reference-{mode}' for mode in references}
    expected_runs.update(f'repeat{i}-{c["name"]}' for c in schedule['configs'] for i in [1, 2, 3])
    expected_runs.update(f'smoke-{c["name"]}' for c in schedule['configs'])
    ledger = [json.loads(line) for line in (directory/'driver-ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    if (len(ledger) != len(expected_runs) or {r['name'] for r in ledger} != expected_runs
            or read(directory/'driver-complete.json')['runs'] != len(expected_runs)):
        raise ValueError('driver did not retain the complete frozen matrix')
    configs = {}
    diagnostics = []
    for config in schedule['configs']:
        name = config['name']
        measured = []
        comparisons = []
        run_summaries = []
        all_pass = True
        for run_name, mode in [(f'repeat{i}-{name}', 'sustained') for i in [1, 2, 3]] + [(f'smoke-{name}', 'smoke')]:
            run = load_run(directory/run_name, head)
            bind_run(run_name)
            manifest = run['manifest']
            for field in ['catalog_sha256', 'workloads_sha256', 'smoke_sha256', 'runner_sha256']:
                if manifest[field] != reference_manifests[mode][field]:
                    raise ValueError('candidate substrate differs from reference')
            if manifest['inputs_sha256'] != sha(directory/f'reference-{mode}/inputs.json'):
                raise ValueError('candidate frozen-input digest mismatch')
            if any(manifest[k] != config[k] for k in ['draft', 'ngl', 'threads']):
                raise ValueError('run configuration differs from frozen schedule')
            if manifest['k'] != (config['k'] if config['draft'] else None) or manifest['mode'] != mode:
                raise ValueError('run mode or draft length differs from frozen schedule')
            expected_seed = schedule['prompt_seeds'][int(run_name[6])-1] if mode == 'sustained' else None
            if manifest['order_seed'] != expected_seed:
                raise ValueError('prompt ordering seed differs from frozen schedule')
            rows = [r for r in run['rows'] if not r['warmup']]
            if run['complete'] and {r['id'] for r in rows} != set(references[mode]):
                raise ValueError('completed run does not cover reference cases')
            for row in rows:
                ref = references[mode][row['id']]
                fidelity = compare(row, ref)
                comparisons.append({'run': run_name, 'id': row['id'], **fidelity})
                if not fidelity['token_match']:
                    prefix = fidelity['common_generated_prefix']
                    diagnostics.append({'run': run_name, 'id': row['id'], 'configuration': config,
                                        'tokens': ref['request']['prompt']+ref['response']['tokens'][:prefix],
                                        'first_difference': fidelity})
            if mode == 'sustained' and run['resource_pass']:
                measured.extend(rows)
            all_pass = all_pass and run['resource_pass']
            run_summaries.append({'name': run_name, 'resource_pass': run['resource_pass'],
                                  'failure': run['failure'], 'ledger_sha256': run['ledger_sha256'],
                                  'startup_seconds': run['startup'].get('seconds'),
                                  'sampled_startup_gpu_peak': run['startup'].get('sampled_gpu_peak'),
                                  'by_case': {r['id']: aggregate([r]) for r in rows},
                                  'summary': aggregate(rows)})
        configs[name] = {'configuration': config, 'all_runs_resource_pass': all_pass,
                         'all_recorded_outputs_match': bool(comparisons) and all(
                             x['token_match'] and x['stop_match'] for x in comparisons),
                         'comparisons': comparisons, 'runs': run_summaries,
                         'sustained_complete_run_summary': aggregate(measured)}
    baseline = configs['target']['sustained_complete_run_summary']
    for name, result in configs.items():
        current = result['sustained_complete_run_summary']
        same_coverage = (all(r['resource_pass'] for r in configs['target']['runs'] if r['name'].startswith('repeat'))
                         and all(r['resource_pass'] for r in result['runs'] if r['name'].startswith('repeat')))
        result['complete_sustained_timing_coverage'] = same_coverage
        result['observed_request_rate_ratio_vs_target'] = (
            current['emitted_per_request_second']/baseline['emitted_per_request_second']
            if same_coverage and current['emitted_per_request_second'] and baseline['emitted_per_request_second'] else None)
        paired = []
        for repeat in [1, 2, 3]:
            base_run = next(r for r in configs['target']['runs'] if r['name'] == f'repeat{repeat}-target')
            candidate = next(r for r in result['runs'] if r['name'] == f'repeat{repeat}-{name}')
            if not base_run['resource_pass'] or not candidate['resource_pass']:
                continue
            for case, base_case in base_run['by_case'].items():
                candidate_case = candidate['by_case'][case]
                paired.append({'repeat': repeat, 'id': case,
                               'reference_seconds': base_case['request_seconds'],
                               'candidate_seconds': candidate_case['request_seconds'],
                               'reference_tokens': base_case['emitted_tokens'],
                               'candidate_tokens': candidate_case['emitted_tokens'],
                               'request_rate_ratio': candidate_case['emitted_per_request_second']/base_case['emitted_per_request_second']})
        result['paired_requests_vs_target'] = paired
    return {'source': head, 'schedule_sha256': sha(directory/'schedule.json'), 'evidence_sha256': evidence_hashes,
            'configurations': configs, 'first_divergence_diagnostics': diagnostics,
            'limitations': ['Timings on divergent outputs do not establish exact-output acceleration.',
                            'Survival includes actual draft truncation and uses all reconciled cycles as denominator.',
                            'Native decode interval excludes the first emitted token.',
                            'Startup and warmup are excluded from warm sustained aggregates.',
                            'Cycle-normalized request totals can include direct decode steps outside recorded speculative cycles.',
                            'Logs do not isolate draft, verification and coordination component times.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('analysis source must be committed and clean')
    value = analyze(args.evaluation.resolve())
    value['analysis_source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    value['analysis_script_sha256'] = sha(Path(__file__))
    value['python'] = sys.version
    with args.output.open('x', encoding='utf-8') as out:
        out.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()

"""Reproduce delivery tables from immutable stock analysis and diagnostic receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from stock_benchmark import completion_payload
from stock_replay import replay_plan, validate_analysis_receipts


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        assert not path.is_symlink() and not path.is_junction()
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = digest(path)
    return result


def resources(path):
    samples = rows(path)
    assert samples and all('gpu' in s and 'host' in s for s in samples)
    peak = max(s['gpu']['used'] for s in samples)
    available = min(s['host']['available'] for s in samples)
    return {'sha256': digest(path), 'samples': len(samples), 'peak_gpu_bytes': peak,
            'minimum_host_available_bytes': available,
            'passes': peak <= 15000*2**20 and available >= 2*2**30}


def buffers(path):
    # Preserve the rounded native records; do not equate their sum with residency.
    records = []
    owner = -1
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.search(r'\b(CUDA0|CUDA_Host|CPU)\s+(model|KV|compute|output) buffer size\s*=\s*([\d.]+) MiB', line)
        if match:
            device, kind, size = match.groups()
            if device == 'CUDA0' and kind == 'model':
                owner += 1
            assert owner in [0, 1]
            records.append({'owner': ['target', 'draft'][owner], 'device': device,
                            'kind': kind, 'MiB': float(size), 'source_line': line})
    return {'sha256': digest(path), 'records': records}


def label(config):
    return f'{config["draft"] or "target"}-ngl{config["ngl"]}-t{config["threads"]}-k{config["k"]}'


def summarize(args):
    analysis = read(args.analysis)
    validate_analysis_receipts(analysis, args.evaluation)
    configurations = analysis['configurations']
    names = ['reference-sustained', 'reference-smoke'] + [r['name'] for c in configurations.values() for r in c['runs']]
    assert len(names) == len(set(names)) == 54
    audits = {name: resources(args.evaluation/name/'resources.jsonl') for name in names}
    requests = [r for name in names for r in rows(args.evaluation/name/'rows.jsonl')]
    output = {'analysis_sha256': digest(args.analysis), 'measurement_source': analysis['source'],
              'builder_sha256': digest(Path(__file__)), 'groups': len(names),
              'requests': len(requests), 'warmups': sum(r['warmup'] for r in requests),
              'all_resource_traces_pass': all(a['passes'] for a in audits.values()),
              'resource_audits': audits, 'configurations': {}}
    for name, config in configurations.items():
        aggregate = config['sustained_complete_run_summary']
        observed = {mode: [r for r in config['comparisons'] if r['run'].startswith(prefix)]
                    for mode, prefix in [('sustained', 'repeat'), ('smoke', 'smoke')]}
        value = {'configuration': config['configuration'],
                 'fidelity': {mode: {'matches': sum(r['token_match'] and r['stop_match'] for r in comparisons),
                                     'total': len(comparisons)} for mode, comparisons in observed.items()},
                 'sustained': aggregate,
                 'repeat_request_rates': [r['summary']['emitted_per_request_second'] for r in config['runs'] if r['name'].startswith('repeat')],
                 'observed_rate_ratio_vs_target': config['observed_request_rate_ratio_vs_target'],
                 'peak_gpu_bytes': max(audits[r['name']]['peak_gpu_bytes'] for r in config['runs']),
                 'minimum_host_available_bytes': min(audits[r['name']]['minimum_host_available_bytes'] for r in config['runs']),
                 'native_buffer_records': buffers(args.evaluation/f'repeat1-{name}'/'server.log')}
        cycles = aggregate['recorded_final_cycles']
        if cycles:
            value['request_seconds_per_recorded_cycle'] = aggregate['request_seconds']/cycles
            value['emitted_tokens_per_recorded_cycle'] = aggregate['emitted_tokens']/cycles
        output['configurations'][name] = value
    fastest = max((name for name, c in configurations.items() if c['configuration']['draft']),
                  key=lambda name: configurations[name]['sustained_complete_run_summary']['emitted_per_request_second'])
    output['fastest_observed_draft'] = fastest
    target_runs = {r['name'].split('-', 1)[0]: r for r in configurations['target']['runs']
                   if r['name'].startswith('repeat')}
    for name, config in configurations.items():
        episodes = {}
        for run in config['runs']:
            if not run['name'].startswith('repeat'):
                continue
            repeat = run['name'].split('-', 1)[0]
            reference = target_runs[repeat]
            assert set(run['by_case']) == set(reference['by_case'])
            for case, measured in run['by_case'].items():
                baseline = reference['by_case'][case]
                episodes.setdefault(case, []).append({
                    'repeat': repeat, 'candidate_request_seconds': measured['request_seconds'],
                    'baseline_request_seconds': baseline['request_seconds'],
                    'candidate_emitted_tokens': measured['emitted_tokens'],
                    'baseline_emitted_tokens': baseline['emitted_tokens'],
                    'observed_request_rate_ratio': measured['emitted_per_request_second']/baseline['emitted_per_request_second']})
        assert len(episodes) == 6 and all(len(values) == 3 for values in episodes.values())
        output['configurations'][name]['paired_episodes'] = {
            case: {'repeats': values,
                   'aggregate_observed_request_rate_ratio':
                       (sum(v['candidate_emitted_tokens'] for v in values)/sum(v['candidate_request_seconds'] for v in values))/
                       (sum(v['baseline_emitted_tokens'] for v in values)/sum(v['baseline_request_seconds'] for v in values))}
            for case, values in sorted(episodes.items())}
    output['hypothetical_hurdles'] = [
        {'comparison': name, 'hypothetical_complete_cycle_seconds': cost, 'desired_rate_factor': factor,
         'required_expected_emitted_tokens': factor*cost*configurations[name]['sustained_complete_run_summary']['emitted_per_request_second']}
        for name in ['target', fastest] for cost in [0.1, 0.2, 0.4, 0.8] for factor in [1, 2, 3]]

    plan = read(args.replay/'plan.json')
    assert plan == read(args.original_replay/'plan.json') == replay_plan(
        analysis, read(args.evaluation/'frozen-selection.json'))
    assert read(args.replay/'complete.json')['groups'] == len(plan)
    ports = read(args.replay/'ports.json')
    assert set(ports) == set(plan) and len(set(ports.values())) == len(plan)
    replay_rows = {}
    replay_audits = {}
    for name, group in plan.items():
        directory = args.replay/name
        manifest = read(directory/'manifest.json')
        assert manifest['analysis_sha256'] == digest(args.analysis)
        assert manifest['measurement_head'] == analysis['source']
        assert manifest['configuration'] == group['configuration'] and manifest['cases'] == group['cases']
        assert int(manifest['command'][manifest['command'].index('--port')+1]) == ports[name]
        actual = rows(directory/'rows.jsonl')
        indexed = {r['prefix_sha256']: r for r in actual}
        assert len(indexed) == len(actual) == read(directory/'completion.json')['cases']
        assert set(indexed) == set(group['cases'])
        for key, row in indexed.items():
            case = group['cases'][key]
            payload = completion_payload(case['tokens'], 1)
            payload.update(n_probs=10, post_sampling_probs=False)
            assert row['request'] == payload and row['origins'] == case['origins']
            assert len(row['response']['tokens']) == 1
            assert row['response'] == read(directory/f'response-{key}.json')
        replay_rows[name] = indexed
        replay_audits[name] = resources(directory/'resources.jsonl')
    target = configurations['target']['configuration']
    baseline = label(target)
    original = {r['prefix_sha256']: r for r in rows(args.original_replay/baseline/'rows.jsonl')}
    assert set(original) == set(replay_rows[baseline])
    replication = {'prefixes': len(original),
                   'returned_token_matches': sum(r['response']['tokens'] == replay_rows[baseline][key]['response']['tokens'] for key, r in original.items()),
                   'top_probability_matches': sum(r['response']['completion_probabilities'] == replay_rows[baseline][key]['response']['completion_probabilities'] for key, r in original.items())}
    triads = {}
    for item in analysis['first_divergence_diagnostics']:
        key = hashlib.sha256(json.dumps(item['tokens']).encode('utf-8')).hexdigest()
        config = item['configuration']
        matched = dict(config, draft=None, k=16)
        names = [baseline, label(matched), label(config)]
        tokens = [replay_rows[name][key]['response']['tokens'][0] for name in names]
        unique = (config['name'], item['id'], key)
        entry = triads.setdefault(unique, {'configuration': config['name'], 'id': item['id'],
                    'prefix_sha256': key, 'prefix_tokens': len(item['tokens']),
                    'original_reference_next': item['first_difference']['reference_next'],
                    'original_candidate_next': item['first_difference']['candidate_next'],
                    'rebuilt_baseline_next': tokens[0], 'rebuilt_matched_next': tokens[1],
                    'rebuilt_candidate_next': tokens[2], 'origins': []})
        assert [entry[f'rebuilt_{role}_next'] for role in ['baseline', 'matched', 'candidate']] == tokens
        entry['origins'].append(item['run'])
    output['replay'] = {'plan_sha256': digest(args.replay/'plan.json'), 'groups': len(plan),
                        'cases': sum(len(g['cases']) for g in plan.values()),
                        'receipt_sha256': inventory(args.replay),
                        'original_receipt_sha256': inventory(args.original_replay),
                        'driver_log_sha256': digest(args.replay.with_suffix('.log')),
                        'original_driver_log_sha256': digest(args.original_replay.with_suffix('.log')),
                        'resource_audits': replay_audits, 'baseline_replication': replication,
                        'triads': list(triads.values()),
                        'baseline_original_reference_matches': sum(
                            r['response']['tokens'][0] == r['origins'][0]['first_difference']['reference_next']
                            for r in replay_rows[baseline].values())}
    output['limitations'] = [
        'Rates on divergent outputs do not qualify as identical-output acceleration.',
        'Request/cycle normalization includes prefill and direct decode steps; it does not isolate individual cycle or component timings.',
        'Hypothetical hurdles hold a supplied fully charged cycle cost fixed; they are not measurements or forecasts.',
        'Native buffer records are rounded allocation logs, not proof of physical residency or absence of eviction.',
        'One-token rebuilt-prefix diagnostics do not recreate historical KV state or original verification-batch shapes.']
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'evaluation', 'replay', 'original-replay', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    arguments = parser.parse_args()
    value = summarize(arguments)
    with arguments.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

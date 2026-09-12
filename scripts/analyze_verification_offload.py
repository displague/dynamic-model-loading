"""Compact audit of the complete stock threshold study; never infer missing CUDA bytes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import stock_benchmark as stock
import verification_offload as study


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def audit_trace(path):
    samples = rows(path)
    valid = [r for r in samples if 'gpu' in r and 'host' in r]
    return {'samples': len(valid), 'errors': [r for r in samples if 'gpu' not in r or 'host' not in r],
            'gpu_peak': max((r['gpu']['used'] for r in valid), default=None),
            'host_available_min': min((r['host']['available'] for r in valid), default=None),
            'pass': bool(valid) and len(samples) == len(valid) and all(
                r['gpu']['used'] <= 15000*2**20 and r['host']['available'] >= 2*2**30 for r in valid)}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def audit_acceptance(log, request):
    start, end = request['log_start'], request['log_end']
    if not 0 <= start <= end <= len(log):
        raise ValueError('invalid native log byte interval')
    segment = log[start:end].decode('utf-8', errors='replace')
    canonical = '\n'.join(line for line in segment.splitlines() if 'new n_tokens =' not in line)
    events = stock.parse_acceptance(canonical)
    summary = stock.acceptance_summary(events, request['response']['timings'])
    if events != request['acceptance'] or summary != request['acceptance_summary'] or not summary['consistent']:
        raise ValueError('acceptance does not reproduce from native log bytes and counters')
    return summary


def summarize_requests(data):
    emitted = sum(len(r['response']['tokens']) for r in data)
    seconds = sum(r['seconds'] for r in data)
    prompt_ms = sum(r['response']['timings']['prompt_ms'] for r in data)
    decode_ms = sum(r['response']['timings']['predicted_ms'] for r in data)
    prompt_tokens = sum(r['response']['timings'].get('prompt_n', 0) for r in data)
    decode_steps = sum(max(r['response']['timings'].get('predicted_n', len(r['response']['tokens']))-1, 0)
                       for r in data)
    events = [e for r in data for e in r['acceptance'] if not e['checkpoint_replay']]
    consistent = all(r['acceptance_summary']['consistent'] for r in data)
    cycles = len(events)
    maximum = max((e['attempted'] for e in events), default=0)
    return {'requests': len(data), 'emitted_tokens': emitted, 'request_seconds': seconds,
            'emitted_per_request_second': emitted/seconds if seconds else None,
            'native_prompt_seconds': prompt_ms/1000, 'native_decode_seconds': decode_ms/1000,
            'native_prompt_tokens': prompt_tokens,
            'native_prompt_tokens_per_second': prompt_tokens/(prompt_ms/1000) if prompt_ms else None,
            'native_decode_steps': decode_steps,
            'native_decode_steps_per_second': decode_steps/(decode_ms/1000) if decode_ms else None,
            'emitted_per_native_decode_second': emitted/(decode_ms/1000) if decode_ms else None,
            'acceptance_consistent': consistent, 'cycles': cycles,
            'accepted': sum(e['accepted'] for e in events), 'attempted': sum(e['attempted'] for e in events),
            'attempted_length_histogram': dict(sorted(Counter(e['attempted'] for e in events).items())),
            'accepted_prefix_histogram': dict(sorted(Counter(e['accepted'] for e in events).items())) if consistent else None,
            'survival': [sum(e['accepted'] >= i for e in events)/cycles for i in range(1, maximum+1)]
                        if consistent and cycles else None,
            'request_ms_per_recorded_cycle': 1000*seconds/cycles if cycles else None,
            'normalization_limit': 'Whole requests include prefill and residual direct decoding; this is not isolated cycle timing.'}


def mechanism(log, request):
    """Retain the first target verification graph before any subsequent draft graph."""
    segment = log[request['log_start']:request['log_end']].decode('utf-8', errors='replace')
    start = re.search(r'generate_draft: id=(\d+), #tokens=(\d+), #draft=(\d+), pos_next=(\d+)', segment)
    if not start:
        return {'available': False, 'reason': 'no native target draft-batch log'}
    tail = segment[start.start():]
    end = re.search(r'accepted\s+\d+/\s*\d+ draft tokens', tail)
    if not end:
        return {'available': False, 'reason': 'no corresponding acceptance event'}
    tail = tail[:end.end()]
    assignments = study.scheduler_summary(tail)['nodes']
    # The first graph following generate_draft is the target graph. Following graphs
    # may belong to the draft model. Stop when node numbering resets.
    graph = []
    previous = -1
    for n in assignments:
        if n['node'] <= previous:
            break
        graph.append(n)
        previous = n['node']
    target_up = [n for n in graph if n['operation'] == 'MUL_MAT'
                 and re.match(r'ffn_up-(?:[0-9]|1[0-9]|20)\s', n['tensor'])]
    return {'available': bool(target_up), 'sampled_input_token': int(start[1]),
            'prefix_tokens': int(start[2]), 'draft_tokens': int(start[3]), 'position_next': int(start[4]),
            'constructed_target_batch_tokens': int(start[3])+1,
            'shape_basis': 'Native draft count and pinned server single-slot construction; 256-token batch capacity is not this size.',
            'first_graph_nodes': len(graph), 'cold_layer_up_assignments': target_up,
            'cold_up_backend_counts': dict(Counter(n['assignment'].split()[0] for n in target_up)),
            'physical_copy_bytes': None}


def analyze(root, reference):
    originals = {r['id']: r['response']['tokens'] for r in rows(reference/'rows.jsonl') if not r['warmup']}
    grouped = defaultdict(list)
    group_records, failures, mechanisms = [], [], {}
    long_reference = None
    first_long = root/'long-r1-target/rows.jsonl'
    if first_long.exists():
        long_reference = {r['id']: r['response']['tokens'] for r in rows(first_long) if not r['warmup']}
    expected_short_inputs = None
    first_long_inputs = root/'long-r1-target/long-inputs.json'
    long_input_hash = stock.digest(first_long_inputs) if first_long_inputs.exists() else None
    repeated = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir() and (p/'manifest.json').exists()):
        manifest = read(path/'manifest.json')
        stage, condition, repeat = manifest['stage'], manifest['condition'], manifest['repeat']
        cfg = study.configuration(stage, condition)
        if path.name != f'{stage}-r{repeat}-{condition}':
            raise ValueError('run directory and manifest identity differ')
        if manifest['configuration'] != cfg:
            raise ValueError(f'configuration mismatch: {path.name}')
        _, effective, _ = study.child_environment({}, cfg['threshold'], cfg['scheduler_debug'])
        if manifest['effective_runtime_environment'] != effective:
            raise ValueError(f'environment mismatch: {path.name}')
        failure = read(path/'failure.json') if (path/'failure.json').exists() else None
        completed = read(path/'completion.json') if (path/'completion.json').exists() else None
        raw = rows(path/'rows.jsonl') if (path/'rows.jsonl').exists() else []
        audit = audit_trace(path/'resources.jsonl') if (path/'resources.jsonl').exists() else {'pass': False}
        if failure or not completed or not completed.get('complete'):
            failures.append({'group': path.name, 'failure': failure, 'completion': completed, 'resources': audit,
                             'retained_requests': len(raw)})
            continue
        if not audit['pass']:
            raise ValueError(f'successful group has failing raw resource trace: {path.name}')
        native_bytes = (path/'server.log').read_bytes()
        native_log = native_bytes.decode('utf-8', errors='replace')
        frozen = read(path/'inputs.json')
        reference_inputs = read(first_long_inputs if stage == 'long' else reference/'inputs.json')
        if frozen != reference_inputs:
            raise ValueError('stored token inputs differ from the frozen reference')
        expected_ids = (['code-cache'] if stage == 'mechanism' else list(frozen) if stage == 'long'
                        else [key for key in frozen if key != 'calibration-explanation'])
        if sorted(r['id'] for r in raw) != sorted(['warmup', *expected_ids]) or (
                [r['id'] for r in raw] != read(path/'case-order.json')):
            raise ValueError('request case coverage/order differs from the frozen inputs')
        for request in raw:
            audit_acceptance(native_bytes, request)
            key = request['id']
            warmup = key == 'warmup'
            if request['warmup'] != warmup:
                raise ValueError('warmup classification differs from case identity')
            tokens = (read(path/'http-warmup-tokens.body')['tokens'] if warmup and stage == 'long'
                      else frozen['calibration-explanation' if warmup else key]['tokens'])
            cap = 32 if warmup or stage == 'mechanism' else 128 if stage == 'long' else 256
            payload = json.loads(read(path/f'http-completion-{key}.request.json')['body_utf8'])
            if payload != stock.completion_payload(tokens, cap) or request['prompt_tokens'] != len(tokens):
                raise ValueError('raw completion request differs from the declared tokens/decoding')
            raw_body = path/f'http-completion-{key}.body'
            http = read(path/f'http-completion-{key}.http.json')
            if http['status'] != 200 or not http['complete'] or http['bytes'] != raw_body.stat().st_size:
                raise ValueError('incomplete completion HTTP receipt')
            if read(raw_body) != request['response']:
                raise ValueError('compact response differs from raw HTTP body')
        native_kv = re.findall(r'size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers[^\n]*?'
                               r'K \(([^)]+)\):[^\n]*?V \(([^)]+)\):', native_log)
        expected_layers = [64, 24] if cfg['k'] else [64]
        if [int(v[2]) for v in native_kv] != expected_layers or any(
            int(v[1]) != cfg['context'] or v[3] != cfg['kv'] or v[4] != cfg['kv'] for v in native_kv):
            raise ValueError(f'effective native KV context/type mismatch: {path.name}')
        stock.validate_placement(native_log, cfg['ngl'], 'draft05' if cfg['k'] else None)
        input_hash = stock.digest(path/'inputs.json')
        if stage != 'long':
            # Mechanism and short retain the same complete token file, even where caps differ.
            if expected_short_inputs is None:
                expected_short_inputs = input_hash
            if input_hash != expected_short_inputs or manifest['input_sha256'] != study.SHORT_INPUT_SHA256:
                raise ValueError('short inputs changed')
        elif input_hash != long_input_hash or (not (condition == 'target' and repeat == 1)
                                              and manifest['input_sha256'] != long_input_hash):
            raise ValueError('long input file changed')
        data = [r for r in raw if not r['warmup']]
        expected_count = {'mechanism': 1, 'short': 6, 'long': 2}[stage]
        if len(data) != expected_count or len(raw) != expected_count+1:
            raise ValueError('incomplete request coverage')
        cases = []
        for r in data:
            response = r['response']
            if len(response['tokens']) != response['tokens_predicted']:
                raise ValueError('emitted ID count differs from native counter')
            events = r['acceptance']
            if stock.acceptance_summary(events, response['timings']) != r['acceptance_summary']:
                raise ValueError('stored acceptance summary does not reproduce')
            if not r['acceptance_summary']['consistent']:
                raise ValueError('acceptance counters do not reconcile')
            reference_ids = long_reference[r['id']] if stage == 'long' else originals[r['id']]
            # Mechanism uses a 32-token cap, so only compare the common prefix there.
            compared = reference_ids[:len(response['tokens'])] if stage == 'mechanism' else reference_ids
            difference = first_difference(response['tokens'], compared)
            prior = repeated.get((stage, condition, r['id']))
            repeated[(stage, condition, r['id'])] = response['tokens']
            cases.append({'id': r['id'], 'prompt_tokens': r['prompt_tokens'],
                          'emitted_tokens': len(response['tokens']), 'seconds': r['seconds'],
                          'native_timings': response['timings'], 'stop_type': response['stop_type'],
                          'first_reference_difference': difference,
                          'repeat_matches': None if prior is None else prior == response['tokens'],
                          'cycles': r['acceptance_summary']['cycles'], 'response_sha256': stock.digest(path/f'response-{r["id"]}.json')})
        result = {'group': path.name, 'head': manifest['head'], 'stage': stage, 'condition': condition,
                  'repeat': repeat, 'configuration': cfg, 'resources': audit, 'summary': summarize_requests(data),
                  'native_kv': [{'total_mib': float(v[0]), 'cells': int(v[1]), 'layers': int(v[2]),
                                 'type_k': v[3], 'type_v': v[4]} for v in native_kv],
                  'allocation_lines': [line for line in native_log.splitlines()
                                       if 'buffer size' in line or 'offloaded' in line],
                  'cases': cases, 'manifest_sha256': stock.digest(path/'manifest.json'),
                  'raw_rows_sha256': stock.digest(path/'rows.jsonl')}
        group_records.append(result)
        grouped[(stage, condition)].extend(data)
        if stage == 'mechanism':
            mechanisms[condition] = mechanism((path/'server.log').read_bytes(), data[0])
    conditions = {f'{stage}/{condition}': summarize_requests(data) for (stage, condition), data in grouped.items()}
    expected = {f'mechanism-r1-{c}' for c in ('cpu-k16', 'offload-k16', 'disabled-k16')}
    expected |= {f'short-r{r}-{c}' for r in (1, 2) for c in study.CONDITIONS if c != 'target'}
    expected |= {f'long-r{r}-{c}' for r in (1, 2) for c in ('target', 'cpu-k4', 'cpu-k16', 'offload-k16')}
    observed = {r['group'] for r in group_records} | {r['group'] for r in failures}
    return {'schema': 1, 'groups': group_records, 'failures': failures, 'conditions': conditions,
            'coverage': {'expected_groups': len(expected), 'successful_groups': len(group_records),
                         'failed_groups': len(failures), 'not_run': sorted(expected-observed),
                         'unexpected': sorted(observed-expected)},
            'mechanism': mechanisms, 'long_input_sha256': long_input_hash,
            'limitations': ['Graph assignments are not physical byte measurements.',
                            'Instrumented diagnostics are excluded from timing rankings.',
                            'Native decode timings do not separately time target verification.',
                            'Counter consistency does not independently establish argmax or KV correctness.',
                            'Two repeats and two populated long prompts do not establish broad workload performance.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.runs, args.reference)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()

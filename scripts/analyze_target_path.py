"""Compare encoded target distributions on aligned prefixes, separately by stratum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stock_benchmark as stock
import target_path_diagnostic as diagnostic
from analyze_verification_offload import audit_trace, read, rows


def compare(left, right, left_token, right_token):
    p = np.exp(left.astype(np.float64))
    q = np.exp(right.astype(np.float64))
    top = np.sort(p)[-2:]
    margin = float(top[-1]-top[-2])
    distance = float(np.max(np.abs(p-q)))
    flip = left_token != right_token
    return {'token_flip': flip, 'reference_token': left_token, 'candidate_token': right_token,
            'probability_linf': distance, 'reference_probability_margin': margin,
            'margin_exceeds_twice_distance': margin > 2*distance,
            'argmax_bound_violation': flip and margin > 2*distance,
            'reference_top_two_logprob_gap': float(np.sort(left)[-1])-float(np.sort(left)[-2]),
            'candidate_top_two_logprob_gap': float(np.sort(right)[-1])-float(np.sort(right)[-2]),
            'raw_logit_linf': None,
            'representation': 'Reconstructed probabilities from float32 serialized log-softmax; not raw logits.'}


def analyze(run_root, fixture_path):
    fixture = read(fixture_path)
    vectors, summaries, groups, failures = {}, {}, {}, []
    for factor in diagnostic.FACTORS:
        path = run_root/factor
        if not path.exists():
            failures.append({'factor': factor, 'reason': 'not run'})
            continue
        if (path/'failure.json').exists() or not (path/'completion.json').exists():
            failures.append({'factor': factor, 'failure': read(path/'failure.json') if (path/'failure.json').exists() else None})
            continue
        manifest = read(path/'manifest.json')
        if manifest['factor'] != factor or manifest['fixture_sha256'] != stock.digest(fixture_path):
            raise ValueError('factor/fixture mismatch')
        audit = audit_trace(path/'resources.jsonl')
        if not audit['pass'] or not read(path/'completion.json')['complete']:
            raise ValueError('completed numerical group has failing resources')
        records = rows(path/'cases.jsonl')
        if {r['case'] for r in records} != set(fixture['cases']) or len(records) != len(fixture['cases']):
            raise ValueError('final case coverage mismatch')
        requests = rows(path/'requests.jsonl')
        expected = []
        for key, case in fixture['cases'].items():
            lengths = (range(case['original_prompt_tokens'], len(case['tokens'])+1)
                       if diagnostic.FACTORS[factor][3] else [len(case['tokens'])])
            expected.extend((key, length, step) for step, length in enumerate(lengths))
        if [(r['case'], r['prefix_length'], r['step']) for r in requests] != expected:
            raise ValueError('request/prefix sequence coverage mismatch')
        for r in requests:
            key, length, step = r['case'], r['prefix_length'], r['step']
            case = fixture['cases'][key]
            final = length == len(case['tokens'])
            label = f'{key}-{length}'
            request = read(path/f'http-{label}.request.json')
            payload = json.loads(request['body_utf8'])
            expected_payload = stock.completion_payload(case['tokens'][:length], 1)
            expected_payload.update(cache_prompt=step > 0, n_probs=diagnostic.VOCAB if final else 0,
                                    post_sampling_probs=False)
            raw_path = path/f'http-{label}.body'
            http = read(path/f'http-{label}.http.json')
            if payload != expected_payload or stock.digest(raw_path) != r['response_sha256']:
                raise ValueError('raw request/response differs from the frozen prefix sequence')
            if http['status'] != 200 or not http['complete'] or http['bytes'] != raw_path.stat().st_size:
                raise ValueError('incomplete intermediate HTTP receipt')
            response = read(raw_path)
            cache, evaluated = response['timings']['cache_n'], response['timings']['prompt_n']
            expected_cache, expected_evaluated = (length-1, 1) if step > 0 else (0, length)
            if (cache, evaluated) != (expected_cache, expected_evaluated) or (
                r['native_cache_n'], r['native_prompt_n']) != (cache, evaluated) or not r['incremental_cache_accounting_pass']:
                raise ValueError('native cache accounting does not reproduce')
        for r in records:
            case = r['case']
            file = path/f'{case}-logprobs.npy'
            if stock.digest(file) != r['vector_sha256']:
                raise ValueError('distribution vector changed')
            raw = path/f'http-{case}-{len(fixture["cases"][case]["tokens"])}.body'
            if stock.digest(raw) != r['raw_body_sha256']:
                raise ValueError('raw probability response changed')
            from_raw, raw_summary = diagnostic.distribution(read(raw))
            value = np.load(file, allow_pickle=False)
            if not np.array_equal(value, from_raw) or raw_summary['token'] != r['token']:
                raise ValueError('compact probability evidence does not reproduce')
            vectors[factor, case] = value
            summaries[factor, case] = r
        groups[factor] = {'resources': audit, 'request_count': len(requests),
                          'final_case_count': len(records),
                          'historical_next_matches': sum(r['token'] == r['historical_next'] for r in records),
                          'encoded_argmax_matches': sum(r['encoded_argmax_matches'] for r in records),
                          'cases': records, 'manifest_sha256': stock.digest(path/'manifest.json')}
    comparisons = []
    for reference, candidate in [('incremental-reference', 'threads'), ('incremental-reference', 'placement'),
                                 ('incremental-reference', 'prefill'), ('prefill', 'microbatch')]:
        if reference not in groups or candidate not in groups:
            continue
        for key, case in fixture['cases'].items():
            result = compare(vectors[reference, key], vectors[candidate, key],
                             summaries[reference, key]['token'], summaries[candidate, key]['token'])
            comparisons.append({'reference': reference, 'candidate': candidate,
                                'case': key, 'stratum': case['stratum'], **result})
    counts = []
    for reference, candidate in dict.fromkeys((r['reference'], r['candidate']) for r in comparisons):
        for stratum in ('selected historical divergence', 'fixed ordinary position'):
            selected = [r for r in comparisons if (r['reference'], r['candidate'], r['stratum']) ==
                        (reference, candidate, stratum)]
            counts.append({'reference': reference, 'candidate': candidate, 'stratum': stratum,
                           'positions': len(selected), 'flips': sum(r['token_flip'] for r in selected),
                           'max_probability_linf': max(r['probability_linf'] for r in selected)})
    return {'fixture_sha256': stock.digest(fixture_path), 'groups': groups, 'failures': failures,
            'comparisons': comparisons, 'stratum_counts': counts,
            'limits': ['No population flip rate follows from eight selected positions.',
                       'Target-only cache accounting does not prove speculative KV/mask correctness.',
                       'All distributions are encoded log-softmax; raw logit infinity distance is unavailable.',
                       'The original v0.14 token-identity failures remain unchanged.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(analyze(args.runs, args.fixture), indent=2)+'\n')


if __name__ == '__main__':
    main()

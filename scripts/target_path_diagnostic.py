"""Bounded target-only numerical factors and incrementally retained prefix state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error

import numpy as np
import stock_benchmark as stock
import verification_offload as study

FACTORS = {'prefill': (45, 16, 256, False), 'microbatch': (45, 16, 1, False),
           'incremental-reference': (45, 16, 256, True), 'threads': (45, 24, 256, True),
           'placement': (44, 16, 256, True)}
VOCAB = 152064


def distribution(response):
    """Preserve encoded log-softmax values, explicitly not raw model logits."""
    entries = response['completion_probabilities'][0]['top_logprobs']
    ids = np.array([x['id'] for x in entries], dtype=np.int64)
    if len(ids) != VOCAB or not np.array_equal(np.sort(ids), np.arange(VOCAB)):
        raise ValueError('full-vocabulary probability coverage is missing or duplicated')
    values = np.array([x['logprob'] for x in entries], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite encoded log probabilities')
    vector = np.empty(VOCAB, dtype=np.float32)
    vector[ids] = values
    ordered = np.argsort(-vector, kind='stable')
    actual = response['tokens']
    if len(actual) != 1 or response['tokens_predicted'] != 1:
        raise ValueError('one-token diagnostic returned an unexpected count')
    maximum = float(vector[ordered[0]])
    return vector, {'token': actual[0], 'top_two_ids': ordered[:2].tolist(),
                    'top_two_logprob_gap': float(vector[ordered[0]])-float(vector[ordered[1]]),
                    'encoded_argmax_matches': float(vector[actual[0]]) == maximum,
                    'zero_probability_sentinels': int(np.sum(vector == np.finfo(np.float32).min)),
                    'probability_sum': float(np.exp(vector.astype(np.float64)).sum()),
                    'representation': 'float32 HTTP log-softmax; -FLT_MAX denotes zero probability. '
                                      'Top-two gaps approximate raw logit gaps, but absolute raw logits are unavailable.'}


def execute(args):
    root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root, encoding='utf-8').strip()
    if git('status', '--porcelain'):
        raise ValueError('diagnostic requires clean committed source')
    head = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/experiment/verification-offload'],
                   cwd=root, check=True, capture_output=True)
    args.models, args.binary = args.models.resolve(), args.binary.resolve()
    fixture_path = root/'data/target-path-diagnostic.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    catalog_path = root/'configs/stock-speculation-artifacts.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    checked = stock.verify_catalog(catalog, args.models, args.binary, ['target'])
    ngl, threads, ubatch, incremental = FACTORS[args.factor]
    cmd = stock.server_command(args.binary/'llama-server.exe', checked['target'][0], None, ngl, threads, 16, args.port)
    cmd[cmd.index('-ub')+1] = str(ubatch)
    environment, effective, removed = study.child_environment(os.environ, 32, 0)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stock.write_json(out/'manifest.json', {'head': head, 'factor': args.factor, 'command': cmd,
                     'effective_runtime_environment': effective, 'removed_override_names': removed,
                     'fixture_sha256': stock.digest(fixture_path), 'catalog_sha256': stock.digest(catalog_path),
                     'runner_sha256': stock.digest(Path(__file__)), 'numpy': np.__version__,
                     'artifacts': checked, 'sampling': 'greedy; no draft; full pre-sampling probability output at final positions'})
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', args.port))
        base = f'http://127.0.0.1:{args.port}'
        with (out/'server.log').open('xb') as log, stock.managed_process(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=args.binary) as proc:
            with stock.Resources(proc.pid, out/'resources.jsonl') as resources:
                start = time.perf_counter()
                for attempt in range(2400):
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}')
                    try:
                        if stock.request(base, '/health', timeout=2,
                                         receipt=out/f'http-health-{attempt}').get('status') == 'ok':
                            break
                    except (OSError, urllib.error.HTTPError):
                        pass
                    if time.perf_counter()-start > 600:
                        raise TimeoutError('diagnostic startup')
                    time.sleep(0.25)
                placement = stock.validate_placement((out/'server.log').read_text(encoding='utf-8', errors='replace'), ngl, None)
                state = study.resource_check(resources)
                stock.write_json(out/'startup.json', {'placement': placement, 'resources': state,
                                 'props': stock.request(base, '/props', receipt=out/'http-props')})
                if not state['pass']:
                    raise RuntimeError('startup resource check failed')
                with (out/'requests.jsonl').open('x', encoding='utf-8') as ledger, \
                     (out/'cases.jsonl').open('x', encoding='utf-8') as compact:
                    for key, case in fixture['cases'].items():
                        tokens = case['tokens']
                        lengths = (range(case['original_prompt_tokens'], len(tokens)+1)
                                   if incremental else [len(tokens)])
                        for step, length in enumerate(lengths):
                            final = length == len(tokens)
                            payload = stock.completion_payload(tokens[:length], 1)
                            payload.update(cache_prompt=step > 0, n_probs=VOCAB if final else 0,
                                           post_sampling_probs=False)
                            label = f'{key}-{length}'
                            response = stock.request(base, '/completion', payload, timeout=600,
                                                     receipt=out/f'http-{label}')
                            cache = response['timings']['cache_n']
                            evaluated = response['timings']['prompt_n']
                            cache_pass = (cache == length-1 and evaluated == 1) if step > 0 else cache == 0
                            state = study.resource_check(resources)
                            row = {'case': key, 'prefix_length': length, 'step': step, 'final': final,
                                   'native_cache_n': cache, 'native_prompt_n': evaluated,
                                   'incremental_cache_accounting_pass': cache_pass,
                                   'response_sha256': stock.digest(out/f'http-{label}.body'),
                                   'resources': state}
                            ledger.write(json.dumps(row)+'\n'); ledger.flush()
                            if not cache_pass or not state['pass']:
                                raise RuntimeError('prefix-cache accounting or resource check failed; request retained')
                            if final:
                                vector, summary = distribution(response)
                                vector_path = out/f'{key}-logprobs.npy'
                                np.save(vector_path, vector)
                                summary.update(case=key, stratum=case['stratum'],
                                               historical_next=case['historical_next'],
                                               vector_sha256=stock.digest(vector_path),
                                               raw_body_sha256=row['response_sha256'])
                                compact.write(json.dumps(summary)+'\n'); compact.flush()
                                print(json.dumps({'factor': args.factor, **summary}), flush=True)
                state = study.resource_check(resources)
                stock.write_json(out/'completion.json', {'complete': state['pass'], 'resources': state})
                if not state['pass']:
                    raise RuntimeError('final resource check failed')
    except BaseException as exc:
        stock.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'binary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--factor', choices=list(FACTORS), required=True)
    parser.add_argument('--port', type=int, default=8103)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()

"""Fresh-process operation-offload experiment on the pinned stock server."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import urllib.error

import stock_benchmark as stock

SHORT_INPUT_SHA256 = '184a3734aabc39f524820ec8f63a9b2fd097a5cb5c56c2f1706a304cef345829'
CONDITIONS = {
    'cpu-k4': (32, 4, True), 'cpu-k8': (32, 8, True),
    'cpu-k16': (32, 16, True), 'offload-k8': (8, 8, True),
    'offload-k16': (8, 16, True), 'disabled-k16': (8, 16, False),
    'target': (32, None, True),
}


def child_environment(parent, threshold, scheduler_debug):
    if threshold not in (8, 32) or scheduler_debug not in (0, 2):
        raise ValueError('unregistered runtime setting')
    removed = sorted(k for k in parent if k.upper().startswith(('LLAMA_', 'GGML_', 'CUDA_')))
    environment = {k: v for k, v in parent.items() if k not in removed}
    effective = {'LLAMA_TRACE': '1', 'GGML_OP_OFFLOAD_MIN_BATCH': str(threshold),
                 'GGML_SCHED_DEBUG': str(scheduler_debug)}
    environment.update(effective)
    return environment, effective, removed


def configuration(stage, condition):
    threshold, k, offload = CONDITIONS[condition]
    allowed = {
        'mechanism': {'cpu-k16', 'offload-k16', 'disabled-k16'},
        'short': set(CONDITIONS) - {'target'},
        'long': {'target', 'cpu-k4', 'cpu-k16', 'offload-k16'},
    }
    if condition not in allowed[stage]:
        raise ValueError('condition is outside the prospective stage matrix')
    return {'threshold': threshold, 'k': k, 'op_offload': offload,
            'context': 18432 if stage == 'long' else 4096,
            'kv': 'q8_0' if stage == 'long' else 'f16',
            'ngl': 38 if stage == 'long' else 44, 'threads': 24,
            'scheduler_debug': 2 if stage == 'mechanism' else 0}


def command(binary, checked, cfg, port):
    result = stock.server_command(binary/'llama-server.exe', checked['target'][0],
                                  checked['draft05'][0] if cfg['k'] else None,
                                  cfg['ngl'], cfg['threads'], cfg['k'] or 16, port)
    replacements = {'-c': cfg['context'], '-ctk': cfg['kv'], '-ctv': cfg['kv'],
                    '--spec-draft-type-k': cfg['kv'], '--spec-draft-type-v': cfg['kv']}
    for flag, value in replacements.items():
        if flag in result:
            result[result.index(flag)+1] = str(value)
    if cfg['scheduler_debug']:
        result[result.index('--verbosity')+1] = '6'
    if not cfg['op_offload']:
        result.append('--no-op-offload')
    return result


def scheduler_summary(text):
    """Assignments are graph decisions, not executions or physical byte counts."""
    nodes = []
    for line in text.splitlines():
        match = re.search(r'node\s+#\s*(\d+)\s+\(([^)]+)\):\s*(.*?)\s+\[([^\]]+)\]', line)
        if match:
            nodes.append({'node': int(match[1]), 'operation': match[2].strip(),
                          'tensor': match[3].strip(), 'assignment': match[4].strip(),
                          'raw': line})
    return {'assignment_count': len(nodes),
            'mul_mat_assignments_by_backend': dict(Counter(n['assignment'].split()[0] for n in nodes
                                                          if n['operation'] == 'MUL_MAT')),
            'host_weight_offload_assignments': None,
            'reason_tags_available': False,
            'nodes': nodes,
            'limitation': 'Graph assignments may repeat or be reused; tensor sizes in text are rounded. '
                          'These records are not executed-operation counts or measured transfer bytes. '
                          'b10919 compiles out assignment reason tags, including 1.off; backend/source '
                          'records must be compared across controls instead.'}


def resource_check(resources):
    rows = list(resources.rows)
    return {'pass': bool(rows) and not resources.errors and all(
                x['gpu']['used'] <= 15000*2**20 and x['host']['available'] >= 2*2**30 for x in rows),
            'samples': len(rows), 'errors': list(resources.errors),
            'gpu_peak': max((x['gpu']['used'] for x in rows), default=None),
            'host_available_min': min((x['host']['available'] for x in rows), default=None)}


def long_inputs(base, out, workloads):
    """Construct exact 16K prefixes by trimming only authored middle context tokens."""
    result = {}
    marker = 'UNIQUE_OFFLOAD_CONTEXT_INSERTION_POINT'
    filler = ''.join(
        f'Record {i:05d}: component=worker-{i % 17:02d}; received={i % 31}; '
        f'processed={i % 29}; copied_label=ITEM-{i:05d}; '
        'review retries, preserve original records, and reconcile daily totals.\n'
        for i in range(1500))
    filler_tokens = stock.request(base, '/tokenize', {'content': filler, 'add_special': False,
                                  'parse_special': False}, receipt=out/'http-long-filler')['tokens']
    for case in workloads['sustained']:
        if case['id'] not in ('code-cache', 'data-audit'):
            continue
        content = ('The following synthetic context is background for an engineering notebook.\n'
                   + marker + '\nNow switch to the current request.\n' + case['text'])
        formatted = stock.request(base, '/apply-template', {'messages': [
            {'role': 'system', 'content': workloads['system']}, {'role': 'user', 'content': content}]},
            receipt=out/f'http-long-template-{case["id"]}')['prompt']
        if formatted.count(marker) != 1:
            raise ValueError('long context marker is not unique')
        first, last = formatted.split(marker)
        parts = []
        for index, text in enumerate((first, last)):
            parts.append(stock.request(base, '/tokenize', {'content': text, 'add_special': index == 0,
                         'parse_special': True}, receipt=out/f'http-long-part-{case["id"]}-{index}')['tokens'])
        needed = 16384 - len(parts[0]) - len(parts[1])
        if not 0 < needed <= len(filler_tokens):
            raise ValueError('long context construction does not fit its fixed token budget')
        tokens = parts[0] + filler_tokens[:needed] + parts[1]
        result[case['id']] = {'tokens': tokens, 'max_tokens': 128, 'domain': case['domain'],
                             'construction': {'header_tokens': len(parts[0]), 'middle_tokens': needed,
                                              'suffix_tokens': len(parts[1]),
                                              'filler_sha256': stock.digest(out/'http-long-filler.body')}}
    stock.write_json(out/'long-inputs.json', result)
    return result


def _run(args):
    root = Path(__file__).resolve().parents[1]
    cfg = configuration(args.stage, args.condition)
    if args.repeat not in (1, 2):
        raise ValueError('only two prospective repeats')
    if args.stage == 'mechanism' and args.repeat != 1:
        raise ValueError('mechanism diagnostics have one repeat')
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root, encoding='utf-8').strip()
    if git('status', '--porcelain'):
        raise ValueError('measurement requires clean committed source')
    head = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/experiment/verification-offload'],
                   cwd=root, check=True, capture_output=True)
    args.binary, args.models = args.binary.resolve(), args.models.resolve()
    catalog_path = root/'configs/stock-speculation-artifacts.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    checked = stock.verify_catalog(catalog, args.models, args.binary,
                                   ['target'] + (['draft05'] if cfg['k'] else []))
    workloads = json.loads((root/'data/stock-speculation-workloads.json').read_text(encoding='utf-8'))
    if args.stage != 'long':
        if stock.digest(args.inputs) != SHORT_INPUT_SHA256:
            raise ValueError('short inputs differ from the frozen v0.14 input file')
        frozen = json.loads(args.inputs.read_text(encoding='utf-8'))
        stock.validate_inputs(frozen, [workloads['calibration'], *workloads['sustained']])
    elif args.inputs:
        frozen = json.loads(args.inputs.read_text(encoding='utf-8'))
        if set(frozen) != {'code-cache', 'data-audit'} or any(
            len(v['tokens']) != 16384 or v['max_tokens'] != 128 or
            not all(type(t) is int and 0 <= t < 152064 for t in v['tokens']) for v in frozen.values()):
            raise ValueError('invalid frozen long inputs')
    elif args.condition != 'target' or args.repeat != 1:
        raise ValueError('only the first long target control may establish long inputs')
    else:
        frozen = None
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cmd = command(args.binary, checked, cfg, args.port)
    environment, effective, removed = child_environment(os.environ, cfg['threshold'], cfg['scheduler_debug'])
    stock.write_json(out/'manifest.json', {'head': head, 'stage': args.stage, 'condition': args.condition,
                     'repeat': args.repeat, 'configuration': cfg, 'command': cmd,
                     'effective_runtime_environment': effective, 'removed_override_names': removed,
                     'artifacts': checked, 'catalog_sha256': stock.digest(catalog_path),
                     'input_sha256': stock.digest(args.inputs) if args.inputs else None,
                     'runner_sha256': stock.digest(Path(__file__)),
                     'stock_runner_sha256': stock.digest(root/'scripts/stock_benchmark.py'),
                     'protocol_sha256': stock.digest(root/'docs/verification-offload-protocol.md'),
                     'python': os.sys.version})
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', args.port))
    base = f'http://127.0.0.1:{args.port}'
    with (out/'server.log').open('xb') as log, stock.managed_process(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=args.binary) as proc:
        try:
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
                        raise TimeoutError('startup exceeded 600 seconds')
                    time.sleep(0.25)
                placement = stock.validate_placement((out/'server.log').read_text(encoding='utf-8', errors='replace'),
                                                      cfg['ngl'], 'draft05' if cfg['k'] else None)
                startup = resource_check(resources)
                stock.write_json(out/'startup.json', {'seconds': time.perf_counter()-start,
                                 'placement': placement, 'resources': startup,
                                 'props': stock.request(base, '/props', receipt=out/'http-props')})
                if not startup['pass']:
                    raise RuntimeError('startup resource/telemetry bound failed')
                if frozen is None:
                    frozen = long_inputs(base, out, workloads)
                stock.write_json(out/'inputs.json', frozen)
                if args.stage == 'long':
                    # Short warmup is generated from the same documented template, outside timing.
                    cal = workloads['calibration']
                    text = stock.request(base, '/apply-template', {'messages': [
                        {'role': 'system', 'content': workloads['system']}, {'role': 'user', 'content': cal['text']}]},
                        receipt=out/'http-warmup-template')['prompt']
                    tokens = stock.request(base, '/tokenize', {'content': text, 'add_special': True,
                                           'parse_special': True}, receipt=out/'http-warmup-tokens')['tokens']
                    warmup = {'tokens': tokens, 'max_tokens': 32, 'domain': 'calibration'}
                    cases = [(k, v) for k, v in frozen.items()]
                else:
                    warmup = dict(frozen['calibration-explanation'], max_tokens=32, domain='calibration')
                    wanted = ['code-cache'] if args.stage == 'mechanism' else [c['id'] for c in workloads['sustained']]
                    cases = [(k, dict(frozen[k], domain=next(c['domain'] for c in workloads['sustained'] if c['id'] == k)))
                             for k in wanted]
                    if args.stage == 'mechanism':
                        cases = [(k, dict(v, max_tokens=32)) for k, v in cases]
                random.Random(20260913 + args.repeat).shuffle(cases)
                cases = [('warmup', warmup), *cases]
                stock.write_json(out/'case-order.json', [k for k, _ in cases])
                with (out/'rows.jsonl').open('x', encoding='utf-8') as ledger:
                    for key, case in cases:
                        if len(case['tokens']) + case['max_tokens'] + (cfg['k'] or 0) > cfg['context']:
                            raise ValueError('prompt/output/speculation exceeds context capacity')
                        payload = stock.completion_payload(case['tokens'], case['max_tokens'])
                        stock.write_json(out/f'request-{key}.json', payload)
                        log_start = (out/'server.log').stat().st_size
                        begin = time.perf_counter()
                        response = stock.request(base, '/completion', payload, timeout=1800,
                                                  receipt=out/f'http-completion-{key}')
                        end = time.perf_counter()
                        stock.write_json(out/f'response-{key}.json', response)
                        time.sleep(0.05)
                        with (out/'server.log').open('rb') as source:
                            source.seek(log_start)
                            segment = source.read().decode('utf-8', errors='replace')
                        # At debug verbosity the stock server also repeats acceptance in a
                        # "new n_tokens" message. Preserve raw logs but count only its canonical event.
                        events = stock.parse_acceptance('\n'.join(line for line in segment.splitlines()
                                                                 if 'new n_tokens =' not in line))
                        state = resource_check(resources)
                        row = {'id': key, 'warmup': key == 'warmup', 'domain': case['domain'],
                               'prompt_tokens': len(case['tokens']), 'begin': begin, 'end': end,
                               'seconds': end-begin, 'log_start': log_start,
                               'log_end': (out/'server.log').stat().st_size,
                               'response': response, 'acceptance': events,
                               'acceptance_summary': stock.acceptance_summary(events, response.get('timings', {})),
                               'resources': state}
                        ledger.write(json.dumps(row, ensure_ascii=False)+'\n')
                        ledger.flush()
                        if cfg['scheduler_debug']:
                            stock.write_json(out/f'scheduler-{key}.json', scheduler_summary(segment))
                        print(json.dumps({'output': str(out), 'case': key, 'seconds': end-begin,
                                          'tokens': len(response.get('tokens', [])), 'resources': state['pass']}), flush=True)
                        if not state['pass']:
                            raise RuntimeError('resource/telemetry bound failed; raw response retained')
                final = resource_check(resources)
                stock.write_json(out/'completion.json', {'complete': final['pass'], 'resources': final})
                if not final['pass']:
                    raise RuntimeError('final resource check failed')
        except BaseException as exc:
            stock.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise


def run(args):
    """Retain setup failures too, without modifying any earlier attempt directory."""
    existed = args.output.exists()
    try:
        return _run(args)
    except BaseException as exc:
        if not existed and args.output.is_dir() and not (args.output/'failure.json').exists():
            stock.write_json(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'binary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--stage', choices=['mechanism', 'short', 'long'], required=True)
    parser.add_argument('--condition', choices=list(CONDITIONS), required=True)
    parser.add_argument('--repeat', type=int, choices=[1, 2], default=1)
    parser.add_argument('--port', type=int, default=8101)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

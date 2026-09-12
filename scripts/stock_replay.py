"""Untimed one-token replays of first divergent decisions on identical prefixes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import stock_benchmark as bench


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_resources(resources):
    if (not resources.rows or resources.errors or any(
            x['gpu']['used'] > 15000*2**20 or x['host']['available'] < 2*2**30
            for x in resources.rows)):
        raise RuntimeError('diagnostic resource bound or telemetry failure')


def validate_analysis_receipts(analysis, evaluation):
    if not analysis.get('evidence_sha256'):
        raise ValueError('analysis has no bound measurement receipts')
    for relative, expected in analysis['evidence_sha256'].items():
        file=(evaluation/relative).resolve()
        if not file.is_relative_to(evaluation.resolve()):raise ValueError('analysis receipt path is outside evaluation')
        actual=bench.digest(file) if file.exists() else None
        if actual!=expected:raise ValueError('measurement receipt changed since analysis')


def replay_plan(analysis, selection):
    target = selection['selected']['target']
    groups = {}
    for item in analysis['first_divergence_diagnostics']:
        if (not item['tokens'] or len(item['tokens']) >= 4096
                or not all(type(t) is int and t >= 0 for t in item['tokens'])):
            raise ValueError('invalid common-prefix tokens')
        candidate = item['configuration']
        configurations = [
            {'draft': None, 'ngl': target['ngl'], 'threads': target['threads'], 'k': 16},
            {'draft': None, 'ngl': candidate['ngl'], 'threads': candidate['threads'], 'k': 16},
            {key: candidate[key] for key in ['draft', 'ngl', 'threads', 'k']},
        ]
        for config in configurations:
            label = f'{config["draft"] or "target"}-ngl{config["ngl"]}-t{config["threads"]}-k{config["k"]}'
            group = groups.setdefault(label, {'configuration': config, 'cases': {}})
            key = bench.hashlib.sha256(json.dumps(item['tokens']).encode('utf-8')).hexdigest()
            case = group['cases'].setdefault(key, {'tokens': item['tokens'], 'origins': []})
            origin = {'run': item['run'], 'id': item['id'], 'first_difference': item['first_difference']}
            if origin not in case['origins']:
                case['origins'].append(origin)
    return groups


def replay_ports(groups, first):
    if not 1 <= first <= 65535 or first+len(groups)-1 > 65535:
        raise ValueError('diagnostic port range must stay within 1..65535')
    return {label: first+index for index, label in enumerate(groups)}


def check_port(port, out):
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', port))
    except OSError as exc:
        bench.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                              'phase': 'port-probe', 'port': port})
        raise


def execute(group, args, out, checked, head, port):
    out.mkdir(exist_ok=False)
    config = group['configuration']
    draft = config['draft']
    command = bench.server_command(args.binary/'llama-server.exe', checked['target'][0],
                                    checked[draft][0] if draft else None,
                                    config['ngl'], config['threads'], config['k'], port)
    bench.write_json(out/'manifest.json', {'head': head, 'configuration': config, 'command': command,
                                          'analysis_sha256': bench.digest(args.analysis),
                                          'measurement_head': read(args.analysis)['source'],
                                          'catalog_sha256': bench.digest(Path(__file__).resolve().parents[1]/'configs/stock-speculation-artifacts.json'),
                                          'mode': 'untimed-common-prefix-replay', 'cases': group['cases']})
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'CUDA_'))}
    environment['LLAMA_TRACE'] = '1'
    base = f'http://127.0.0.1:{port}'
    check_port(port, out)
    begin = time.perf_counter()
    with (out/'server.log').open('xb') as log, bench.managed_process(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=args.binary) as proc:
        try:
            with bench.Resources(proc.pid, out/'resources.jsonl') as resources:
                ready = False
                attempt = 0
                while time.perf_counter()-begin < 600:
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}')
                    try:
                        ready = bench.request(base, '/health', timeout=2, receipt=out/f'health-{attempt}').get('status') == 'ok'
                    except OSError:
                        pass
                    if ready:
                        break
                    attempt += 1
                    time.sleep(0.25)
                if not ready:
                    raise TimeoutError('diagnostic startup')
                time.sleep(0.05)
                bench.validate_placement((out/'server.log').read_text(encoding='utf-8', errors='replace'), config['ngl'], draft)
                check_resources(resources)
                with (out/'rows.jsonl').open('x', encoding='utf-8') as rows:
                    for key, case in group['cases'].items():
                        # This rebuilds target state on a common prefix. A one-token cap
                        # cannot reproduce the original verification-batch kernel shape.
                        payload = bench.completion_payload(case['tokens'], 1)
                        payload['n_probs'] = 10
                        payload['post_sampling_probs'] = False
                        response = bench.request(base, '/completion', payload, receipt=out/f'completion-{key}')
                        bench.write_json(out/f'response-{key}.json', response)
                        rows.write(json.dumps({'prefix_sha256': key, 'request': payload,
                                               'response': response, 'origins': case['origins']}, ensure_ascii=False)+'\n')
                        rows.flush()
                        check_resources(resources)
                bench.write_json(out/'completion.json', {'complete': True, 'cases': len(group['cases']),
                                  'sampled_gpu_peak': max(x['gpu']['used'] for x in resources.rows),
                                  'sampled_host_available_min': min(x['host']['available'] for x in resources.rows)})
        except BaseException as exc:
            bench.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['analysis', 'evaluation', 'models', 'binary', 'output']:
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--port', type=int, default=8099)
    args = parser.parse_args()
    for flag in ['analysis', 'evaluation', 'models', 'binary', 'output']:
        setattr(args, flag, getattr(args, flag).resolve())
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('diagnostic source must be committed and clean')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/experiment/stock-speculation'], cwd=root, check=True)
    analysis = read(args.analysis)
    validate_analysis_receipts(analysis, args.evaluation)
    selection = read(args.evaluation/'frozen-selection.json')
    if analysis['source'] != selection['source'] or analysis['schedule_sha256'] != bench.digest(args.evaluation/'schedule.json'):
        raise ValueError('analysis and measurement provenance differ')
    groups = replay_plan(analysis, selection)
    ports = replay_ports(groups, args.port)
    catalogue = read(root/'configs/stock-speculation-artifacts.json')
    reference_manifest = read(args.evaluation/'reference-sustained/manifest.json')
    if bench.digest(root/'configs/stock-speculation-artifacts.json') != reference_manifest['catalog_sha256']:
        raise ValueError('replay artifacts differ from measured target substrate')
    keys = ['target'] + sorted({g['configuration']['draft'] for g in groups.values() if g['configuration']['draft']})
    checked = bench.verify_catalog(catalogue, args.models, args.binary, keys)
    args.output.mkdir(exist_ok=False)
    bench.write_json(args.output/'plan.json', groups)
    bench.write_json(args.output/'ports.json', ports)
    for label, group in groups.items():
        print('REPLAY', label, len(group['cases']), flush=True)
        execute(group, args, args.output/label, checked, head, ports[label])
    bench.write_json(args.output/'complete.json', {'groups': len(groups)})


if __name__ == '__main__':
    main()

"""Bounded, separately instrumented Nsight capture of stock verification requests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error

import psutil
import stock_benchmark as stock
import verification_offload as study


def _run(args):
    root = Path(__file__).resolve().parents[1]
    cfg = study.configuration('short', args.condition)
    assert args.condition in ('cpu-k16', 'offload-k16')
    args.models, args.binary, args.profiler = args.models.resolve(), args.binary.resolve(), args.profiler.resolve()
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root, encoding='utf-8').strip()
    if git('status', '--porcelain'):
        raise ValueError('profile requires clean committed source')
    head = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/experiment/verification-offload'],
                   cwd=root, check=True, capture_output=True)
    if stock.digest(args.inputs) != study.SHORT_INPUT_SHA256:
        raise ValueError('unregistered inputs')
    frozen = json.loads(args.inputs.read_text(encoding='utf-8'))
    catalog_path = root/'configs/stock-speculation-artifacts.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    checked = stock.verify_catalog(catalog, args.models, args.binary, ['target', 'draft05'])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    session = 'dml-' + args.condition + '-' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    child = study.command(args.binary, checked, cfg, args.port)
    # Profiling is a separate diagnostic. INFO/TRACE logging preserves ordinary
    # acceptance events without scheduler dumps or extra debug acceptance messages.
    command = [str(args.profiler), 'profile', '--sample=none', '--cpuctxsw=none',
               '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--force-overwrite=false',
               '--kill=false', '--wait=primary', '--show-output=true', '--session-new='+session,
               '--output='+str(out/'cuda-trace'), '--export=sqlite', *child]
    environment, effective, removed = study.child_environment(os.environ, cfg['threshold'], 0)
    manifest = {'head': head, 'condition': args.condition, 'configuration': cfg, 'command': command,
                'effective_runtime_environment': effective, 'removed_override_names': removed,
                'catalog_sha256': stock.digest(catalog_path), 'artifacts': checked,
                'profiler': str(args.profiler), 'profiler_sha256': stock.digest(args.profiler),
                'profiler_version': subprocess.check_output([str(args.profiler), '--version'], encoding='utf-8'),
                'runner_sha256': stock.digest(Path(__file__)), 'input_sha256': stock.digest(args.inputs),
                'resource_process_scope': 'Nsight wrapper PID; GPU and host totals cover startup through the last request. '
                                          'Export and shutdown are excluded. '
                                          'Wrapper process IO/memory is not target process IO/memory.',
                'limitation': 'Instrumented request times are excluded from throughput comparisons.'}
    stock.write_json(out/'manifest.json', manifest)
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', args.port))
        base = f'http://127.0.0.1:{args.port}'
        with (out/'profiler-server.log').open('xb') as log, stock.managed_process(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=args.binary) as proc:
            with stock.Resources(proc.pid, out/'resources.jsonl') as resources:
                target = None
                start = time.perf_counter()
                for attempt in range(2400):
                    if proc.poll() is not None:
                        raise RuntimeError(f'profiler exited before readiness: {proc.returncode}')
                    try:
                        if stock.request(base, '/health', timeout=2,
                                         receipt=out/f'http-health-{attempt}').get('status') == 'ok':
                            owned = psutil.Process(proc.pid).children(recursive=True)
                            targets = [p for p in owned if Path(p.exe()).resolve() == (args.binary/'llama-server.exe')]
                            if len(targets) != 1:
                                raise RuntimeError('cannot identify exactly one owned server under profiler')
                            target = targets[0]
                            break
                    except (OSError, urllib.error.HTTPError):
                        pass
                    if time.perf_counter()-start > 600:
                        raise TimeoutError('profile startup')
                    time.sleep(0.25)
                if target is None:
                    raise RuntimeError('no owned target reached readiness')
                state = study.resource_check(resources)
                stock.write_json(out/'startup.json', {'target_pid': target.pid, 'target_created': target.create_time(),
                                 'resources': state, 'props': stock.request(base, '/props', receipt=out/'http-props')})
                if not state['pass']:
                    raise RuntimeError('profile startup resource bound failed')
                with (out/'rows.jsonl').open('x', encoding='utf-8') as ledger:
                    for key in ('calibration-explanation', 'code-cache'):
                        entry = frozen[key]
                        payload = stock.completion_payload(entry['tokens'], 32)
                        log_begin = (out/'profiler-server.log').stat().st_size
                        begin_unix_ns = time.time_ns()
                        begin_counter_ns = time.perf_counter_ns()
                        response = stock.request(base, '/completion', payload, timeout=600,
                                                  receipt=out/f'http-completion-{key}')
                        end_counter_ns = time.perf_counter_ns()
                        end_unix_ns = time.time_ns()
                        stock.write_json(out/f'response-{key}.json', response)
                        time.sleep(0.05)
                        with (out/'profiler-server.log').open('rb') as source:
                            source.seek(log_begin)
                            segment = source.read().decode('utf-8', errors='replace')
                            log_end = source.tell()
                        events = stock.parse_acceptance(segment)
                        state = study.resource_check(resources)
                        row = {'id': key, 'warmup': key == 'calibration-explanation',
                               'log_start': log_begin, 'log_end': log_end,
                               'begin_unix_ns': begin_unix_ns, 'end_unix_ns': end_unix_ns,
                               'begin_counter_ns': begin_counter_ns, 'end_counter_ns': end_counter_ns,
                               'seconds': (end_counter_ns-begin_counter_ns)/1e9, 'response': response,
                               'acceptance': events,
                               'acceptance_summary': stock.acceptance_summary(events, response.get('timings', {})),
                               'resources': state}
                        ledger.write(json.dumps(row, ensure_ascii=False)+'\n'); ledger.flush()
                        print(json.dumps({'case': key, 'seconds': row['seconds'], 'resources': state['pass']}), flush=True)
                        if not state['pass']:
                            raise RuntimeError('profile request resource bound failed')
                resources.stop.set()
                resources.thread.join()
                stop_command = [str(args.profiler), 'stop', '--session='+session]
                stopped = subprocess.run(stop_command, capture_output=True, encoding='utf-8', timeout=120)
                stock.write_json(out/'profiler-stop.json', {'command': stop_command, 'exit_code': stopped.returncode,
                                 'stdout': stopped.stdout, 'stderr': stopped.stderr})
                if stopped.returncode:
                    raise RuntimeError('profiler did not stop/export cleanly')
                # Ask Nsight to end this exact owned session rather than interpreting
                # an externally killed target as an unexplained profiler failure.
                if proc.poll() is None:
                    shutdown_command = [str(args.profiler), 'shutdown', '--session='+session, '--kill=true']
                    shutdown = subprocess.run(shutdown_command, capture_output=True, encoding='utf-8', timeout=120)
                    stock.write_json(out/'profiler-shutdown.json', {'command': shutdown_command,
                                     'exit_code': shutdown.returncode, 'stdout': shutdown.stdout, 'stderr': shutdown.stderr})
                    if shutdown.returncode:
                        raise RuntimeError('profiler session did not shut down cleanly')
                # Wait for SQLite export and process completion before reporting success.
                proc.wait(timeout=120)
                if target.is_running():
                    raise RuntimeError('profiler left its owned target running')
                files = list(out.glob('cuda-trace*'))
                stock.write_json(out/'completion.json', {'profiler_exit_code': proc.returncode,
                                 'files': [{'name': p.name, 'bytes': p.stat().st_size,
                                            'sha256': stock.digest(p)} for p in files],
                                 'resources': study.resource_check(resources),
                                 'cuda_capture_verified': False})
                # Data presence and actual CUDA activity are checked by the offline analyzer;
                # an exported report by itself is not evidence that CUDA capture succeeded.
    except BaseException as exc:
        stock.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def run(args):
    existed = args.output.exists()
    try:
        return _run(args)
    except BaseException as exc:
        if not existed and args.output.is_dir() and not (args.output/'failure.json').exists():
            stock.write_json(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'binary', 'profiler', 'inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=['cpu-k16', 'offload-k16'], required=True)
    parser.add_argument('--port', type=int, default=8102)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

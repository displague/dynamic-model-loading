"""Stock llama-server measurement harness; no model execution is implemented here."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import socket
from contextlib import contextmanager
from ctypes import wintypes

import psutil

SCALAR_PREFIX = 'Return only the requested answer on one line, without explanation.\n'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def verify_catalog(catalog: dict, models: Path, binary: Path, keys: list[str]) -> dict:
    checked = {}
    expected = {f['name'].lower() for f in catalog['runtime']['files']}
    actual = {p.name.lower() for p in binary.iterdir() if p.is_file() and p.suffix.lower() in {'.exe', '.dll'}}
    if actual != expected:
        raise ValueError('runtime directory has missing or unpinned executable/DLL files')
    for key in keys:
        checked[key] = []
        for item in catalog['models'][key]['files']:
            path = models / key / item['name']
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f'artifact mismatch: {path}')
            checked[key].append(str(path.resolve()))
    for item in catalog['runtime']['files']:
        path = binary / item['name']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'runtime mismatch: {path}')
    return checked


@contextmanager
def managed_process(command, **kwargs):
    """Assign a suspended child to a kill-on-close Windows job before it executes."""
    if os.name != 'nt':
        raise RuntimeError('This native measurement fixture requires Windows')
    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(f'v{i}', ctypes.c_ulonglong) for i in range(6)]
    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    declarations = {
        'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        'OpenThread': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        'ResumeThread': ([wintypes.HANDLE], wintypes.DWORD),
        'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (argtypes, restype) in declarations.items():
        function = getattr(kernel, name); function.argtypes = argtypes; function.restype = restype
    job = kernel.CreateJobObjectW(None, None)
    if not job: raise ctypes.WinError(ctypes.get_last_error())
    proc = None
    try:
        limits = Extended(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        proc = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW | 0x4, **kwargs)
        if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        threads = psutil.Process(proc.pid).threads()
        if len(threads) != 1: raise RuntimeError('suspended child has unexpected thread count')
        thread = kernel.OpenThread(0x0002, False, threads[0].id)
        if not thread: raise ctypes.WinError(ctypes.get_last_error())
        try:
            if kernel.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(thread)
        yield proc
    finally:
        kernel.CloseHandle(job)  # Also kills descendants if the Python runner was interrupted.
        if proc is not None:
            if proc.poll() is None: proc.kill()  # Covers a failure before assignment.
            proc.wait(timeout=20)


class Nvml:
    """Read device totals without spawning a process per sample (Windows fixture)."""
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    def __init__(self):
        self.dll = ctypes.CDLL(str(Path(os.environ['SystemRoot']) / 'System32/nvml.dll'))
        self.check(self.dll.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.dll.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f'NVML error {code}')

    def sample(self):
        mem = self.Memory()
        self.check(self.dll.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)))
        result = {'total': mem.total, 'used': mem.used, 'free': mem.free}
        for name, function, selector in [
            ('temperature_c', 'nvmlDeviceGetTemperature', 0),
            ('sm_clock_mhz', 'nvmlDeviceGetClockInfo', 1),
        ]:
            value = ctypes.c_uint()
            code = getattr(self.dll, function)(self.handle, selector, ctypes.byref(value))
            result[name] = value.value if code == 0 else None
        power = ctypes.c_uint()
        code = self.dll.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
        result['power_mw'] = power.value if code == 0 else None
        return result

    def close(self):
        self.dll.nvmlShutdown()


class Resources:
    def __init__(self, pid: int, path: Path):
        self.process = psutil.Process(pid)
        self.path = path
        self.rows = []
        self.errors = []
        self.stop = threading.Event()
        self.nvml = Nvml()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open('x', encoding='utf-8') as out:
            while not self.stop.is_set():
                try:
                    row = {'monotonic': time.perf_counter(), 'unix_time': time.time(),
                           'gpu': self.nvml.sample(), 'host': psutil.virtual_memory()._asdict(),
                           'process_memory': self.process.memory_info()._asdict(),
                           'process_io': self.process.io_counters()._asdict(),
                           'process_cpu': self.process.cpu_times()._asdict()}
                    self.rows.append(row)
                    out.write(json.dumps(row) + '\n')
                    out.flush()
                except Exception as exc:
                    self.errors.append(repr(exc))
                    out.write(json.dumps({'error': repr(exc), 'error_type': type(exc).__name__,
                                          'error_module': type(exc).__module__,
                                          'monotonic': time.perf_counter()}) + '\n')
                    out.flush()
                    break
                self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self.nvml.close()


def request(base: str, endpoint: str, payload=None, timeout=1800, receipt=None, raw_text=False):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(base + endpoint, data=data, headers={'Content-Type': 'application/json'})
    if receipt is not None:
        write_json(Path(str(receipt)+'.request.json'), {'url':base+endpoint,
                   'method':req.get_method(),'body_utf8':data.decode('utf-8') if data else None})
    error = None
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc; error = exc
    with response:
        metadata = {'status': response.status, 'headers': dict(response.headers),
                    'bytes': 0, 'complete': False}
        if receipt is not None: write_json(Path(str(receipt)+'.http.json'), metadata)
        body = Path(str(receipt)+'.body').open('wb') if receipt is not None else None
        data = bytearray()
        def retain(chunk):
            data.extend(chunk)
            if body is not None: body.write(chunk); body.flush()
        try:
            read = getattr(response, 'read1', response.read)
            while True:
                try: chunk = read(65536)
                except http.client.IncompleteRead as exc:
                    retain(exc.partial)
                    raise
                if not chunk: break
                retain(chunk)
            declared = response.headers.get('Content-Length')
            if declared is not None and len(data) != int(declared):
                raise http.client.IncompleteRead(b'', max(0, int(declared)-len(data)))
            metadata['complete'] = True
        except BaseException as exc:
            metadata['read_error'] = type(exc).__name__
            raise
        finally:
            if body is not None: body.close()
            metadata['bytes'] = len(data)
            if receipt is not None: write_json(Path(str(receipt)+'.http.json'), metadata)
        raw = bytes(data)
    if error is not None: raise error
    return raw.decode('utf-8') if raw_text else json.loads(raw)


def metrics(base, receipt=None):
    return request(base, '/metrics', timeout=10, receipt=receipt, raw_text=True)


def server_command(executable, target, draft, ngl, threads, k, port):
    args = [str(executable), '-m', str(target), '--host', '127.0.0.1', '--port', str(port),
            '-c', '4096', '-b', '256', '-ub', '256', '-np', '1', '-fa', 'on',
            '-ctk', 'f16', '-ctv', 'f16', '-ngl', str(ngl), '--fit', 'off',
            '--load-mode', 'mmap', '--lazy-mode', 'off', '--cache-ram', '0',
            '--no-context-shift', '-t', str(threads), '-tb', '24',
            '--metrics', '--log-colors', 'off', '--log-timestamps', '--perf', '--device', 'CUDA0']
    if draft:
        args += ['--spec-type', 'draft-simple', '-md', str(draft), '--spec-draft-ngl', 'all',
                 '--spec-draft-n-max', str(k), '--spec-draft-n-min', '0',
                 '--spec-draft-p-min', '0', '--spec-draft-type-k', 'f16',
                 '--spec-draft-type-v', 'f16', '--no-spec-draft-backend-sampling',
                 '--spec-draft-threads', '8', '--spec-draft-threads-batch', '24']
        args += ['--spec-draft-device', 'CUDA0']
    else:
        args += ['--spec-type', 'none']
    return args


def validate_placement(log, ngl, draft):
    actual = [tuple(map(int, x)) for x in re.findall(r'offloaded (\d+)/(\d+) layers to GPU', log)]
    expected = [(ngl, 65)]
    if draft:
        layers = {'draft05': 25, 'draft15': 29, 'draft32': 65}[draft]
        expected.append((layers, layers))
    # At zero GPU layers the loader may omit the target offload message.
    if ngl == 0 and actual == expected[1:]: return actual
    if actual != expected:
        raise ValueError(f'effective GPU placement mismatch: expected {expected}, observed {actual}')
    return actual


def validate_inputs(frozen, cases):
    if not isinstance(frozen, dict) or set(frozen) != {c['id'] for c in cases}:
        raise ValueError('frozen inputs do not cover the exact expected cases')
    for case in cases:
        entry = frozen[case['id']]
        if (not isinstance(entry, dict) or not isinstance(entry.get('text'), str)
                or entry.get('max_tokens') != case['max_tokens'] or not isinstance(entry.get('tokens'), list)
                or not entry['tokens'] or not all(type(t) is int and t >= 0 for t in entry['tokens'])):
            raise ValueError('invalid frozen input schema')


def parse_acceptance(log: str):
    events = []
    for line in log.splitlines():
        match = re.search(r'accepted\s+(\d+)/\s*(\d+) draft tokens', line)
        if match:
            accepted, attempted = map(int, match.groups())
            if not 0 <= accepted <= attempted:
                raise ValueError('invalid acceptance event')
            events.append({'accepted': accepted, 'attempted': attempted,
                           'checkpoint_replay': 'restore checkpoint' in line, 'raw': line})
    return events


def acceptance_summary(events, timings):
    final = [e for e in events if not e['checkpoint_replay']]
    accepted = sum(e['accepted'] for e in final)
    attempted = sum(e['attempted'] for e in final)
    replays = any(e['checkpoint_replay'] for e in events)
    consistent = (not replays and accepted == timings.get('draft_n_accepted', 0)
                  and attempted == timings.get('draft_n', 0))
    maximum = max((e['attempted'] for e in final), default=0)
    return {'consistent': consistent, 'checkpoint_replay_present': replays,
            'cycles': len(final), 'accepted': accepted, 'attempted': attempted,
            'survival': [sum(e['accepted'] >= i for e in final)/len(final)
                         for i in range(1, maximum+1)] if consistent else None}


def completion_payload(tokens, cap):
    return {'prompt': tokens, 'n_predict': cap, 'temperature': 0, 'samplers': ['temperature'],
            'seed': 20260912, 'repeat_penalty': 1.0, 'frequency_penalty': 0.0,
            'presence_penalty': 0.0, 'cache_prompt': False, 'return_tokens': True,
            'stream': False, 'n_probs': 0, 'stop': [], 'ignore_eos': False}


def measure(args):
    root = Path(__file__).resolve().parents[1]
    args.models = args.models.resolve()
    args.binary = args.binary.resolve()
    if not 0 <= args.ngl <= 65:
        raise ValueError('target placement must be within 0..65 layers including output')
    if args.mode != 'calibration' and args.inputs is None and not args.reference:
        raise ValueError('evaluation requires frozen reference inputs')
    if args.reference and args.draft:
        raise ValueError('only target-only execution can establish reference inputs')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    if status.strip():
        raise ValueError('measurement requires a clean committed worktree')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/experiment/stock-speculation'],
                   cwd=root, check=True, capture_output=True)
    catalog_path = root / 'configs/stock-speculation-artifacts.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    checked = verify_catalog(catalog, args.models, args.binary, ['target'] + ([args.draft] if args.draft else []))
    workloads = json.loads((root/'data/stock-speculation-workloads.json').read_text(encoding='utf-8'))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [workloads['calibration']]
    if args.mode == 'sustained':
        cases += workloads['sustained']
    elif args.mode == 'smoke':
        for line in (root/'data/dense-interface-fresh.jsonl').read_text(encoding='utf-8').splitlines():
            task = json.loads(line)
            cases.append({'id': task['id'], 'domain': task['domain'],
                          'text': SCALAR_PREFIX + task['task'], 'max_tokens': 64})
    else:
        cases += [dict(workloads['calibration'], id=f'calibration-{i}') for i in range(2)]
    if args.order_seed is not None:
        import random
        measured = cases[1:]
        random.Random(args.order_seed).shuffle(measured)
        cases = [cases[0], *measured]
    cmd = server_command(args.binary/'llama-server.exe', checked['target'][0],
                         checked[args.draft][0] if args.draft else None,
                         args.ngl, args.threads, args.k, args.port)
    removed_environment = {key: value for key, value in os.environ.items()
                           if key.startswith(('LLAMA_', 'GGML_', 'CUDA_'))}
    environment = {key: value for key, value in os.environ.items() if key not in removed_environment}
    environment['LLAMA_TRACE'] = '1'
    # The child receives no inherited llama.cpp tuning/synthetic-acceptance overrides.
    write_json(out/'manifest.json', {'head': head, 'catalog_sha256': digest(catalog_path),
               'workloads_sha256': digest(root/'data/stock-speculation-workloads.json'),
               'mode': args.mode, 'draft': args.draft, 'ngl': args.ngl, 'threads': args.threads,
               'k': args.k if args.draft else None, 'command': cmd, 'artifacts': checked,
               'llama_environment': {'LLAMA_TRACE': '1'}, 'case_order': [x['id'] for x in cases],
               'removed_override_names': sorted(removed_environment),
               'inputs_sha256': digest(args.inputs) if args.inputs else None,
               'smoke_sha256': digest(root/'data/dense-interface-fresh.jsonl'),
               'runner_sha256': digest(Path(__file__)),
               'order_seed': args.order_seed, 'python': os.sys.version, 'psutil': psutil.__version__})
    frozen = json.loads(args.inputs.read_text(encoding='utf-8')) if args.inputs else {}
    if args.inputs is not None: validate_inputs(frozen, cases)
    rendered = {}
    base = f'http://127.0.0.1:{args.port}'
    # Fail rather than talk to a pre-existing process on the chosen port.
    with socket.socket() as probe:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', args.port))
    t_start = time.perf_counter()
    with (out/'server.log').open('xb') as log, managed_process(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=args.binary.resolve()) as proc:
        try:
            with Resources(proc.pid, out/'resources.jsonl') as resources:
                ready = False
                health_attempt = 0
                while time.perf_counter()-t_start < 600:
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}')
                    try:
                        ready = request(base, '/health', timeout=2, receipt=out/f'http-health-{health_attempt}').get('status') == 'ok'
                    except (OSError, urllib.error.HTTPError):
                        pass
                    if ready:
                        break
                    health_attempt += 1
                    time.sleep(0.25)
                if not ready:
                    raise TimeoutError('server startup')
                time.sleep(0.05)
                placement = validate_placement((out/'server.log').read_text(encoding='utf-8',errors='replace'), args.ngl, args.draft)
                startup_samples = list(resources.rows)
                startup_pass = bool(startup_samples) and not resources.errors and all(
                    x['gpu']['used'] <= 15000*2**20 and x['host']['available'] >= 2*2**30
                    for x in startup_samples)
                write_json(out/'startup.json', {'seconds': time.perf_counter()-t_start,
                           'properties': request(base, '/props', receipt=out/'http-props'),
                           'effective_gpu_layers': placement, 'resource_pass': startup_pass,
                           'sampled_gpu_peak': max((x['gpu']['used'] for x in startup_samples), default=None),
                           'sampled_host_available_min': min((x['host']['available'] for x in startup_samples), default=None)})
                if not startup_pass:
                    raise RuntimeError('resource bound or telemetry failure during startup')
                with (out/'rows.jsonl').open('x', encoding='utf-8') as ledger:
                    for index, case in enumerate(cases):
                        messages = [{'role': 'system', 'content': workloads['system']},
                                    {'role': 'user', 'content': case['text']}]
                        formatted = request(base, '/apply-template', {'messages': messages}, receipt=out/f'http-template-{case["id"]}')['prompt']
                        tokens = request(base, '/tokenize', {'content': formatted, 'add_special': True, 'parse_special': True}, receipt=out/f'http-tokenize-{case["id"]}')['tokens']
                        rendered[case['id']] = {'text': formatted, 'tokens': tokens, 'max_tokens': case['max_tokens']}
                        write_json(out/'inputs.json', rendered)
                        if args.inputs is not None:
                            if rendered[case['id']] != frozen[case['id']]:
                                raise ValueError('input rendering diverged from frozen target input')
                        if len(tokens) + case['max_tokens'] > 4096:
                            raise ValueError('case exceeds context')
                        payload = completion_payload(tokens, case['max_tokens'])
                        write_json(out/f'request-{case["id"]}.json', payload)
                        (out/f'metrics-before-{case["id"]}.txt').write_text(metrics(base,receipt=out/f'http-metrics-before-{case["id"]}'), encoding='utf-8')
                        log_start = (out/'server.log').stat().st_size
                        begin = time.perf_counter()
                        response = request(base, '/completion', payload, receipt=out/f'http-completion-{case["id"]}')
                        end = time.perf_counter()
                        write_json(out/f'response-{case["id"]}.json', response)
                        (out/f'metrics-after-{case["id"]}.txt').write_text(metrics(base,receipt=out/f'http-metrics-after-{case["id"]}'), encoding='utf-8')
                        time.sleep(0.05)  # Outside measured request; allow asynchronous stock logs to flush.
                        # Read the exact log interval; final server timings precede its response.
                        with (out/'server.log').open('rb') as source:
                            source.seek(log_start)
                            segment = source.read().decode('utf-8', errors='replace')
                        samples = [x for x in resources.rows if begin <= x['monotonic'] <= end]
                        events = parse_acceptance(segment)
                        gpu_peak = max((x['gpu']['used'] for x in samples), default=None)
                        host_min = min((x['host']['available'] for x in samples), default=None)
                        row = {'id': case['id'], 'domain': case['domain'], 'warmup': index == 0,
                               'request': payload, 'response': response, 'seconds': end-begin,
                               'begin': begin, 'end': end, 'log_start': log_start,
                               'log_end': (out/'server.log').stat().st_size,
                               'acceptance_summary': acceptance_summary(events, response.get('timings', {})),
                               'acceptance': events, 'sampled_gpu_peak': gpu_peak,
                               'sampled_host_available_min': host_min,
                               'telemetry_errors': list(resources.errors),
                               'resource_pass': bool(samples) and not resources.errors and
                                gpu_peak <= 15000*2**20 and host_min >= 2*2**30}
                        ledger.write(json.dumps(row, ensure_ascii=False)+'\n')
                        ledger.flush()
                        write_json(out/'inputs.json', rendered)
                        print(json.dumps({'output': str(out), 'id': case['id'], 'seconds': end-begin,
                                          'tokens': len(response.get('tokens', [])), 'resource_pass': row['resource_pass']}), flush=True)
                        if not row['resource_pass']:
                            raise RuntimeError('resource bound or telemetry failure; raw response retained')
                write_json(out/'completion.json', {'complete': True, 'resource_errors': resources.errors})
        except BaseException as exc:
            write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ngl', type=int, required=True)
    parser.add_argument('--threads', type=int, choices=[8, 16, 24], required=True)
    parser.add_argument('--draft', choices=['draft05', 'draft15', 'draft32'])
    parser.add_argument('--k', type=int, choices=[4, 8, 16], default=16)
    parser.add_argument('--mode', choices=['calibration', 'sustained', 'smoke'], required=True)
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--order-seed', type=int)
    parser.add_argument('--reference', action='store_true')
    measure(parser.parse_args())


if __name__ == '__main__':
    main()

"""Frozen API and installed-client integration checks; no throughput scoring."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import local_session as local
import stock_benchmark as stock

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'def add(a, b):\n    return a - b\n'
EXPECTED = 'def add(a, b):\n    return a + b\n'
TASK = ('Inspect calculator.py using your file tools. Fix add(a, b) so it adds a and b instead of subtracting. '
        'Change only calculator.py; preserve the function name and arguments. Then give a brief summary. '
        'Do not create tests or other files.')
SCHEMA = {'type': 'object', 'properties': {'key': {'type': 'string', 'enum': ['alpha']}},
          'required': ['key'], 'additionalProperties': False}
TOOL = {'name': 'read_probe', 'description': 'Read the probe value for alpha.', 'parameters': SCHEMA}
VERSIONS = {'codex': 'codex-cli 0.154.0', 'claude': '2.1.260 (Claude Code)'}


def events(raw):
    return [json.loads(line[5:].strip()) for line in raw.splitlines()
            if line.startswith('data:') and line[5:].strip() not in ('', '[DONE]')]


def text_output(kind, body, stream=False):
    if stream:
        es = events(body)
        if kind == 'chat':
            return ''.join(e.get('choices', [{}])[0].get('delta', {}).get('content') or ''
                           for e in es if e.get('choices'))
        if kind == 'responses':
            return ''.join(e.get('delta', '') for e in es if e.get('type') == 'response.output_text.delta')
        return ''.join(e.get('delta', {}).get('text', '') for e in es if e.get('type') == 'content_block_delta')
    if kind == 'chat':
        return body['choices'][0]['message'].get('content') or ''
    if kind == 'responses':
        return ''.join(c.get('text', '') for o in body.get('output', []) for c in o.get('content', []))
    return ''.join(c.get('text', '') for c in body.get('content', []))


def stream_complete(kind, raw):
    es = events(raw)
    if any(e.get('error') or e.get('type') in ('error', 'response.failed', 'response.incomplete') for e in es):
        return False
    if kind == 'chat':
        return ('data: [DONE]' in raw and any(c.get('finish_reason') == 'stop'
                for e in es for c in e.get('choices', [])))
    if kind == 'responses':
        return any(e.get('type') == 'response.completed' and e.get('response', {}).get('status') == 'completed'
                   for e in es)
    return (any(e.get('type') == 'message_stop' for e in es) and
            any(e.get('type') == 'message_delta' and e.get('delta', {}).get('stop_reason') == 'end_turn'
                for e in es))


def http(base, route, payload, prefix):
    stock.write_json(prefix.with_suffix('.request.json'), {'route': route, 'body': payload})
    request = urllib.request.Request(base+route, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        response = exc
    chunks = []
    with response, prefix.with_suffix('.response.txt').open('xb') as saved:
        status = response.status
        stock.write_json(prefix.with_suffix('.http.json'), {'status': status, 'headers': dict(response.headers)})
        start = time.monotonic()
        try:
            while True:
                block = response.read1(16384)
                if not block:
                    remaining = getattr(response, 'length', None)
                    if remaining:
                        raise IncompleteRead(b'', remaining)
                    break
                saved.write(block)
                saved.flush()
                chunks.append(block)
                if time.monotonic()-start > 180:
                    raise TimeoutError('HTTP body deadline')
        except BaseException as exc:
            if isinstance(exc, IncompleteRead):
                saved.write(exc.partial)
                saved.flush()
            stock.write_json(prefix.with_suffix('.transport-error.json'),
                             {'type': type(exc).__name__, 'message': str(exc)})
            raise
    raw = b''.join(chunks)
    if status != 200:
        raise ValueError(f'HTTP {status}: {raw.decode(errors="replace")}')
    return raw.decode() if payload.get('stream') else json.loads(raw)


def base_request(kind, prompt):
    common = {'model': local.ALIAS, 'temperature': 0, 'seed': 0}
    if kind == 'responses':
        return dict(common, input=[{'role': 'user', 'content': prompt}], max_output_tokens=128)
    return dict(common, messages=[{'role': 'user', 'content': prompt}], max_tokens=128)


def route(kind):
    return '/v1/'+{'chat': 'chat/completions', 'responses': 'responses', 'messages': 'messages'}[kind]


def api_check(base, out, kind, tools=True):
    simple = base_request(kind, 'Reply with exactly READY_LOCAL and nothing else.')
    simple['stream'] = True
    raw = http(base, route(kind), simple, out/(kind+'-text'))
    result = {'text_stream': stream_complete(kind, raw) and text_output(kind, raw, True).strip() == 'READY_LOCAL'}
    if not tools:
        return result
    first = base_request(kind, 'Use read_probe with key alpha, then report the exact value returned by the tool.')
    if kind == 'chat':
        first.update(tools=[{'type': 'function', 'function': TOOL}], tool_choice='required')
    elif kind == 'responses':
        first.update(tools=[dict(TOOL, type='function')], tool_choice='required')
    else:
        first.update(tools=[{'name': TOOL['name'], 'description': TOOL['description'], 'input_schema': SCHEMA}],
                     tool_choice={'type': 'any'})
    answer = http(base, route(kind), first, out/(kind+'-call'))
    second = base_request(kind, '')
    if kind == 'chat':
        message = answer['choices'][0]['message']
        call = message['tool_calls'][0]
        name, arguments = call['function']['name'], json.loads(call['function']['arguments'])
        second['messages'] = first['messages']+[message,
            {'role': 'tool', 'tool_call_id': call['id'], 'content': 'PROBE_4821'}]
        second['tool_choice'] = 'none'
    elif kind == 'responses':
        call = next(o for o in answer['output'] if o['type'] == 'function_call')
        name, arguments = call['name'], json.loads(call['arguments'])
        second['input'] = first['input']+answer['output']+[
            {'type': 'function_call_output', 'call_id': call['call_id'], 'output': 'PROBE_4821'}]
        second['tool_choice'] = 'none'
    else:
        call = next(o for o in answer['content'] if o['type'] == 'tool_use')
        name, arguments = call['name'], call['input']
        second['messages'] = first['messages']+[
            {'role': 'assistant', 'content': answer['content']},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'PROBE_4821'}]}]
    second['stream'] = True
    raw = http(base, route(kind), second, out/(kind+'-result'))
    result.update(tool_call=name == 'read_probe' and arguments == {'key': 'alpha'},
                  tool_result=stream_complete(kind, raw) and 'PROBE_4821' in text_output(kind, raw, True))
    return result


def correct_file(path):
    try:
        return ast.dump(ast.parse(path.read_text(encoding='utf-8'))) == ast.dump(ast.parse(EXPECTED))
    except (OSError, SyntaxError):
        return False


def tool_events(client, parsed):
    if client == 'claude':
        messages = [row['message'] for row in parsed if isinstance(row.get('message'), dict)]
        return [block for message in messages for block in message.get('content', [])
                if isinstance(block, dict) and block.get('type') == 'tool_use']
    return [row for row in parsed if isinstance(row.get('item'), dict) and row['item'].get('type') in
            ('command_execution', 'file_change', 'mcp_tool_call')]


def client_check(base, out, client):
    work = out/client/'workspace'
    work.mkdir(parents=True, exist_ok=False)
    (work/'calculator.py').write_text(ORIGINAL, encoding='utf-8')
    task = TASK+f'\nThe exact file is {work / "calculator.py"}. '
    if client == 'codex':
        task += ('Use PowerShell code directly in exec_command, for example Get-Content -LiteralPath calculator.py; '
                 'do not wrap it in another powershell.exe or pwsh.exe invocation. Omit the sandbox_permissions argument entirely.')
    cmd, env, effective = local.client_settings(client, base, out/client/'state', work, os.environ,
        prompt=task, result=out/client/'last-message.txt')
    if client == 'codex':
        Path(env['CODEX_HOME']).mkdir(parents=True)
    stock.write_json(out/client/'invocation.json', {'command': cmd, 'child_environment': effective,
                                                  'cwd': str(work), 'task': task})
    version = subprocess.check_output([local.executable(client), '--version'], text=True)
    (out/client/'version.txt').write_text(version, encoding='utf-8')
    if version.strip() != VERSIONS[client]:
        raise ValueError(f'client version differs from protocol: {version.strip()}')
    timeout = False
    with (out/client/'stdout.jsonl').open('xb') as stdout, (out/client/'stderr.log').open('xb') as stderr:
        with stock.managed_process(cmd, cwd=work, env=env, stdout=stdout, stderr=stderr) as proc:
            try:
                code = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timeout = True
                code = None
    raw = (out/client/'stdout.jsonl').read_text(encoding='utf-8', errors='replace')
    parsed = []
    for line in raw.splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    observed_tools = tool_events(client, parsed)
    files = sorted(str(p.relative_to(work)) for p in work.rglob('*') if p.is_file())
    passed = (code == 0 and not timeout and correct_file(work/'calculator.py') and bool(observed_tools)
              and files == ['calculator.py'])
    result = {'passed': passed, 'exit_code': code, 'timeout': timeout,
              'correct_ast': correct_file(work/'calculator.py'), 'tool_events': len(observed_tools),
              'files': files}
    stock.write_json(out/client/'result.json', result)
    return result


@contextmanager
def server(root, out, profile, port):
    stop = out/'stop'
    cmd = [os.sys.executable, str(ROOT/'scripts/local_session.py'), 'serve', '--root', str(root),
           '--profile', profile, '--port', str(port), '--output', str(out/'server'), '--stop-file', str(stop)]
    out.mkdir(parents=True, exist_ok=False)
    with (out/'launcher.log').open('xb') as log, stock.managed_process(cmd, stdout=log, stderr=subprocess.STDOUT) as proc:
        start = time.monotonic()
        while not (out/'server/ready.json').exists():
            if proc.poll() is not None:
                raise RuntimeError('launcher failed: '+str(out/'launcher.log'))
            if time.monotonic()-start > 660:
                raise TimeoutError('launcher deadline')
            time.sleep(.5)
        try:
            yield f'http://127.0.0.1:{port}'
        finally:
            stop.touch()
            if proc.wait(timeout=40) != 0:
                raise RuntimeError('launcher failed during native shutdown')


def all_passed(results):
    if set(results) != set(local.PROFILES):
        return False
    for profile, checks in results.items():
        expected = {'chat', 'responses', 'messages', 'codex', 'claude'} if profile == 'measured' else {'chat'}
        if set(checks) != expected:
            return False
        for kind, values in checks.items():
            if kind in ('codex', 'claude'):
                if values.get('passed') is not True:
                    return False
            else:
                keys = {'text_stream', 'tool_call', 'tool_result'} if profile == 'measured' else {'text_stream'}
                if set(values) != keys or not all(values[k] is True for k in keys):
                    return False
    return True


def run(args):
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('clean committed source required')
    args.output.mkdir(parents=True, exist_ok=False)
    stock.write_json(args.output/'source.json', {'head': git('rev-parse', 'HEAD'),
        'files': {p: stock.digest(ROOT/p) for p in ['scripts/local_session.py', 'scripts/local_session_smoke.py',
                  'docs/local-session-protocol.md', 'docs/local-session-amendment-1.md',
                  'configs/stock-speculation-artifacts.json']}})
    results = {}
    for profile in local.PROFILES:
        out = args.output/profile
        try:
            with server(args.root, out, profile, args.port) as base:
                results[profile] = {}
                for kind in (['chat', 'responses', 'messages'] if profile == 'measured' else ['chat']):
                    try:
                        results[profile][kind] = api_check(base, out, kind, profile == 'measured')
                    except Exception as exc:
                        results[profile][kind] = {'error': repr(exc)}
                    stock.write_json(args.output/'progress.json', results)
                if profile == 'measured':
                    for client in ['codex', 'claude']:
                        try:
                            results[profile][client] = client_check(base, out, client)
                        except Exception as exc:
                            results[profile][client] = {'error': repr(exc)}
                        stock.write_json(args.output/'progress.json', results)
        except Exception as exc:
            results.setdefault(profile, {})['server_error'] = repr(exc)
        stock.write_json(args.output/'progress.json', results)
    stock.write_json(args.output/'results.json', results)
    stock.write_json(args.output/'verdict.json', {'passed': all_passed(results)})
    print(json.dumps(results, indent=2), flush=True)
    return all_passed(results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8107)
    a = p.parse_args()
    a.root = a.root.resolve()
    a.output = a.output.resolve()
    raise SystemExit(0 if run(a) else 1)

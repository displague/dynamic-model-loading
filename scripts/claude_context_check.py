"""Exercise installed Claude compaction against a synthetic loopback API, without inference."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import os

import local_session as local
import stock_benchmark as stock


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(content_text(v) for v in value)
    if isinstance(value, dict):
        return content_text(value.get('text', value.get('content', '')))
    return ''


def successful_fixture_reads(requests):
    return {b['tool_use_id'] for r in requests for m in r['body'].get('messages', [])
            for b in (m.get('content') if isinstance(m.get('content'), list) else [])
            if b.get('type') == 'tool_result' and b.get('tool_use_id')
            and not b.get('is_error') and 'A small fixture.' in content_text(b)}


def read_budget_evidence(rows, requests):
    results = [b for r in requests for m in r['body'].get('messages', [])
               for b in (m.get('content') if isinstance(m.get('content'), list) else [])
               if b.get('type') == 'tool_result']
    errors = [b for b in results if b.get('tool_use_id') == 'tool_1' and b.get('is_error')
              and 'maximum allowed tokens (2048)' in content_text(b)]
    trimmed = []
    for row in rows:
        if row.get('type') != 'user' or not isinstance(row.get('tool_use_result'), dict):
            continue
        blocks = row.get('message', {}).get('content', [])
        if not isinstance(blocks, list) or not any(b.get('type') == 'tool_result'
                and b.get('tool_use_id') == 'tool_1' and not b.get('is_error') for b in blocks):
            continue
        f = row['tool_use_result'].get('file', {})
        if f.get('truncatedByTokenCap') and 0 < f.get('numLines', 0) < f.get('totalLines', 0):
            trimmed.append(f)
    bounded = [b for b in results if b.get('tool_use_id') == 'tool_2' and not b.get('is_error')
               and 'row 0:' in content_text(b) and 'row 1:' in content_text(b) and 'row 2:' not in content_text(b)]
    return errors, trimmed, bounded


class Fixture:
    def __init__(self, directory, kind):
        self.directory, self.kind = directory, kind
        self.requests, self.main_calls, self.compactions = [], 0, 0

    def response(self, body):
        # Main requests expose Read/Edit; compaction disables tools.
        summary = not body.get('tools') or body.get('tool_choice', {}).get('type') == 'none'
        if summary:
            self.compactions += 1
            block = {'type': 'text', 'text': 'The task is to read fixture.txt. Continue with that task.'}
        else:
            self.main_calls += 1
            n = self.main_calls
            if self.kind == 'read-budget' and n <= 2:
                inp = {'file_path': str(self.directory/'large.txt')}
                if n == 2:
                    inp.update(offset=1, limit=2)
            elif self.kind == 'manual-command' and n > 1 or self.kind == 'read-budget' and n > 2 or n > 5:
                inp = None
            else:
                inp = {'file_path': str(self.directory/f'fixture-{n}.txt'), 'offset': 1, 'limit': 1}
            block = ({'type': 'tool_use', 'id': f'tool_{n}', 'name': 'Read', 'input': inp}
                     if inp else {'type': 'text', 'text': 'FIXTURE_DONE'})
        tokens = 6000 if self.kind == 'loop' else 1000
        msg = {'id': f'msg_{len(self.requests)}', 'type': 'message', 'role': 'assistant',
               'model': local.ALIAS, 'content': [], 'stop_reason': None, 'stop_sequence': None,
               'usage': {'input_tokens': tokens, 'output_tokens': 1,
                         'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}
        start = dict(block)
        if block['type'] == 'tool_use':
            start['input'] = {}
            delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
        else:
            start['text'] = ''
            delta = {'type': 'text_delta', 'text': block['text']}
        events = [
            {'type': 'message_start', 'message': msg},
            {'type': 'content_block_start', 'index': 0, 'content_block': start},
            {'type': 'content_block_delta', 'index': 0, 'delta': delta},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if block['type'] == 'tool_use' else 'end_turn',
                                               'stop_sequence': None}, 'usage': {'output_tokens': 12}},
            {'type': 'message_stop'},
        ]
        return summary, ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events)


def case(root, name, mode, kind):
    directory = root/name
    directory.mkdir()
    for n in range(1, 6):
        (directory/f'fixture-{n}.txt').write_text(f'A small fixture. Number {n}.\n', encoding='utf-8')
    (directory/'large.txt').write_text(''.join(f'row {i}: a b c d e f g h i j k l m n o p\n' for i in range(1500)), encoding='utf-8')
    fixture = Fixture(directory, kind)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            if self.path.split('?')[0].endswith('/count_tokens'):
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"input_tokens":1000}')
                return
            if self.path.split('?')[0] != '/v1/messages':
                self.send_error(404)
                return
            index = len(fixture.requests)
            summary, raw = fixture.response(body)
            fixture.requests.append({'path': self.path, 'summary': summary, 'body': body})
            write(directory/f'request-{index:02}.json', fixture.requests[-1])
            (directory/f'response-{index:02}.sse').write_text(raw, encoding='utf-8')
            encoded = raw.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = serving = None
    rows, inbox = [], queue.Queue()
    started = time.monotonic()
    error = None
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        base = f'http://127.0.0.1:{server.server_port}'
        cmd, env, effective = local.client_settings('claude', base, directory/'state', directory, os.environ,
                                                  prompt='Read the fixture.', result=directory/'unused',
                                                  claude_compaction=mode)
        # Test-only config isolation: do not archive a real user's device ID or preferences.
        (directory/'state').mkdir()
        effective['CLAUDE_CONFIG_DIR'] = str(directory/'state')
        # This accepted minimum is capped by the truthful model window (18432).
        # Name a window explicitly so fresh installations exercise the proactive path.
        effective['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] = '100000'
        env.update(effective)
        # Stream one user turn, then an explicit command on the same live session.
        cmd.remove('Read the fixture.')
        cmd += ['--input-format', 'stream-json']
        write(directory/'invocation.json', {'command': cmd, 'environment': effective, 'kind': kind,
                                            'synthetic_usage': 6000 if kind == 'loop' else 1000})
        with (directory/'stderr.log').open('w', encoding='utf-8') as err, \
                (directory/'stdout.jsonl').open('w', encoding='utf-8') as out, \
                stock.managed_process(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                      text=True, encoding='utf-8', env=env, cwd=directory) as proc:
            def read():
                for line in proc.stdout:
                    out.write(line); out.flush()
                    try:
                        row = json.loads(line)
                        rows.append(row); inbox.put(row)
                    except ValueError:
                        pass
                inbox.put(None)
            reader = threading.Thread(target=read, daemon=True)
            reader.start()

            def send(text):
                proc.stdin.write(json.dumps({'type': 'user', 'message': {'role': 'user', 'content': text}})+'\n')
                proc.stdin.flush()

            send('Read the fixture files as requested.')
            results = 0
            while True:
                remaining = 120-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('CLI fixture exceeded 120 seconds')
                row = inbox.get(timeout=remaining)
                if row is None:
                    break
                if row.get('type') == 'result':
                    results += 1
                    if kind == 'manual-command' and results == 1:
                        send('/compact Keep only the fixture filename and next action.')
                    else:
                        proc.stdin.close()
            proc.wait(timeout=max(.1, 120-(time.monotonic()-started)))
            exit_code = proc.returncode
            reader.join(timeout=5)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        exit_code = None
    finally:
        if server is not None:
            if serving is not None and serving.is_alive():
                server.shutdown()
                serving.join()
            server.server_close()
    boundaries = [r for r in rows if r.get('subtype') == 'compact_boundary']
    diagnostics = content_text([r.get('message', '') for r in rows])+' '+json.dumps(rows)
    done = any(r.get('type') == 'result' and r.get('result') == 'FIXTURE_DONE' and not r.get('is_error') for r in rows)
    read_errors, trimmed, bounded_reads = read_budget_evidence(rows, fixture.requests)
    good_reads = successful_fixture_reads(fixture.requests)
    common = error is None and exit_code == 0
    if kind == 'loop':
        passed = common and (bool(boundaries) or 'Autocompact is thrashing' in diagnostics) if mode == 'auto' else common and done and fixture.main_calls == 6 and good_reads == {f'tool_{n}' for n in range(1, 6)} and not boundaries
    elif kind == 'read-budget':
        passed = common and done and bool(read_errors or trimmed) and bool(bounded_reads)
    else:
        passed = common and done and any(r.get('compact_metadata', {}).get('trigger') == 'manual' for r in boundaries)
    result = {'case': name, 'passed': bool(passed), 'error': error, 'exit_code': exit_code,
              'main_requests': fixture.main_calls, 'summary_requests': fixture.compactions,
              'boundaries': boundaries, 'done': done, 'successful_fixture_reads': sorted(good_reads), 'read_budget_rejected': bool(read_errors),
              'read_budget_trimmed': [{'num_lines': f['numLines'], 'total_lines': f['totalLines']} for f in trimmed],
              'bounded_read_succeeded': bool(bounded_reads), 'seconds': time.monotonic()-started}
    write(directory/'result.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    exe = Path(local.executable('claude'))
    write(root/'source.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=local.ROOT, text=True).strip(),
        'claude_version': subprocess.check_output([str(exe), '--version'], text=True).strip(),
        'claude_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'inference': False})
    results = [case(root, name, mode, kind) for name, mode, kind in [
        ('auto-loop', 'auto', 'loop'), ('manual-loop', 'manual', 'loop'),
        ('read-budget', 'manual', 'read-budget'), ('manual-command', 'manual', 'manual-command')]]
    write(root/'results.json', results)
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()

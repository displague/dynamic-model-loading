"""Native Claude file-tool checks with scripted API responses, not model inference."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import hashlib
import json
import os
import subprocess
import threading

import local_session as local
from claude_context_check import write, content_text

ORIGINAL = 'first();\nrepeat();\nsecond();\nrepeat();\n'
OLD = 'second();\nrepeat();'
NEW = 'second();\nfixed();'
EXPECTED = ORIGINAL.replace(OLD, NEW)


def stream(block):
    start = dict(block)
    is_tool = block['type'] == 'tool_use'
    start['input' if is_tool else 'text'] = {} if is_tool else ''
    delta = ({'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
             if is_tool else {'type': 'text_delta', 'text': block['text']})
    message = {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': local.ALIAS,
               'content': [], 'stop_reason': None, 'stop_sequence': None,
               'usage': {'input_tokens': 1000, 'output_tokens': 1,
                         'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}
    events = [{'type': 'message_start', 'message': message},
              {'type': 'content_block_start', 'index': 0, 'content_block': start},
              {'type': 'content_block_delta', 'index': 0, 'delta': delta},
              {'type': 'content_block_stop', 'index': 0},
              {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if is_tool else 'end_turn',
                                                 'stop_sequence': None}, 'usage': {'output_tokens': 50}},
              {'type': 'message_stop'}]
    return ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events)


def plan(work, bare):
    existing, new = str(work/'existing.txt'), str(work/'new.txt')
    read = ('Read', {'file_path': existing, 'offset': 1, 'limit': 10})
    good = ('Edit', {'file_path': existing, 'old_string': OLD, 'new_string': NEW})
    if bare:
        return [read, ('Edit', {'file_path': new, 'old_string': '', 'new_string': 'created\n'}), good]
    return [read, *[('Edit', {'file_path': existing, 'old_string': old, 'new_string': 'wrong\n'})
                    for old in ['', 'repeat();', 'absent();']], good,
            ('Write', {'file_path': new, 'content': 'created\n'}),
            ('Read', {'file_path': new, 'offset': 1, 'limit': 10}),
            ('Write', {'file_path': new, 'content': 'updated\n'})]


def verdict(requests, bare, final, exit_code):
    expected_tools = ['Edit', 'Read'] if bare else ['Edit', 'Read', 'Write']
    tools_ok = bool(requests) and all(sorted(t['name'] for t in r['body'].get('tools', [])) == expected_tools
                                     for r in requests)
    results = {}
    for r in requests:
        for msg in r['body'].get('messages', []):
            if not isinstance(msg.get('content'), list):
                continue
            for block in msg['content']:
                if block.get('type') == 'tool_result':
                    results[block['tool_use_id']] = block
    count = 3 if bare else 8
    expected_errors = {} if bare else {2: 'file already exists', 3: 'Found 2 matches', 4: 'String to replace not found'}
    steps = {}
    for n in range(1, count+1):
        result = results.get(f'tool_{n}')
        wanted_error = expected_errors.get(n)
        steps[str(n)] = bool(result and
            (result.get('is_error') is True and wanted_error in content_text(result.get('content', ''))
             if wanted_error else not result.get('is_error', False)))
    unchanged = bare or (len(requests) == count+1 and all(
        requests[n]['files'] == {'existing.txt': ORIGINAL} for n in (2, 3, 4)))
    created_at = 2 if bare else 6
    creation_ok = len(requests) > created_at and requests[created_at]['files'] == {
        'existing.txt': ORIGINAL if bare else EXPECTED, 'new.txt': 'created\n'}
    first_read = content_text(results.get('tool_1', {}).get('content', ''))
    reads_ok = 'first();' in first_read and 'second();' in first_read and 'repeat();' in first_read
    if not bare:
        reads_ok = reads_ok and 'created' in content_text(results.get('tool_7', {}).get('content', ''))
    final_ok = final == {'existing.txt': EXPECTED, 'new.txt': 'created\n' if bare else 'updated\n'}
    return {'passed': exit_code == 0 and tools_ok and all(steps.values()) and unchanged and final_ok and creation_ok and reads_ok
                     and len(requests) == count+1,
            'advertised_tools': sorted(t['name'] for t in requests[0]['body'].get('tools', [])) if requests else [],
            'tools_ok': tools_ok, 'steps': steps, 'rejections_preserve_bytes': unchanged,
            'creation_contents_ok': creation_ok, 'read_contents_ok': reads_ok, 'final_files_ok': final_ok}


def case(root, bare):
    directory = root/('bare-control' if bare else 'safe-candidate')
    work = directory/'workspace'
    work.mkdir(parents=True)
    (directory/'state').mkdir()
    (work/'existing.txt').write_text(ORIGINAL, encoding='utf-8', newline='')
    actions, requests = plan(work, bare), []

    def snapshot():
        return {p.relative_to(work).as_posix(): p.read_bytes().decode('utf-8') for p in sorted(work.rglob('*')) if p.is_file()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            path = self.path.split('?')[0]
            if path.endswith('/count_tokens'):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"input_tokens":1000}')
                return
            if path != '/v1/messages':
                self.send_error(404)
                return
            n = len(requests)
            record = {'body': body, 'files': snapshot()}
            requests.append(record)
            write(directory/f'request-{n:02}.json', record)
            block = ({'type': 'tool_use', 'id': f'tool_{n+1}', 'name': actions[n][0], 'input': actions[n][1]}
                     if n < len(actions) else {'type': 'text', 'text': 'FIXTURE_DONE'})
            raw = stream(block)
            (directory/f'response-{n:02}.sse').write_text(raw, encoding='utf-8')
            encoded = raw.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers(); self.wfile.write(encoded)

    server, thread, error, code = None, None, None, None
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cmd, env, effective = local.client_settings('claude', f'http://127.0.0.1:{server.server_port}',
            directory/'state', work, os.environ, prompt='Perform the fixture file operations.', result=directory/'unused')
        if bare:
            cmd[cmd.index('--safe-mode')] = '--bare'
        effective['CLAUDE_CONFIG_DIR'] = str(directory/'state')
        env.update(effective)
        write(directory/'invocation.json', {'command': cmd, 'environment': effective, 'cwd': str(work)})
        with (directory/'stdout.jsonl').open('wb') as out, (directory/'stderr.log').open('wb') as err, \
                local.stock.managed_process(cmd, env=env, cwd=work, stdout=out, stderr=err) as proc:
            code = proc.wait(timeout=120)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown(); thread.join()
            server.server_close()
    final = snapshot()
    value = verdict(requests, bare, final, code)
    value.update(exit_code=code, error=error, final=final, passed=value['passed'] and error is None)
    write(directory/'result.json', value)
    print(json.dumps(value), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    exe = Path(local.executable('claude'))
    version = subprocess.check_output([str(exe), '--version'], text=True).strip()
    write(root/'source.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=local.ROOT, text=True).strip(),
        'claude_version': version, 'claude_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'inference': False})
    if version != '2.1.260 (Claude Code)':
        raise ValueError('CLI version differs from protocol')
    results = [case(root, bare) for bare in (True, False)]
    write(root/'results.json', results)
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()

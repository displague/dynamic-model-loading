"""Recompute compact integration findings from API bodies, client events and files."""
import argparse
import json
from pathlib import Path

import local_session as local
import local_session_smoke as smoke
import stock_benchmark as stock


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def api(out, kind, tools):
    stems = [kind+'-text']+([kind+'-call', kind+'-result'] if tools else [])
    for stem in stems:
        if read(out/(stem+'.http.json'))['status'] != 200 or (out/(stem+'.transport-error.json')).exists():
            raise ValueError('failed HTTP receipt: '+stem)
    raw = (out/(kind+'-text.response.txt')).read_text(encoding='utf-8')
    result = {'text_stream': smoke.stream_complete(kind, raw) and smoke.text_output(kind, raw, True).strip() == 'READY_LOCAL'}
    if tools:
        body = read(out/(kind+'-call.response.txt'))
        if kind == 'chat':
            call = body['choices'][0]['message']['tool_calls'][0]['function']
            name, args = call['name'], json.loads(call['arguments'])
        elif kind == 'responses':
            call = next(x for x in body['output'] if x['type'] == 'function_call')
            name, args = call['name'], json.loads(call['arguments'])
        else:
            call = next(x for x in body['content'] if x['type'] == 'tool_use')
            name, args = call['name'], call['input']
        raw = (out/(kind+'-result.response.txt')).read_text(encoding='utf-8')
        result.update(tool_call=name == 'read_probe' and args == {'key': 'alpha'},
                      tool_result=smoke.stream_complete(kind, raw) and 'PROBE_4821' in smoke.text_output(kind, raw, True))
    return result


def client(out, name):
    folder = out/name
    rows = [json.loads(x) for x in (folder/'stdout.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    observed = smoke.tool_events(name, rows)
    files = sorted(p.relative_to(folder/'workspace').as_posix() for p in (folder/'workspace').rglob('*') if p.is_file())
    correct = smoke.correct_file(folder/'workspace/calculator.py')
    stored = read(folder/'result.json') if (folder/'result.json').exists() else None
    result = {'version': (folder/'version.txt').read_text(encoding='utf-8').strip(),
              'correct_ast': correct, 'tool_events': len(observed), 'files': files,
              'recorded_result': stored, 'passed': bool(stored and stored['exit_code'] == 0 and
                  not stored['timeout'] and correct and observed and files == ['calculator.py'])}
    if stored:
        for key in ('correct_ast', 'tool_events', 'files', 'passed'):
            if result[key] != stored[key]:
                raise ValueError('client verdict differs from raw records: '+name+'/'+key)
    result['tool_names'] = [x.get('name') for x in observed] if name == 'claude' else None
    return result


def analyze(root):
    result = {'schema': 1, 'runs': {}}
    for label in ['v1', 'v2']:
        path = root/('local-session-20260913-'+label)
        recorded = read(path/'results.json')
        item = {'source': read(path/'source.json'), 'recorded_all_passed': read(path/'verdict.json')['passed'], 'profiles': {}}
        for profile in local.PROFILES:
            out = path/profile
            launch = read(out/'server/launch.json')
            ready = read(out/'server/ready.json')
            done = read(out/'server/completion.json')
            rows = [json.loads(x) for x in (out/'server/resources.jsonl').read_text().splitlines()]
            peak = max(x['gpu']['used'] for x in rows)
            minimum = min(x['host']['available'] for x in rows)
            if peak != done['resources']['gpu_peak'] or minimum != done['resources']['host_available_min']:
                raise ValueError('resource summary differs')
            checks = {k: api(out, k, profile == 'measured') for k in
                      (['chat', 'responses', 'messages'] if profile == 'measured' else ['chat'])}
            if any(checks[k] != recorded[profile][k] for k in checks):
                raise ValueError('API verdict differs')
            item['profiles'][profile] = {'configuration': launch['configuration'], 'api': checks,
                'placement': ready['placement'], 'resources': done['resources'], 'requested_stop': done['requested_stop']}
            if profile == 'measured':
                item['clients'] = {name: client(out, name) for name in ['codex', 'claude']}
        result['runs'][label] = item
    result['scope'] = 'API and one-file integration checks; no throughput, broad agent quality or independent target-fidelity claim.'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    stock.write_json(a.output, analyze(a.root))

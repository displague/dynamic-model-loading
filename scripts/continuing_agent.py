"""Stock target replay and paired continuing-conversation measurements (no runtime patch)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

import stock_benchmark as stock
import verification_offload as previous

ROOT = Path(__file__).resolve().parents[1]
EOG = {151643, 151645}
SOURCES = ['scripts/continuing_agent.py', 'scripts/stock_benchmark.py', 'scripts/verification_offload.py',
           'data/committed-replay.json', 'data/continuing-agent-turns.json',
           'docs/continuing-agent-protocol.md', 'configs/stock-speculation-artifacts.json']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def settings(kind, condition):
    cfg = previous.configuration('long' if kind != 'replay-short' else 'short',
                                 'cpu-k4' if condition == 'default' else 'offload-k16')
    if kind.startswith('replay'):
        cfg.update(k=None, threshold=32)
    return cfg


def parse_sse(events):
    """A final stock event has empty content/tokens; never count it twice."""
    finals = [e for e in events if e.get('stop') is True]
    if len(finals) != 1 or events[-1] is not finals[0]:
        raise ValueError('SSE has no unique terminal event')
    parts = events[:-1]
    if any('error' in e or e.get('stop') is not False for e in parts):
        raise ValueError('invalid SSE partial event')
    final = dict(finals[0])
    if final['tokens'] or final['content']:
        raise ValueError('unexpected final SSE payload; cannot safely assemble')
    final['tokens'] = [t for e in parts for t in e['tokens']]
    final['content'] = ''.join(e['content'] for e in parts)
    if len(final['tokens']) != final['tokens_predicted']:
        raise ValueError('streamed token coverage differs from native count')
    return final


def streaming_request(base, payload, prefix):
    """Retain raw SSE and event arrival times before parsing the completed response."""
    data = json.dumps(payload).encode('utf-8')
    stock.write_json(prefix.with_suffix('.request.json'),
                     {'url': base+'/completion', 'method': 'POST', 'body_utf8': data.decode()})
    req = urllib.request.Request(base+'/completion', data=data, headers={'Content-Type':'application/json'})
    begin = time.perf_counter()
    meta = {'begin': begin, 'complete': False}
    events, arrivals, first_token, first_text = [], [], None, None
    try:
        error=None
        try:
            response=urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            response=exc; error=exc
        with response, prefix.with_suffix('.body').open('xb') as raw:
            meta.update(status=response.status, headers=dict(response.headers), headers_at=time.perf_counter())
            if error is not None:
                raw.write(response.read()); raw.flush()
                raise error
            pending = []
            for line in response:
                raw.write(line); raw.flush()
                if time.perf_counter()-begin>1800: raise TimeoutError('SSE request wall deadline')
                if line.strip():
                    if line.startswith(b'data: '):
                        pending.append(line[6:].rstrip(b'\r\n'))
                    elif not line.startswith(b':'):
                        raise ValueError('unexpected SSE field')
                    continue
                if not pending:
                    continue
                body = b'\n'.join(pending); pending = []
                if body == b'[DONE]':
                    continue
                event = json.loads(body)
                now = time.perf_counter()
                events.append(event); arrivals.append(now)
                if event.get('tokens') and first_token is None: first_token = now
                if event.get('content') and first_text is None: first_text = now
            if pending:
                raise ValueError('unterminated SSE event')
        result = parse_sse(events)
        meta['complete'] = True
        return result, meta
    finally:
        meta.update(end=time.perf_counter(), event_arrivals=arrivals,
                    first_token_at=first_token, first_text_at=first_text)
        stock.write_json(prefix.with_suffix('.http.json'), meta)
        stock.write_json(prefix.with_suffix('.events.json'), events)


def check_output(response, cap):
    tokens = response['tokens']
    if not tokens or len(tokens)>cap or len(tokens) != response['tokens_predicted'] or response['truncated']:
        raise ValueError('invalid or truncated output')
    if response['stop_type'] == 'limit':
        if len(tokens) != cap or any(t in EOG for t in tokens):
            raise ValueError('limit stopping is inconsistent')
    elif response['stop_type'] == 'eos':
        if tokens[-1] not in EOG or any(t in EOG for t in tokens[:-1]):
            raise ValueError('EOS stopping is inconsistent')
    else:
        raise ValueError('unexpected stopping rule')


@contextmanager
def server(args, cfg):
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, encoding='utf-8').strip()
    if git('status', '--porcelain'): raise ValueError('source must be clean')
    head = git('rev-parse', 'HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/experiment/continuing-agent'],
                   cwd=ROOT, check=True, capture_output=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    try:
        catalog_path = ROOT/'configs/stock-speculation-artifacts.json'
        checked = stock.verify_catalog(read(catalog_path), args.models, args.binary,
                                       ['target'] + (['draft05'] if cfg['k'] else []))
        cmd = previous.command(args.binary, checked, cfg, args.port)
        env, effective, removed = previous.child_environment(os.environ, cfg['threshold'], 0)
        stock.write_json(out/'manifest.json', {'head':head, 'configuration':cfg, 'command':cmd,
            'kind':args.kind, 'condition':args.condition, 'repeat':args.repeat,
            'effective_runtime_environment':effective, 'removed_override_names':removed,
            'source_sha256':{p:stock.digest(ROOT/p) for p in SOURCES}, 'artifacts':checked,
            'python':os.sys.version, 'retained_source':str(args.retained) if args.retained else None})
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', args.port))
        base = f'http://127.0.0.1:{args.port}'
        with (out/'server.log').open('xb') as log, stock.managed_process(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=args.binary) as proc:
            with stock.Resources(proc.pid, out/'resources.jsonl') as resources:
                start=time.perf_counter()
                for attempt in range(2400):
                    if proc.poll() is not None: raise RuntimeError('server exited at startup')
                    try:
                        if stock.request(base, '/health', timeout=2, receipt=out/f'health-{attempt}').get('status')=='ok':
                            break
                    except (OSError, urllib.error.HTTPError): pass
                    if time.perf_counter()-start>600: raise TimeoutError('server startup')
                    time.sleep(.25)
                placement=stock.validate_placement((out/'server.log').read_text(encoding='utf-8',errors='replace'),
                                                  cfg['ngl'], 'draft05' if cfg['k'] else None)
                stock.write_json(out/'startup.json', {'placement':placement,
                    'props':stock.request(base,'/props',receipt=out/'props')})
                if not previous.resource_check(resources)['pass']: raise RuntimeError('startup resource bound')
                yield base, resources
            state=previous.resource_check(resources)
            stock.write_json(out/'completion.json', {'complete':state['pass'],'resources':state})
            if not state['pass']: raise RuntimeError('resource bound')
    except BaseException as exc:
        stock.write_json(out/'failure.json', {'type':type(exc).__name__,'message':str(exc)})
        raise


def replay(args):
    fixture=read(ROOT/'data/committed-replay.json')
    stage=args.kind.split('-')[1]
    cfg=settings(args.kind,args.condition)
    with server(args,cfg) as (base,resources), (args.output/'rows.jsonl').open('x',encoding='utf-8') as ledger:
        for key,case in fixture['cases'].items():
            if case['stage']!=stage: continue
            for i,expected in enumerate(case['tokens']):
                prefix=case['prompt']+case['tokens'][:i]
                payload=stock.completion_payload(prefix,1)
                payload.update(cache_prompt=i>0, n_probs=2, post_sampling_probs=False)
                label=f'{key}-{i:04d}'
                begin=time.perf_counter()
                response=stock.request(base,'/completion',payload,timeout=600,receipt=args.output/label)
                end=time.perf_counter()
                stock.write_json(args.output/f'{label}.interval.json',{'begin':begin,'end':end})
                check_output(response,1)
                cache=response['timings']['cache_n']; evaluated=response['timings']['prompt_n']
                valid_cache=(cache==len(prefix)-1 and evaluated==1) if i else (cache==0 and evaluated==len(prefix))
                top=response['completion_probabilities'][0]['top_logprobs']
                actual=response['tokens'][0]
                argmax=actual in [e['id'] for e in top if e['logprob']==max(x['logprob'] for x in top)]
                state=previous.resource_check(resources)
                row={'case':key,'index':i,'prefix_length':len(prefix),'expected':expected,'actual':actual,'begin':begin,'end':end,
                     'matches':expected==actual,'cache_pass':valid_cache,'argmax_pass':argmax,
                     'top_two':top,'response_sha256':stock.digest(args.output/f'{label}.body'),'resources':state}
                ledger.write(json.dumps(row)+'\n'); ledger.flush()
                if not valid_cache or not argmax or not state['pass']: raise RuntimeError('replay apparatus check')
                if i%32==0 or i==len(case['tokens'])-1:
                    print(json.dumps({'case':key,'position':i,'match':row['matches']}),flush=True)


def suffix_text(turn, stopped_eos):
    return ('' if stopped_eos else '<|im_end|>')+'\n<|im_start|>user\n'+turn['text']+'<|im_end|>\n<|im_start|>assistant\n'


def agent(args, *, configuration=None, server_factory=None):
    """Reuse the frozen conversation construction with an explicitly supplied server."""
    cfg=settings(args.kind,args.condition) if configuration is None else configuration
    launch=server if server_factory is None else server_factory
    fixture=read(ROOT/'data/committed-replay.json')
    turns=read(ROOT/'data/continuing-agent-turns.json')
    initial=fixture['cases'][turns['initial_case']]
    reset=args.kind=='agent-reset'
    retained=[]
    if reset:
        if not args.retained or not read(args.retained/'completion.json')['complete']:
            raise ValueError('reset requires complete paired retained run')
        manifest=read(args.retained/'manifest.json')
        if manifest['kind']!='agent-retained' or manifest['condition']!=args.condition or manifest['repeat']!=args.repeat:
            raise ValueError('reset source does not match its declared pair')
        retained=[json.loads(x) for x in (args.retained/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(retained)!=6: raise ValueError('retained turn coverage')
    with launch(args,cfg) as (base,resources), (args.output/'rows.jsonl').open('x',encoding='utf-8') as ledger:
        # Same untimed short calibration before every fresh process's conversation.
        warm=fixture['cases']['short-code-cache']['prompt']
        stock.request(base,'/completion',stock.completion_payload(warm,32),receipt=args.output/'warmup')
        prefix=list(initial['prompt']); previous_response=None
        for i in range(6):
            key='initial' if i==0 else turns['turns'][i-1]['id']
            cap=128 if i==0 else turns['turns'][i-1]['cap']
            if reset:
                original=args.retained/f'turn-{i}.request.json'
                payload=json.loads(read(original)['body_utf8'])
                prefix=payload['prompt']
                stock.write_json(args.output/f'retained-input-{i}.json', {'sha256':stock.digest(original),'payload':payload})
            elif i:
                extension=suffix_text(turns['turns'][i-1], previous_response['stop_type']=='eos')
                suffix=stock.request(base,'/tokenize',{'content':extension,'add_special':False,'parse_special':True},
                                     receipt=args.output/f'suffix-{i}')['tokens']
                prefix=prefix+previous_response['tokens']+suffix
            if len(prefix)+cap+cfg['k']>cfg['context']: raise ValueError('conversation exceeds context bound')
            payload=stock.completion_payload(prefix,cap)
            payload.update(stream=True,cache_prompt=(not reset and i>0))
            log_start=(args.output/'server.log').stat().st_size
            response,meta=streaming_request(base,payload,args.output/f'turn-{i}')
            stock.write_json(args.output/f'response-{i}.json',response)
            check_output(response,cap)
            time.sleep(.05)
            with (args.output/'server.log').open('rb') as log:
                log.seek(log_start); segment=log.read().decode('utf-8',errors='replace')
            acceptance=stock.parse_acceptance('\n'.join(l for l in segment.splitlines() if 'new n_tokens =' not in l))
            summary=stock.acceptance_summary(acceptance,response['timings'])
            state=previous.resource_check(resources)
            row={'turn':i,'id':key,'prompt_tokens':len(prefix),'response':response,'http':meta,
                 'log_start':log_start,'log_end':(args.output/'server.log').stat().st_size,
                 'acceptance':acceptance,'acceptance_summary':summary,'resources':state,
                 'paired_retained_ids_match':response['tokens']==retained[i]['response']['tokens'] if reset else None}
            ledger.write(json.dumps(row)+'\n'); ledger.flush()
            if not summary['consistent'] or not state['pass']: raise RuntimeError('acceptance or resources')
            cache=response['timings']['cache_n']; evaluated=response['timings']['prompt_n']
            if cache+evaluated!=len(prefix) or (reset or i==0) and cache!=0:
                raise RuntimeError('prompt accounting/reset check')
            # A retained miss is an experimental result, not a reason to replace the run.
            print(json.dumps({'kind':args.kind,'condition':args.condition,'repeat':args.repeat,'turn':key,
                 'seconds':meta['end']-meta['begin'],'cache':cache,'new':evaluated}),flush=True)
            previous_response=response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('models','binary','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind',choices=['replay-short','replay-long','agent-retained','agent-reset'],required=True)
    parser.add_argument('--condition',choices=['default','offload'],default='offload')
    parser.add_argument('--repeat',type=int,choices=[1,2],default=1)
    parser.add_argument('--retained',type=Path)
    parser.add_argument('--port',type=int,default=8104)
    args=parser.parse_args()
    for name in ('models','binary','output','retained'):
        if getattr(args,name): setattr(args,name,getattr(args,name).resolve())
    if args.kind.startswith('replay'):
        if args.repeat!=1 or args.condition!='offload': raise ValueError('replay has one declared run per context')
        replay(args)
    else: agent(args)


if __name__=='__main__': main()

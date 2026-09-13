"""Original-cap continuing turns on the fixed v0.17 memory boundary."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error

import stock_benchmark as stock
import verification_offload as old
import stock_placement as placement
import continuing_agent as conversation

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['scripts/attention_agent.py','scripts/analyze_attention_agent.py',
         'scripts/continuing_agent.py','scripts/analyze_continuing.py',
         'scripts/stock_placement.py','scripts/analyze_stock_placement.py',
         'scripts/stock_benchmark.py','scripts/verification_offload.py',
         'data/committed-replay.json','data/continuing-agent-turns.json',
         'configs/stock-speculation-artifacts.json','docs/attention-agent-protocol.md']
read=conversation.read


def settings(condition):
    if condition not in ('whole','attention'): raise ValueError('unregistered layout')
    return placement.configuration('timed',2,32 if condition=='attention' else 0)


def matrix():
    return [(c,r,m) for r,order in [(1,['attention','whole']),(2,['whole','attention'])]
            for c in order for m in ['retained','reset']]


def identity(condition,repeat,mode):
    return f'{condition}-r{repeat}-{mode}'


@contextmanager
def server(args,cfg):
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise ValueError('requires clean source')
    head=git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/experiment/attention-agent'],
                   cwd=ROOT,check=True,capture_output=True)
    args.output.mkdir(parents=True,exist_ok=False)
    out=args.output
    stock.write_json(out/'attempt.json',{'head':head,'kind':args.kind,'repeat':args.repeat,'configuration':cfg,
        'started_monotonic':time.perf_counter(),'source_sha256':{p:stock.digest(ROOT/p) for p in SOURCES},
        'condition':args.condition,'retained_source':str(args.retained) if args.retained else None})
    phase='preflight'
    try:
        checked=stock.verify_catalog(read(ROOT/'configs/stock-speculation-artifacts.json'),args.models,args.binary,['target','draft05'])
        cmd=placement.command(args.binary,checked,cfg,args.port)
        env,effective,removed=placement.environment(cfg['threshold'],cfg['scheduler_debug'])
        stock.write_json(out/'manifest.json',{'head':head,'kind':args.kind,'repeat':args.repeat,'configuration':cfg,
            'command':cmd,'effective_runtime_environment':effective,'removed_override_names':removed,
            'artifacts':checked,'source_sha256':{p:stock.digest(ROOT/p) for p in SOURCES},
            'python':os.sys.version,'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','psutil']},
            'condition':args.condition,'retained_source':str(args.retained) if args.retained else None,
            'study':'attention-agent-v1'})
        phase='port'
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            probe.bind(('127.0.0.1',args.port))
        base=f'http://127.0.0.1:{args.port}'
        phase='launch'
        with (out/'server.log').open('xb') as log,stock.managed_process(cmd,stdout=log,stderr=subprocess.STDOUT,
                env=env,cwd=args.binary) as proc:
            with stock.Resources(proc.pid,out/'resources.jsonl') as resources:
                phase='startup'
                start=time.perf_counter()
                for attempt in range(2400):
                    if proc.poll() is not None: raise RuntimeError('native startup exit')
                    try:
                        if stock.request(base,'/health',timeout=2,receipt=out/f'health-{attempt}').get('status')=='ok': break
                    except (OSError,urllib.error.HTTPError): pass
                    if time.perf_counter()-start>600: raise TimeoutError('startup deadline')
                    time.sleep(.25)
                evidence=placement.placement_evidence((out/'server.log').read_text(encoding='utf-8',errors='replace'),cfg)
                stock.write_json(out/'startup.json',{'placement':evidence,
                    'props':stock.request(base,'/props',receipt=out/'props')})
                phase='startup-validation'
                if not evidence['kv_pass'] or evidence['host_override_pass'] is False or not evidence['actual_host_pass']:
                    raise RuntimeError('placement or host-buffer evidence differs')
                if not old.resource_check(resources)['pass']: raise RuntimeError('startup resource bound')
                phase='requests'
                yield base,resources
            phase='completion'
            state=old.resource_check(resources)
            stock.write_json(out/'completion.json',{'complete':state['pass'],'resources':state,'ended_monotonic':time.perf_counter()})
            if not state['pass']: raise RuntimeError('resource bound')
    except BaseException as exc:
        stock.write_json(out/'failure.json',{'phase':phase,'type':type(exc).__name__,'message':str(exc),'ended_monotonic':time.perf_counter()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['models','binary','output']: p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--condition',choices=['whole','attention'])
    p.add_argument('--repeat',type=int,choices=[1,2])
    p.add_argument('--kind',choices=['agent-retained','agent-reset'])
    p.add_argument('--retained',type=Path)
    p.add_argument('--port',type=int,choices=[8106],default=8106)
    a=p.parse_args()
    for n in ['models','binary','output','retained']:
        if getattr(a,n): setattr(a,n,getattr(a,n).resolve())
    if a.worker:
        if not a.condition or not a.repeat or not a.kind: raise ValueError('incomplete worker identity')
        if (a.kind=='agent-reset') != bool(a.retained): raise ValueError('retained/reset pairing')
        conversation.agent(a,configuration=settings(a.condition),server_factory=server)
        return
    if any([a.condition,a.repeat,a.kind,a.retained]): raise ValueError('matrix takes no worker settings')
    a.output.mkdir(parents=True,exist_ok=False)
    for c,r,m in matrix():
        cmd=[os.sys.executable,str(Path(__file__).resolve()),'--worker','--models',str(a.models),
             '--binary',str(a.binary),'--output',str(a.output/identity(c,r,m)),
             '--condition',c,'--repeat',str(r),'--kind','agent-'+m]
        if m=='reset': cmd+=['--retained',str(a.output/identity(c,r,'retained'))]
        subprocess.run(cmd,check=True)
    from analyze_attention_agent import analyze
    stock.write_json(a.output/'analysis.json',analyze(a.output))


if __name__=='__main__': main()

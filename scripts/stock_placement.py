"""Prospective stock threshold and attention-resident/host-FFN measurements."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error

import stock_benchmark as stock
import verification_offload as old
import continuing_agent as transport

ROOT=Path(__file__).resolve().parents[1]
THRESHOLDS=[8,4,2]
ALLOCATIONS=[36,34,32,30]
SOURCES=['scripts/stock_placement.py','scripts/stock_benchmark.py','scripts/verification_offload.py',
         'scripts/continuing_agent.py','scripts/analyze_stock_placement.py','scripts/stock_placement_matrix.py',
         'scripts/analyze_continuing.py','data/committed-replay.json',
         'configs/stock-speculation-artifacts.json','docs/stock-placement-protocol.md']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def configuration(kind,threshold,cold_ffns):
    if threshold not in THRESHOLDS or cold_ffns not in [0,*ALLOCATIONS]:
        raise ValueError('outside registered setting grid')
    cfg=old.configuration('long','offload-k16')
    cfg.update(threshold=threshold,cold_ffns=cold_ffns,ngl=65 if cold_ffns else 38,
               scheduler_debug=2 if kind=='mechanism' else 0,
               verbosity=6 if kind in ('allocation','mechanism') else 4)
    if kind=='threshold' and cold_ffns: raise ValueError('threshold study keeps whole-layer placement')
    if kind=='allocation' and not cold_ffns: raise ValueError('allocation study needs host FFNs')
    return cfg


def environment(threshold,debug):
    if threshold not in THRESHOLDS: raise ValueError('unregistered threshold')
    env,effective,removed=old.child_environment(os.environ,8,debug)
    effective['GGML_OP_OFFLOAD_MIN_BATCH']=str(threshold)
    env.update(effective)
    return env,effective,removed


def command(binary,checked,cfg,port):
    result=old.command(binary,checked,cfg,port)
    result[result.index('--verbosity')+1]=str(cfg['verbosity'])
    if cfg['cold_ffns']: result+=['--n-cpu-ffn',str(cfg['cold_ffns'])]
    return result


def placement_evidence(log,cfg):
    stock.validate_placement(log,cfg['ngl'],'draft05')
    kv=[[backend,float(size)] for backend,size in re.findall(r'llama_kv_cache:\s+(\S+) KV buffer size =\s+([\d.]+) MiB',log)]
    overrides=[[int(layer),kind,buffer] for layer,kind,buffer in re.findall(
        r'tensor blk\.(\d+)\.ffn_(up|down|gate)\.weight.*buffer type overridden to (\S+)',log)]
    count=cfg['cold_ffns']
    wanted={(n,k) for n in range(count) for k in ('up','down','gate')}
    # Override lines are DEBUG; the allocation/mechanism processes carry that evidence.
    override_pass=(len(overrides)==3*count and {(n,k) for n,k,b in overrides}==wanted and
                   all(b=='CUDA_Host' for n,k,b in overrides)) if cfg['verbosity']>=5 else None
    expected_kv=[['CUDA0',2448.0],['CUDA0',114.77]] if count else [
        ['CPU',1032.75],['CUDA0',1415.25],['CUDA0',114.77]]
    kv_pass=kv==expected_kv
    buffers=[[backend,float(size)] for backend,size in re.findall(r'(\S+) model buffer size =\s+([\d.]+) MiB',log)]
    actual_host_pass=(len(buffers)==4 and [b for b,s in buffers]==['CUDA0','CUDA_Host','CUDA0','CUDA_Host']
                      and all(s>0 for b,s in buffers))
    return {'kv_allocations_mib':kv,'kv_pass':kv_pass,'overrides':overrides,
            'host_override_pass':override_pass,'model_buffers_mib':buffers,'actual_host_pass':actual_host_pass,
            'intended_attention_layers_gpu':64 if count else 37}


@contextmanager
def server(args,cfg):
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise ValueError('requires clean source')
    head=git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/experiment/stock-placement'],
                   cwd=ROOT,check=True,capture_output=True)
    args.output.mkdir(parents=True,exist_ok=False)
    out=args.output
    stock.write_json(out/'attempt.json',{'head':head,'kind':args.kind,'repeat':args.repeat,'configuration':cfg,
        'started_monotonic':time.perf_counter(),'source_sha256':{p:stock.digest(ROOT/p) for p in SOURCES},
        'decision_sha256':stock.digest(args.decision) if args.decision else None})
    if args.decision: stock.write_json(out/'decision.json',read(args.decision))
    phase='preflight'
    try:
        checked=stock.verify_catalog(read(ROOT/'configs/stock-speculation-artifacts.json'),args.models,args.binary,['target','draft05'])
        cmd=command(args.binary,checked,cfg,args.port)
        env,effective,removed=environment(cfg['threshold'],cfg['scheduler_debug'])
        stock.write_json(out/'manifest.json',{'head':head,'kind':args.kind,'repeat':args.repeat,'configuration':cfg,
            'command':cmd,'effective_runtime_environment':effective,'removed_override_names':removed,
            'artifacts':checked,'source_sha256':{p:stock.digest(ROOT/p) for p in SOURCES},
            'python':os.sys.version,'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','psutil']},
            'decision_sha256':stock.digest(args.decision) if args.decision else None,
            'study':'stock-placement-v1'})
        if args.decision: stock.write_json(out/'decision.json',read(args.decision))
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
                placement=placement_evidence((out/'server.log').read_text(encoding='utf-8',errors='replace'),cfg)
                stock.write_json(out/'startup.json',{'placement':placement,
                    'props':stock.request(base,'/props',receipt=out/'props')})
                phase='startup-validation'
                if not placement['kv_pass'] or placement['host_override_pass'] is False or not placement['actual_host_pass']:
                    raise RuntimeError('placement or host-buffer evidence differs')
                if not old.resource_check(resources)['pass']: raise RuntimeError('startup resource bound')
                phase='requests'
                yield base,resources
            phase='completion'
            state=old.resource_check(resources)
            stock.write_json(out/'completion.json',{'complete':state['pass'],'resources':state,'ended_monotonic':time.perf_counter(),
                'allocation_eligible':state['pass'] and state['gpu_peak']<=14800*2**20})
            if not state['pass']: raise RuntimeError('resource bound')
    except BaseException as exc:
        stock.write_json(out/'failure.json',{'phase':phase,'type':type(exc).__name__,'message':str(exc),'ended_monotonic':time.perf_counter()})
        raise


def measure_one(base,out,label,tokens,cap,cache,resources,ledger):
    payload=stock.completion_payload(tokens,cap)
    payload.update(stream=True,cache_prompt=cache)
    log_start=(out/'server.log').stat().st_size
    response,meta=transport.streaming_request(base,payload,out/label)
    stock.write_json(out/f'{label}-response.json',response)
    transport.check_output(response,cap)
    time.sleep(.05)
    with (out/'server.log').open('rb') as stream:
        stream.seek(log_start); segment=stream.read().decode('utf-8',errors='replace')
    events=stock.parse_acceptance('\n'.join(l for l in segment.splitlines() if 'new n_tokens =' not in l))
    summary=stock.acceptance_summary(events,response['timings'])
    state=old.resource_check(resources)
    row={'id':label,'prompt_tokens':len(tokens),'response':response,'http':meta,'acceptance':events,
         'acceptance_summary':summary,'resources':state,'log_start':log_start,'log_end':(out/'server.log').stat().st_size}
    ledger.write(json.dumps(row)+'\n'); ledger.flush()
    timing=response['timings']
    if timing['cache_n']+timing['prompt_n']!=len(tokens) or not cache and timing['cache_n']:
        raise RuntimeError('prompt/reset accounting')
    if not state['pass'] or not summary['consistent']: raise RuntimeError('resource or acceptance accounting')
    print(json.dumps({'output':out.name,'case':label,'seconds':meta['end']-meta['begin'],
                     'new':timing['prompt_n'],'cache':timing['cache_n'],'emitted':len(response['tokens'])}),flush=True)
    return response


def execute(args):
    cfg=configuration(args.kind,args.threshold,args.cold_ffns)
    if args.kind!='threshold':
        if not args.decision: raise ValueError('requires prior-stage decision')
        decision=read(args.decision)
        if decision['threshold']!=args.threshold: raise ValueError('threshold decision mismatch')
        if args.kind not in ('allocation',) and args.cold_ffns and decision.get('cold_ffns')!=args.cold_ffns:
            raise ValueError('allocation decision mismatch')
    fixture=read(ROOT/'data/committed-replay.json')['cases']
    with server(args,cfg) as (base,resources), (args.output/'rows.jsonl').open('x',encoding='utf-8') as ledger:
        measure_one(base,args.output,'warmup',fixture['short-code-cache']['prompt'],32,False,resources,ledger)
        keys=['long-code-cache'] if args.kind in ('allocation','mechanism') else ['long-code-cache','long-data-audit']
        if args.repeat==2: keys.reverse()
        for key in keys:
            cap=32 if args.kind in ('allocation','mechanism') else 128
            measure_one(base,args.output,key,fixture[key]['prompt'],cap,False,resources,ledger)



def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['models','binary','output']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',choices=['threshold','allocation','mechanism','placement'],required=True)
    p.add_argument('--threshold',type=int,choices=THRESHOLDS,required=True)
    p.add_argument('--cold-ffns',type=int,choices=[0,*ALLOCATIONS],default=0)
    p.add_argument('--repeat',type=int,choices=[1,2],default=1)
    p.add_argument('--decision',type=Path)
    p.add_argument('--port',type=int,default=8105)
    a=p.parse_args()
    for name in ['models','binary','output','decision']:
        if getattr(a,name): setattr(a,name,getattr(a,name).resolve())
    if a.kind in ('allocation','mechanism') and a.repeat!=1: raise ValueError('one diagnostic attempt per condition')
    execute(a)


if __name__=='__main__': main()

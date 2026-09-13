"""Reconstruct the prospective stock-placement matrix from raw receipts."""
import argparse
import json
import math
from pathlib import Path
import re

import stock_placement as study
import stock_benchmark as stock
import analyze_continuing as prior

read=study.read
require=prior.require


def identity(kind,threshold,cold,repeat):
    return f'{kind}-t{threshold}-f{cold}-r{repeat}'


def graph_evidence(log):
    graphs=[]; current=[]
    for node in study.old.scheduler_summary(log)['nodes']:
        if current and node['node']<=current[-1]['node']:
            graphs.append(current); current=[]
        current.append(node)
    if current: graphs.append(current)
    targets=[]
    for graph in graphs:
        attention={int(m[1]):n['assignment'] for n in graph if n['operation']=='FLASH_ATTN'
                   and (m:=re.search(r'cache_k_l(\d+)',n['raw']))}
        if set(attention)!=set(range(64)): continue
        ffns=[{'layer':int(m[2]),'projection':'down' if m[1]=='out' else m[1],'backend':n['assignment']}
              for n in graph if n['operation']=='MUL_MAT'
              and (m:=re.search(r'^ffn_(up|out|gate)-(\d+)\b',n['tensor']))]
        targets.append({'attention':attention,'ffn':ffns,'ffn_complete':len(ffns)==192 and
                        {(r['layer'],r['projection']) for r in ffns}=={(i,k) for i in range(64) for k in ['up','gate','down']}})
    return {'target_graphs':targets,'all_target_attention_cuda':bool(targets) and all(
        g['ffn_complete'] and all(b.startswith('CUDA') for b in g['attention'].values()) for g in targets),
        'limitation':'Graph assignments are not executed operation counts or transferred bytes.'}


def attempt_identity(run):
    m=read(run/'attempt.json'); cfg=m['configuration']
    require(run.name==identity(m['kind'],cfg['threshold'],cfg['cold_ffns'],m['repeat']),'matrix identity')
    require(m['configuration']==study.configuration(m['kind'],cfg['threshold'],cfg['cold_ffns']),'configuration')
    require(set(m['source_sha256'])==set(study.SOURCES),'source coverage')
    require(all(stock.digest(study.ROOT/p)==h for p,h in m['source_sha256'].items()),'source hash')
    require(math.isfinite(m['started_monotonic']),'attempt start')
    if m['kind']!='threshold': require(stock.digest(run/'decision.json')==m['decision_sha256'],'decision hash')
    return m


def audit(run):
    attempt=attempt_identity(run)
    m=read(run/'manifest.json'); cfg=m['configuration']
    require(all(m[k]==attempt[k] for k in ['head','kind','repeat','configuration','source_sha256','decision_sha256']),'attempt/manifest mismatch')
    require(m['command']==study.command(Path(m['command'][0]).parent,m['artifacts'],cfg,8105),'command')
    require(m['effective_runtime_environment']=={'LLAMA_TRACE':'1','GGML_OP_OFFLOAD_MIN_BATCH':str(cfg['threshold']),
            'GGML_SCHED_DEBUG':str(cfg['scheduler_debug'])},'environment')
    if m['kind']!='threshold': require(stock.digest(run/'decision.json')==m['decision_sha256'],'decision hash')
    require(not (run/'failure.json').exists() and read(run/'completion.json')['complete'],'incomplete native run')
    log=(run/'server.log').read_bytes()
    placement=study.placement_evidence(log.decode('utf-8',errors='replace'),cfg)
    require(placement==read(run/'startup.json')['placement'],'startup placement evidence')
    require(placement['kv_pass'] and placement['host_override_pass'] is not False and placement['actual_host_pass'],'placement')
    resources=prior.lines(run/'resources.jsonl')
    require(resources and all('error' not in r and r['gpu']['used']<=15000*2**20 and
            r['host']['available']>=2*2**30 for r in resources),'resource bounds')
    fixture=read(study.ROOT/'data/committed-replay.json')['cases']
    keys=['long-code-cache'] if m['kind'] in ('allocation','mechanism') else ['long-code-cache','long-data-audit']
    if m['repeat']==2: keys.reverse()
    keys=['warmup',*keys]
    rows=prior.lines(run/'rows.jsonl'); require([r['id'] for r in rows]==keys,'request coverage')
    result=[]; intervals=[]
    for row in rows:
        key=row['id']; cap=32 if key=='warmup' or m['kind'] in ('allocation','mechanism') else 128
        case=fixture['short-code-cache' if key=='warmup' else key]
        expected=stock.completion_payload(case['prompt'],cap); expected.update(stream=True,cache_prompt=False)
        require(prior.payload(run/f'{key}.request.json')==expected,'input or decoding differs')
        events=prior.sse_from_raw(run/f'{key}.body')
        require(events==read(run/f'{key}.events.json'),'SSE mismatch')
        response=study.transport.parse_sse(events); study.transport.check_output(response,cap)
        require(response==read(run/f'{key}-response.json')==row['response'],'response mismatch')
        meta=read(run/f'{key}.http.json'); prior.timing_interval(meta['begin'],meta['end'])
        require(meta==row['http'] and meta['complete'] and meta['status']==200,'HTTP completion')
        stamps=meta['event_arrivals']
        require(len(stamps)==len(events) and all(math.isfinite(x) for x in stamps) and
                stamps==sorted(stamps) and meta['begin']<=meta['headers_at']<=stamps[0]<=stamps[-1]<=meta['end'],'SSE times')
        require(meta['first_token_at']==next((t for t,e in zip(stamps,events) if e.get('tokens')),None),'TTFT')
        intervals.append((meta['begin'],meta['end']))
        segment=log[row['log_start']:row['log_end']].decode('utf-8',errors='replace')
        accepted=stock.parse_acceptance('\n'.join(l for l in segment.splitlines() if 'new n_tokens =' not in l))
        summary=stock.acceptance_summary(accepted,response['timings'])
        require(accepted==row['acceptance'] and summary==row['acceptance_summary'] and summary['consistent'],'acceptance')
        timing=response['timings']
        require(timing['cache_n']==0 and timing['prompt_n']==len(case['prompt']),'reset prompt count')
        validate_native_timing(timing,len(response['tokens']),meta['end']-meta['begin'])
        result.append({'case':key,'emitted':len(response['tokens']),'tokens':response['tokens'],
            'reference_match':response['tokens']==case['tokens'] if cap==128 else None,
            'request_s':meta['end']-meta['begin'],'ttft_s':meta['first_token_at']-meta['begin'],
            'timings':timing,'acceptance':accepted,'acceptance_summary':summary,'stop_type':response['stop_type']})
    prior.resource_coverage(run,intervals)
    completion=read(run/'completion.json')
    prior.timing_interval(attempt['started_monotonic'],completion['ended_monotonic'],3600)
    process_bounds([attempt['started_monotonic'],completion['ended_monotonic']],intervals,
                   [r['monotonic'] for r in resources])
    require(completion['allocation_eligible']==(completion['resources']['gpu_peak']<=14800*2**20),'allocation rule')
    return {'name':run.name,'head':m['head'],'kind':m['kind'],'repeat':m['repeat'],'configuration':cfg,
            'placement':placement,'resources':completion['resources'],'allocation_eligible':completion['allocation_eligible'],
            'interval':[attempt['started_monotonic'],completion['ended_monotonic']],
            'rows':result,'graph':graph_evidence(log.decode('utf-8',errors='replace')) if m['kind']=='mechanism' else None}


def validate_native_timing(timing,emitted,request_s):
    require(all(type(timing[k]) is int and timing[k]>=0 for k in ['prompt_n','cache_n','predicted_n','draft_n','draft_n_accepted']),
            'integral native counters')
    require(timing['predicted_n']==emitted,'native emitted count')
    require(all(type(timing[k]) in (int,float) and math.isfinite(timing[k]) and 0<timing[k]<1800000
                for k in ['prompt_ms','predicted_ms']),'native timing')
    require(timing['prompt_ms']+timing['predicted_ms']<=request_s*1000+100,'native timing exceeds HTTP interval')


def serial_order(runs):
    require(all(a['interval'][1]<=b['interval'][0] for a,b in zip(runs,runs[1:])),'registered serial order')


def process_bounds(bounds,requests,samples):
    require(all(bounds[0]<=begin<=end<=bounds[1] for begin,end in requests),'request outside process')
    require(all(a[1]<=b[0] for a,b in zip(requests,requests[1:])),'request order')
    require(all(bounds[0]<=t<=bounds[1] for t in samples),'resource outside process')



def pooled(runs):
    rows=[row for r in runs for row in r['rows'] if row['case']!='warmup']
    require(rows,'empty score')
    ms=sum(r['timings']['predicted_ms'] for r in rows); emitted=sum(r['emitted'] for r in rows)
    return {'requests':len(rows),'emitted':emitted,'decode_ms_per_emitted':ms/emitted,
        'native_steps_per_s':1000*sum(max(0,r['timings']['predicted_n']-1) for r in rows)/ms,
        'emitted_per_request_s':emitted/sum(r['request_s'] for r in rows),
        'mean_request_s':sum(r['request_s'] for r in rows)/len(rows),
        'mean_prefill_ms':sum(r['timings']['prompt_ms'] for r in rows)/len(rows),
        'reference_matches':sum(r['reference_match'] is True for r in rows),
        'gpu_peak_bytes':max(r['resources']['gpu_peak'] for r in runs)}


def choose_threshold(scores):
    require(set(scores)=={2,4,8},'complete threshold grid required')
    require(all(math.isfinite(v) and v>0 for v in scores.values()),'positive finite scores')
    best=min(scores.values())
    return max(t for t,v in scores.items() if v<=1.01*best)


def threshold_decision(root):
    runs=[audit(root/identity('threshold',t,0,r)) for r,order in [(1,[8,4,2]),(2,[2,4,8])] for t in order]
    serial_order(runs)
    scores={t:pooled([r for r in runs if r['configuration']['threshold']==t]) for t in study.THRESHOLDS}
    return {'stage':'threshold','threshold':choose_threshold({t:s['decode_ms_per_emitted'] for t,s in scores.items()}),
            'scores':{str(k):v for k,v in scores.items()},
            'inputs':{r['name']:stock.digest(root/r['name']/'rows.jsonl') for r in runs}}


def allocation_decision(root,threshold):
    attempts=[]; eligible=[]; processes=[]
    for n in study.ALLOCATIONS:
        run=root/identity('allocation',threshold,n,1)
        if not run.exists(): break
        require(not (run/'failure.json').exists(),'native failure is unscored; no allocation selection')
        data=audit(run); ok=data['allocation_eligible']
        processes.append(data)
        attempts.append({'cold_ffns':n,'eligible':ok,'resources':data['resources']})
        if not ok: break
        eligible.append(n)
    require(attempts,'no allocation attempts')
    serial_order(processes)
    require(len(attempts)==len(study.ALLOCATIONS) or not attempts[-1]['eligible'],
            'allocation stopped before registered boundary')
    return {'stage':'allocation','threshold':threshold,'cold_ffns':min(eligible) if eligible else None,'attempts':attempts}


def analyze(root):
    td=threshold_decision(root); require(td==read(root/'threshold-decision.json'),'threshold decision')
    ad=allocation_decision(root,td['threshold']); require(ad==read(root/'allocation-decision.json'),'allocation decision')
    expected={identity('threshold',t,0,r) for t in study.THRESHOLDS for r in (1,2)}
    expected.update(identity('allocation',td['threshold'],a['cold_ffns'],1) for a in ad['attempts'])
    runs=[audit(root/n) for n in sorted(expected)]
    for a in ad['attempts']:
        require(read(root/identity('allocation',td['threshold'],a['cold_ffns'],1)/'decision.json')==td,'allocation decision binding')
    for r in runs:
        if r['kind']=='allocation': require(read(root/r['name']/'decision.json')==td,'allocation prior binding')
    placement={}
    if ad['cold_ffns'] is not None:
        names=[identity('mechanism',td['threshold'],n,1) for n in [0,ad['cold_ffns']]]
        mech=[audit(root/n) for n in names]; runs+=mech; expected.update(names)
        for n in names: require(read(root/n/'decision.json')==ad,'mechanism prior binding')
        if mech[1]['graph']['all_target_attention_cuda']:
            names=[identity('placement',td['threshold'],n,r) for r,order in [(1,[0,ad['cold_ffns']]),(2,[ad['cold_ffns'],0])] for n in order]
            timed=[audit(root/n) for n in names]; runs+=timed; expected.update(names)
            serial_order(timed)
            for n in names: require(read(root/n/'decision.json')==ad,'placement prior binding')
            placement={str(n):pooled([r for r in timed if r['configuration']['cold_ffns']==n]) for n in [0,ad['cold_ffns']]}
    require({p.name for p in root.iterdir() if p.is_dir()}==expected,'extra or missing matrix directory')
    require(len({r['head'] for r in runs})==1,'mixed measured source')
    chronology=sorted(runs,key=lambda r:r['interval'][0]); serial_order(chronology)
    phase={'threshold':0,'allocation':1,'mechanism':2,'placement':3}
    order=[phase[read(root/r['name']/'attempt.json')['kind']] for r in chronology]
    require(order==sorted(order),'stage order')
    mechanisms=[r for r in chronology if r.get('kind')=='mechanism']
    if mechanisms: require([r['configuration']['cold_ffns'] for r in mechanisms]==[0,ad['cold_ffns']],'mechanism order')
    return {'threshold':td,'allocation':ad,'placement':placement,'runs':runs}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); stock.write_json(a.output,analyze(a.runs))

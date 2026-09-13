"""Reconstruct bounded replay/cache findings from raw HTTP and native logs."""
import argparse
import json
import math
from pathlib import Path

import continuing_agent as study
import stock_benchmark as stock


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def payload(path):
    return json.loads(study.read(path)['body_utf8'])


def require(value,message):
    if not value: raise ValueError(message)


def matrix_identity(name,manifest):
    if name in ('replay-long','replay-short'):
        expected=(name,'offload',1)
    else:
        condition,repeat,mode=name.split('-')
        expected=('agent-'+mode,condition,int(repeat[1:]))
    require((manifest['kind'],manifest['condition'],manifest['repeat'])==expected,'matrix identity')


def timing_interval(begin,end,limit=1800):
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (begin,end))
            and 0<=end-begin<=limit,'finite bounded timing interval')


def resource_coverage(run,intervals):
    samples=lines(run/'resources.jsonl')
    final=study.read(run/'completion.json')['resources']
    require(final['samples']==len(samples),'resource sample count')
    times=[r['monotonic'] for r in samples]
    require(times and all(math.isfinite(t) for t in times) and times==sorted(times),'resource timestamps')
    require(all(b-a<=2.0 for a,b in zip(times,times[1:])),'resource sample gap')
    for begin,end in intervals:
        timing_interval(begin,end)
        require(times[0]<=begin and times[-1]>=end-2.0,'resource interval coverage')
    require(final['gpu_peak']==max(r['gpu']['used'] for r in samples) and
            final['host_available_min']==min(r['host']['available'] for r in samples),'resource extrema')


def audit_run(run):
    manifest=study.read(run/'manifest.json')
    matrix_identity(run.name,manifest)
    require(study.read(run/'completion.json')['complete'] and not (run/'failure.json').exists(),'run incomplete')
    require(set(manifest['source_sha256'])==set(study.SOURCES),'source identity coverage')
    for name,sha in manifest['source_sha256'].items():
        require(stock.digest(study.ROOT/name)==sha,'source identity: '+name)
    require(manifest['configuration']==study.settings(manifest['kind'],manifest['condition']),'configuration')
    cfg=manifest['configuration']
    expected_command=study.previous.command(Path(manifest['command'][0]).parent,manifest['artifacts'],cfg,8104)
    require(manifest['command']==expected_command,'native command differs')
    require(manifest['effective_runtime_environment']=={
        'LLAMA_TRACE':'1','GGML_OP_OFFLOAD_MIN_BATCH':str(cfg['threshold']),'GGML_SCHED_DEBUG':'0'},'environment')
    resources=lines(run/'resources.jsonl')
    require(resources and all('error' not in r and r['gpu']['used']<=15000*2**20 and
                             r['host']['available']>=2*2**30 for r in resources),'resource bounds')
    stock.validate_placement((run/'server.log').read_text(encoding='utf-8',errors='replace'),cfg['ngl'],
                             'draft05' if cfg['k'] else None)
    return manifest,{'gpu_peak_bytes':max(r['gpu']['used'] for r in resources),
                     'host_available_min_bytes':min(r['host']['available'] for r in resources)}


def raw_response(run,label):
    meta=study.read(run/f'{label}.http.json')
    require(meta['status']==200 and meta['complete'],'HTTP completion')
    require(meta['bytes']==(run/f'{label}.body').stat().st_size,'HTTP byte count')
    return study.read(run/f'{label}.body')


def replay(run):
    manifest,resources=audit_run(run)
    fixture=study.read(study.ROOT/'data/committed-replay.json')['cases']
    stage=manifest['kind'].split('-')[1]
    expected=[(k,i) for k,c in fixture.items() if c['stage']==stage for i in range(len(c['tokens']))]
    ledger=lines(run/'rows.jsonl')
    require([(r['case'],r['index']) for r in ledger]==expected,'replay sequence coverage')
    result=[]; intervals=[]
    for key,i in expected:
        c=fixture[key]; label=f'{key}-{i:04d}'
        req=payload(run/f'{label}.request.json')
        prefix=c['prompt']+c['tokens'][:i]
        wanted=stock.completion_payload(prefix,1)
        wanted.update(cache_prompt=i>0,n_probs=2,post_sampling_probs=False)
        require(req==wanted,'teacher-forced prefix or decoding settings')
        res=raw_response(run,label); study.check_output(res,1)
        interval=study.read(run/f'{label}.interval.json')
        timing_interval(interval['begin'],interval['end'],600)
        intervals.append((interval['begin'],interval['end']))
        cache=res['timings']['cache_n']; new=res['timings']['prompt_n']
        require((cache,new)==((len(prefix)-1,1) if i else (0,len(prefix))),'independent cache state')
        entries=res['completion_probabilities'][0]['top_logprobs']
        maximum=max(e['logprob'] for e in entries)
        require(any(e['id']==res['tokens'][0] and e['logprob']==maximum for e in entries),'independent argmax')
        result.append({'case':key,'index':i,'expected':c['tokens'][i],'actual':res['tokens'][0],
                       'match':c['tokens'][i]==res['tokens'][0],'independent_top_two':entries})
    resource_coverage(run,intervals)
    return {'run':run.name,'head':manifest['head'],'resources':resources,'positions':len(result),
            'matches':sum(r['match'] for r in result),'discrepancies':[r for r in result if not r['match']],
            'by_case':{k:{'positions':sum(r['case']==k for r in result),
                          'matches':sum(r['case']==k and r['match'] for r in result)}
                       for k,c in fixture.items() if c['stage']==stage}}


def sse_from_raw(path):
    events=[]
    for frame in path.read_bytes().replace(b'\r\n',b'\n').split(b'\n\n'):
        data=b'\n'.join(l[6:] for l in frame.splitlines() if l.startswith(b'data: '))
        if data and data!=b'[DONE]': events.append(json.loads(data))
    return events


def agent(run,retained=None,*,audit=None):
    manifest,resources=(audit_run if audit is None else audit)(run)
    reset=manifest['kind']=='agent-reset'
    ledger=lines(run/'rows.jsonl')
    require([r['turn'] for r in ledger]==list(range(6)),'agent sequence coverage')
    turns=study.read(study.ROOT/'data/continuing-agent-turns.json')
    fixture=study.read(study.ROOT/'data/committed-replay.json')['cases']
    prefix=fixture[turns['initial_case']]['prompt']
    log=(run/'server.log').read_bytes()
    results=[]; prev=None; intervals=[]
    if reset:
        require(retained is not None,'paired retained source missing')
        rm=study.read(retained/'manifest.json')
        require(rm['kind']=='agent-retained' and rm['condition']==manifest['condition'] and
                rm['repeat']==manifest['repeat'],'paired retained source identity')
    for i,row in enumerate(ledger):
        cap=128 if i==0 else 64
        if reset:
            prior=retained/f'turn-{i}.request.json'
            prefix=payload(prior)['prompt']
            binding=study.read(run/f'retained-input-{i}.json')
            require(binding=={'sha256':stock.digest(prior),'payload':payload(prior)},'reset input binding')
        elif i:
            sr=raw_response(run,f'suffix-{i}')
            require(payload(run/f'suffix-{i}.request.json')=={'content':study.suffix_text(turns['turns'][i-1],
                prev['stop_type']=='eos'),'add_special':False,'parse_special':True},'authored append')
            prefix=prefix+prev['tokens']+sr['tokens']
        expected=stock.completion_payload(prefix,cap)
        expected.update(stream=True,cache_prompt=not reset and i>0)
        require(payload(run/f'turn-{i}.request.json')==expected,'conversation prefix or settings')
        events=sse_from_raw(run/f'turn-{i}.body')
        require(events==study.read(run/f'turn-{i}.events.json'),'raw SSE differs')
        response=study.parse_sse(events); study.check_output(response,cap)
        require(response==study.read(run/f'response-{i}.json')==row['response'],'assembled response differs')
        meta=study.read(run/f'turn-{i}.http.json')
        require(meta['status']==200 and meta['complete'],'SSE HTTP incomplete')
        stamps=meta['event_arrivals']
        timing_interval(meta['begin'],meta['end'])
        intervals.append((meta['begin'],meta['end']))
        require(all(type(t) in (int,float) and math.isfinite(t) for t in [meta['headers_at'],*stamps]),'finite SSE timestamps')
        require(len(stamps)==len(events) and stamps==sorted(stamps) and
                meta['begin']<=meta['headers_at']<=stamps[0]<=stamps[-1]<=meta['end'],'SSE timing order')
        first_token=next((t for t,e in zip(stamps,events) if e.get('tokens')),None)
        first_text=next((t for t,e in zip(stamps,events) if e.get('content')),None)
        require(first_token==meta['first_token_at'] and first_text==meta['first_text_at'],'TTFT event mismatch')
        segment=log[row['log_start']:row['log_end']].decode('utf-8',errors='replace')
        acceptance=stock.parse_acceptance('\n'.join(l for l in segment.splitlines() if 'new n_tokens =' not in l))
        summary=stock.acceptance_summary(acceptance,response['timings'])
        require(summary['consistent'],'acceptance ledger differs from native counters')
        timing=response['timings']; cache=timing['cache_n']; new=timing['prompt_n']
        for name in ('prompt_ms','predicted_ms'):
            require(type(timing[name]) in (int,float) and math.isfinite(timing[name]) and
                    0<=timing[name]<=1800000,'finite native duration')
        require(cache+new==len(prefix) and (cache==0 if reset or i==0 else True),'cache/reset accounting')
        paired=study.read(retained/f'response-{i}.json')['tokens']==response['tokens'] if reset else None
        results.append({'turn':i,'id':row['id'],'prompt_tokens':len(prefix),'emitted_tokens':len(response['tokens']),
            'cache_n':cache,'prompt_n':new,'draft_forwarded_prompt_positions_source_derived':new,
            'draft_independent_cache_counter':None,'ttft_s':first_token-meta['begin'] if first_token else None,
            'first_text_s':first_text-meta['begin'] if first_text else None,'request_s':meta['end']-meta['begin'],
            'prefill_ms':timing['prompt_ms'],'decode_ms':timing['predicted_ms'],
            'native_predicted_n':timing['predicted_n'],'stop_type':response['stop_type'],
            'acceptance':summary,'paired_ids_match':paired})
        prev=response
    continuing=results[1:]
    resource_coverage(run,intervals)
    return {'run':run.name,'head':manifest['head'],'condition':manifest['condition'],'kind':manifest['kind'],
            'repeat':manifest['repeat'],'resources':resources,'turns':results,
            'continuing_request_s':sum(r['request_s'] for r in continuing),
            'continuing_emitted':sum(r['emitted_tokens'] for r in continuing),
            'continuing_ttft_s':sum(r['ttft_s'] for r in continuing)/len(continuing)}


def analyze(root):
    expected={'replay-long','replay-short'}|{f'{c}-r{r}-{m}' for c in ['offload','default'] for r in [1,2]
                                            for m in ['retained','reset']}
    require({p.name for p in root.iterdir() if p.is_dir()}==expected,'exact run coverage')
    replays=[replay(root/name) for name in ['replay-long','replay-short']]
    agents=[]
    for r in [1,2]:
        for c in ['offload','default']:
            retained=root/f'{c}-r{r}-retained'
            agents.extend([agent(retained),agent(root/f'{c}-r{r}-reset',retained)])
    return {'replay':replays,'agent':agents,'limitations':[
        'Independent replay is an observed trajectory agreement test, not universal speculative state correctness.',
        'One authored continuing conversation; paired reset is identical-input but always follows retained.',
        'Draft prompt processing count is derived from pinned source and target batch accounting; no independent draft KV counter.',
        'Sampled resources do not establish absence of paging between samples.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'analysis output already exists')
    stock.write_json(args.output,analyze(args.runs))

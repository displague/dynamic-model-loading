"""Reconstruct the paired retained/reset placement comparison without inference."""
import argparse
import math
from pathlib import Path

import attention_agent as study
import analyze_continuing as prior
import analyze_stock_placement as placement_audit
import stock_benchmark as stock

read=study.read
require=prior.require


def audit_run(run):
    m=read(run/'manifest.json'); a=read(run/'attempt.json'); cfg=m['configuration']
    prior.matrix_identity(run.name,m)
    require(m['study']=='attention-agent-v1','study identity')
    require(all(m[k]==a[k] for k in ['head','kind','condition','repeat','configuration','source_sha256','retained_source']),
            'attempt/manifest mismatch')
    require(cfg==study.settings(m['condition']),'configuration')
    require(set(m['source_sha256'])==set(study.SOURCES),'source coverage')
    require(all(stock.digest(study.ROOT/p)==h for p,h in m['source_sha256'].items()),'source hash')
    require(m['command']==study.placement.command(Path(m['command'][0]).parent,m['artifacts'],cfg,8106),'command')
    require(m['effective_runtime_environment']=={'LLAMA_TRACE':'1','GGML_OP_OFFLOAD_MIN_BATCH':'2',
                                               'GGML_SCHED_DEBUG':'0'},'environment')
    catalog=read(study.ROOT/'configs/stock-speculation-artifacts.json')
    require(set(m['artifacts'])=={'target','draft05'},'artifact coverage')
    for key,paths in m['artifacts'].items():
        require([Path(p).name for p in paths]==[p['name'] for p in catalog['models'][key]['files']],
                'artifact names differ from verified catalog')
    require(not (run/'failure.json').exists() and read(run/'completion.json')['complete'],'incomplete native run')
    log=(run/'server.log').read_text(encoding='utf-8',errors='replace')
    evidence=study.placement.placement_evidence(log,cfg)
    require(evidence==read(run/'startup.json')['placement'],'startup evidence')
    require(evidence['kv_pass'] and evidence['actual_host_pass'],'actual KV/host buffers')
    samples=prior.lines(run/'resources.jsonl')
    require(samples and all('error' not in r and r['gpu']['used']<=15000*2**20 and
            r['host']['available']>=2*2**30 for r in samples),'resource bounds')
    warm=read(study.ROOT/'data/committed-replay.json')['cases']['short-code-cache']['prompt']
    require(prior.payload(run/'warmup.request.json')==stock.completion_payload(warm,32),'warmup input')
    study.conversation.check_output(prior.raw_response(run,'warmup'),32)
    return m,{'gpu_peak_bytes':max(r['gpu']['used'] for r in samples),
              'host_available_min_bytes':min(r['host']['available'] for r in samples)}


def checked_agent(run,retained):
    result=prior.agent(run,retained,audit=audit_run)
    rows=prior.lines(run/'rows.jsonl'); intervals=[]
    for i,row in enumerate(rows):
        meta=read(run/f'turn-{i}.http.json'); timing=row['response']['timings']
        require(row['http']==meta,'ledger HTTP mismatch')
        placement_audit.validate_native_timing(timing,len(row['response']['tokens']),meta['end']-meta['begin'])
        require(row['acceptance_summary']==result['turns'][i]['acceptance'],'ledger acceptance mismatch')
        require(row['paired_retained_ids_match']==result['turns'][i]['paired_ids_match'],'paired output mismatch')
        intervals.append((meta['begin'],meta['end']))
        turn=result['turns'][i]
        turn['native_steps_per_s']=1000*max(0,timing['predicted_n']-1)/timing['predicted_ms']
        history=retained if retained is not None else run
        turn['new_client_tokens']=None if i==0 else turn['prompt_tokens']-result['turns'][i-1]['prompt_tokens']-len(read(history/f'response-{i-1}.json')['tokens'])
    a=read(run/'attempt.json'); end=read(run/'completion.json')['ended_monotonic']
    prior.timing_interval(a['started_monotonic'],end,3600)
    placement_audit.process_bounds([a['started_monotonic'],end],intervals,
                                  [r['monotonic'] for r in prior.lines(run/'resources.jsonl')])
    result['interval']=[a['started_monotonic'],end]
    result['placement']=read(run/'startup.json')['placement']
    return result


def pooled(runs):
    turns=[t for r in runs for t in r['turns'][1:]]
    require(len(turns)==10,'two repeats of five continuing turns')
    seconds=sum(t['request_s'] for t in turns); emitted=sum(t['emitted_tokens'] for t in turns)
    return {'continuing_requests':len(turns),'mean_five_turn_s':seconds/2,
        'mean_ttft_s':sum(t['ttft_s'] for t in turns)/len(turns),'emitted':emitted,
        'emitted_per_request_s':emitted/seconds,'limit_stops':sum(t['stop_type']=='limit' for t in turns),
        'native_steps_per_s':1000*sum(t['native_predicted_n']-1 for t in turns)/sum(t['decode_ms'] for t in turns),
        'mean_initial_request_s':sum(r['turns'][0]['request_s'] for r in runs)/2,
        'target_reused':sum(t['cache_n'] for t in turns),'target_evaluated':sum(t['prompt_n'] for t in turns),
        'gpu_peak_bytes':max(r['resources']['gpu_peak_bytes'] for r in runs)}


def analyze(root):
    order=study.matrix(); names=[study.identity(*x) for x in order]
    require({p.name for p in root.iterdir() if p.is_dir()}==set(names),'exact native matrix')
    results=[]
    for c,r,mode in order:
        retained=root/study.identity(c,r,'retained') if mode=='reset' else None
        results.append(checked_agent(root/study.identity(c,r,mode),retained))
    require(len({r['head'] for r in results})==1,'one measured source')
    placement_audit.serial_order(results)
    comparisons=[]
    for repeat in [1,2]:
        for mode in ['retained','reset']:
            left=root/study.identity('whole',repeat,mode); right=root/study.identity('attention',repeat,mode)
            for i in range(6):
                comparisons.append({'repeat':repeat,'mode':mode,'turn':i,
                    'same_prompt':prior.payload(left/f'turn-{i}.request.json')['prompt']==prior.payload(right/f'turn-{i}.request.json')['prompt'],
                    'same_output':read(left/f'response-{i}.json')['tokens']==read(right/f'response-{i}.json')['tokens']})
    repeats=[]
    for c in ['whole','attention']:
        for mode in ['retained','reset']:
            a=root/study.identity(c,1,mode); b=root/study.identity(c,2,mode)
            for i in range(6):
                repeats.append({'condition':c,'mode':mode,'turn':i,
                    'same_prompt':prior.payload(a/f'turn-{i}.request.json')['prompt']==prior.payload(b/f'turn-{i}.request.json')['prompt'],
                    'same_output':read(a/f'response-{i}.json')['tokens']==read(b/f'response-{i}.json')['tokens']})
    return {'study':'attention-agent-v1','agent':results,
        'pooled':{c+'-'+m:pooled([r for r in results if r['condition']==c and r['kind']=='agent-'+m])
                  for c in ['whole','attention'] for m in ['retained','reset']},
        'cross_layout':comparisons,'repeat_agreement':repeats,'limitations':[
        'One scripted conversation, two repeats, 128/64 output caps; latency is not task completion.',
        'Each reset uses its paired retained exact prompts; reset always follows retained.',
        'Cross-layout histories may differ; compare prompt identity before describing acceleration.',
        'Draft forwarded prompt positions are source-derived, not an independent draft cache counter.',
        'In-process retention only; target-only disk persistence does not qualify target/draft restart.',
        'No new independent replay; v0.16 nine short discrepancies remain open.',
        'Sampled resources and allocation receipts are not component timers or executed copy counts.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); require(not a.output.exists(),'analysis already exists')
    stock.write_json(a.output,analyze(a.runs))

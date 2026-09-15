"""Short early-versus-late acquisition timing screen, with exact demand repair."""
import argparse
import json
from pathlib import Path
import numpy as np
from .fault_screen import ROOT,supervise
from .debt_analysis import demand
from .retention_screen import worker as base_worker,analyze as base_analyze,audit_pages as packet_audit
from .early_rows import EarlyRows,completion_rows

CONFIG=ROOT/'configs/early-screen.json'
ORDER=[(0,'packet'),(0,'late'),(0,'early'),(1,'early'),(1,'packet'),(1,'late'),(2,'late'),(2,'early'),(2,'packet')]
CONDITIONS=('packet','late','early')


def worker(output):
    return base_worker(output,config_path=CONFIG,hybrid_order=ORDER,bank_type=EarlyRows,
        bank_conditions=CONDITIONS,source_file=__file__)


def audit_pages(pages,arrays,calls,condition,capacity):
    if condition=='packet': return packet_audit(pages,arrays,calls,condition,capacity)
    demand(len(pages)==24*len(calls) and capacity==0,'Wrong anticipation matrix/cache')
    previous=[[] for _ in range(24)]; last_clock=0
    stats=dict(weight_h2d_bytes=0,metadata_h2d_bytes=0,activity_d2h_bytes=0,hits=0,active=0,
        prefill_h2d_bytes=0,decode_h2d_bytes=0)
    for j,r in enumerate(pages):
        step,layer=divmod(j,24); n=len(calls[step]['input_ids']); a=arrays[f'activity.{j+1}']
        demand(a.shape==(n,1024) and a.dtype==np.uint8,'Activity drift')
        active=np.flatnonzero(np.unpackbits(a,axis=1).any(axis=0)).tolist()
        predicted,missing=completion_rows(previous[layer],active); previous[layer]=active
        useful=len(set(predicted)&set(active)); wasted=len(predicted)-useful; loaded=len(predicted)+len(missing)
        demand(r['call']==j+1 and r['layer']==layer and r['tokens']==n and r['condition']==condition,'Call identity changed')
        demand(r['active']==active and r['predicted']==predicted and r['misses']==missing
            and r['useful_prefetched']==useful and r['wasted_prefetched']==wasted,'Noncausal forecast/completion')
        values=dict(weight_h2d_bytes=loaded*8192,metadata_h2d_bytes=loaded*8,activity_d2h_bytes=n*8192+1)
        for k,v in values.items(): demand(r[k]==v,'Traffic mismatch'); stats[k]+=v
        clocks=[r[k] for k in ('started','selection_finished','acquisition_finished','finished')]
        demand(all(np.isfinite(clocks)) and clocks==sorted(clocks) and
            max(last_clock,calls[step]['started'])<=clocks[0]<=clocks[-1]<=calls[step]['finished'],'Clock drift')
        last_clock=clocks[-1]
        demand(all(np.isfinite(r[k]) and r[k]>=0 for k in ('copy_ms','overlap_ms','wait_seconds')) and
            r['overlap_ms']<=r['copy_ms']+.00001,'Invalid event accounting')
        if not predicted: demand(r['copy_ms']==r['overlap_ms']==r['wait_seconds']==0,'Empty copy was timed')
        if condition=='late': demand(r['overlap_ms']==0,'Late transfer cannot overlap earlier fc1 region')
        t=r['event_intervals']; ca,cb,fa,fb=[t[k] for k in ('copy_start_ms','copy_end_ms','fc1_start_ms','fc1_end_ms')]
        demand(all(np.isfinite([ca,cb,fa,fb])) and 0<=ca<=cb and 0<=fa<=fb,'Bad CUDA intervals')
        overlap=max(0.,min(cb,fb)-max(ca,fa)) if condition=='early' else 0.
        demand(r['copy_ms']==cb-ca and r['overlap_ms']==overlap,'CUDA overlap reconstruction changed')
        stats['hits']+=useful; stats['active']+=len(active)
        stats['prefill_h2d_bytes' if step==0 else 'decode_h2d_bytes']+=loaded*8200
    return stats


def decision(conditions):
    p,l,e=[conditions[c] for c in CONDITIONS]
    demand(p['h2d_bytes']>0 and min(c['wall_seconds'] for c in (p,l,e))>0,'Missing denominator')
    return dict(checks=dict(Hfaithfulness=True,Hresources=True,
        Htraffic=e['h2d_bytes']==l['h2d_bytes'] and 4*e['h2d_bytes']<=5*p['h2d_bytes'],
        Htiming=all(20*e['wall_seconds']<=19*c['wall_seconds'] for c in (p,l))),
        h2d_saving=1-e['h2d_bytes']/p['h2d_bytes'],wall_saving=1-e['wall_seconds']/p['wall_seconds'])


def analyze(output):
    result=base_analyze(output,config_relative='configs/early-screen.json',hybrid_order=ORDER,
        bank_conditions=CONDITIONS,page_auditor=audit_pages,decider=decision)
    output=Path(output); episodes=json.loads((output/'episodes.json').read_text()); stats={}
    for r in episodes:
        if r['warmup'] or r['condition'] not in ('early','late'): continue
        s=stats.setdefault(r['condition'],dict(copy_ms=0.,overlap_ms=0.,wait_seconds=0.,wasted_prefetched=0,useful_prefetched=0))
        for row in json.loads((output/r['episode']/'pages.json').read_text()):
            for k in s: s[k]+=row[k]
    e=stats['early']; result['checks']['Hoverlap']=e['copy_ms']>0 and 10*e['overlap_ms']>=e['copy_ms']
    a=result['allocation']; demand(a['early_cuda_bytes']==a['early_pinned_bytes']==8396800
        and a['prefetch_rows']==1024 and a['retained_weight_cache'] is False,'Forecast buffer charge changed')
    result.update(forecasts=stats,event_region_overlap_fraction=e['overlap_ms']/e['copy_ms'] if e['copy_ms'] else 0.,
        decision='eligible_for_separate_followup' if all(result['checks'].values()) else 'stop_this_early_fetch_candidate')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.early_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

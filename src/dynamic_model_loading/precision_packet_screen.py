"""FP16 packet transport against dense streaming and warm resident precision."""
import argparse
import json
from pathlib import Path
import numpy as np
from .fault_screen import ROOT,supervise
from .debt_analysis import demand,read_rows
from .retention_screen import worker as base_worker,analyze as base_analyze,audit_pages as packet_audit
from .precision_rows import PrecisionRows

CONFIG=ROOT/'configs/precision-packet-screen.json'
ORDER=[(0,'stream'),(0,'packet'),(1,'packet'),(1,'stream'),(2,'stream'),(2,'packet')]


def worker(output):
    base_worker(output,config_path=CONFIG,hybrid_order=ORDER,bank_type=PrecisionRows,
        bank_conditions=('stream','packet'),source_file=__file__)


def audit_pages(pages,arrays,calls,condition,capacity):
    if condition=='packet': return packet_audit(pages,arrays,calls,condition,capacity,itemsize=2)
    demand(condition=='stream' and len(pages)==24*len(calls),'Missing dense control rows')
    totals=dict(weight_h2d_bytes=0,metadata_h2d_bytes=0,activity_d2h_bytes=0,hits=0,active=0,
        prefill_h2d_bytes=0,decode_h2d_bytes=0)
    last=0.
    for j,r in enumerate(pages):
        step,layer=divmod(j,24); payload=2048*8192*2
        demand(r['layer']==layer and r['call']==j+1 and r['tokens']==len(calls[step]['input_ids'])
            and r['condition']=='stream' and r['direct_contiguous_copy'] is True,'Wrong dense control')
        demand(r['weight_h2d_bytes']==payload and r['metadata_h2d_bytes']==r['activity_d2h_bytes']==0,'Stream bytes changed')
        clocks=[r[k] for k in ('started','selection_finished','acquisition_finished','finished')]
        demand(all(np.isfinite(clocks)) and clocks==sorted(clocks) and
            max(last,calls[step]['started'])<=clocks[0]<=clocks[-1]<=calls[step]['finished'],'Stream clocks changed')
        last=clocks[-1]; totals['weight_h2d_bytes']+=payload
        totals['prefill_h2d_bytes' if step==0 else 'decode_h2d_bytes']+=payload
    return totals


def decision(conditions):
    p,s,r=[conditions[k] for k in ('packet','stream','resident')]
    demand(min(c['wall_seconds'] for c in (p,s,r))>0 and s['h2d_bytes']>0,'Missing precision denominator')
    return dict(checks=dict(Hfaithfulness=True,Hresources=True,
        Htransport=2*p['h2d_bytes']<=s['h2d_bytes'] and 5*p['wall_seconds']<=4*s['wall_seconds'],
        Hresident_speed=20*p['wall_seconds']<=19*r['wall_seconds']),
        h2d_saving=1-p['h2d_bytes']/s['h2d_bytes'],wall_saving=1-p['wall_seconds']/s['wall_seconds'])


def analyze(output):
    output=Path(output)
    result=base_analyze(output,config_relative='configs/precision-packet-screen.json',hybrid_order=ORDER,
        bank_conditions=('stream','packet'),page_auditor=audit_pages,decider=decision)
    a=result['allocation']; demand(a['parameter_dtype']=='torch.float16' and a['direct_host_contiguous'] is True
        and a['extra_host_linear_bytes']==a['host_weights_bytes']==805306368,'FP16 representation charge changed')
    m=json.loads((output/'manifest.json').read_text())
    demand(m['environment']['fp16_reduced_precision_reduction'] is True,'FP16 arithmetic setting changed')
    rows=json.loads((output/'episodes.json').read_text()); samples=list(read_rows(output/'resources.jsonl')); peaks={}
    for r in rows:
        if r['warmup']: continue
        values=[s['gpu_used'] for s in samples if r['started']<=s['monotonic']<=r['finished']]
        demand(bool(values),'Episode lacks a sampled memory observation')
        peaks[r['condition']]=max(peaks.get(r['condition'],0),max(values))
    useful=all(r['wall_seconds']<=5 for r in rows if not r['warmup'] and r['condition']=='packet')
    result['checks']['Hfootprint']=20*peaks['packet']<=17*peaks['resident'] and useful
    admission=all(result['checks'][k] for k in ('Hfaithfulness','Hresources','Htransport','Hfootprint'))
    result.update(scored_peak_gpu_bytes=peaks,resident_memory_saving=1-peaks['packet']/peaks['resident'],
        useful_packet_latency_passed=useful,reference_dtype='float16',fp32_quality_evaluated=False,
        optimized_q4_compared=False,decision='eligible_for_short_scale_protocol' if admission else 'stop_this_precision_candidate')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.precision_packet_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

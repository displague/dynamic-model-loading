"""CPU-first OPT2.7B FP16 acquisition under a fixed 4800 MiB allowance."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT,supervise
from .debt_analysis import demand,read_rows
from .retention_screen import worker as base_worker,analyze as base_analyze
from .precision_packet_screen import audit_pages as precision_audit
from .precision_rows import PrecisionRows
from .scale_support import audit_allocation,TOTAL,LIMIT

CONFIG=ROOT/'configs/scale-screen.json'
ARCHITECTURE=(32,2560,10240)
ORDER=[(0,'packet'),(1,'packet'),(2,'packet')]


def worker(output,*,config_path=CONFIG,source_file=__file__):
    base_worker(output,config_path=config_path,hybrid_order=ORDER,bank_type=PrecisionRows,
        bank_conditions=('packet',),source_file=source_file,architecture=ARCHITECTURE,cold_reference=True)


def audit_pages(pages,arrays,calls,condition,capacity):
    return precision_audit(pages,arrays,calls,condition,capacity,architecture=ARCHITECTURE)


def decision(conditions):
    p,s=[conditions[k] for k in ('packet','stream')]
    demand(p['wall_seconds']>0 and s['wall_seconds']>0 and s['h2d_bytes']>0,'Missing scale denominator')
    return dict(checks=dict(Hfaithfulness=True,Hresources=True,
        Hacquisition=2*p['h2d_bytes']<=s['h2d_bytes'],Hruntime=5*p['wall_seconds']<=4*s['wall_seconds']),
        h2d_saving=1-p['h2d_bytes']/s['h2d_bytes'],wall_saving=1-p['wall_seconds']/s['wall_seconds'])


def analyze(output,*,config_relative='configs/scale-screen.json'):
    output=Path(output)
    result=base_analyze(output,config_relative=config_relative,hybrid_order=ORDER,
        bank_conditions=('packet',),page_auditor=audit_pages,decider=decision,
        architecture=ARCHITECTURE,cold_reference=True)
    m=json.loads((output/'manifest.json').read_text()); a=result['allocation']
    audit_allocation(a,m['environment']['gpu']['total_bytes'])
    demand(m['environment']['fp16_reduced_precision_reduction'] is True,'FP16 arithmetic changed')
    rows=json.loads((output/'episodes.json').read_text()); samples=list(read_rows(output/'resources.jsonl'))
    peaks={}; warmups={}
    for r in rows:
        demand(r['cuda']['peak_reserved_bytes']<=LIMIT,'Cold episode allocator cap exceeded')
        values=[s['gpu_used'] for s in samples if r['started']<=s['monotonic']<=r['finished']]
        demand(bool(values) and max(values)<=LIMIT,'Cold episode sampled cap exceeded')
        if r['warmup']: warmups[r['condition']]=r['wall_seconds']
        else: peaks[r['condition']]=max(peaks.get(r['condition'],0),max(values))
    useful=all(r['wall_seconds']<=5 for r in rows if r['condition']=='packet' and not r['warmup'])
    calls=json.loads((output/'episode-0/calls.json').read_text())
    cold_ttft=calls[0]['finished']-a['worker_started']
    completion=json.loads((output/'completion.json').read_text())
    demand(0<a['candidate_setup_seconds']<cold_ttft<completion['worker_inner_seconds'],'Cold clock ordering changed')
    result['checks'].update(Hcapacity=True,Huseful_latency=useful)
    result.update(scored_peak_gpu_bytes=peaks,warmup_wall_seconds=warmups,
        cold_worker_entry_to_first_logit_seconds=cold_ttft,
        full_resident_parameter_bytes=TOTAL,full_resident_parameter_over_budget_bytes=TOTAL-LIMIT,
        cpu_first_measured=True,both_offloaded_controls_fit=True,unique_access_claim=False,
        reference_kind='dense_contiguous_fp16_stream',fully_resident_scale_reference_measured=False,
        optimized_q4_compared=False,fp32_quality_evaluated=False,
        decision='short_scale_component_pass_not_deployment' if all(result['checks'].values()) else 'stop_this_scale_candidate')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.scale_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

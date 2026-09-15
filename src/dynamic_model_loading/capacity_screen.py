"""CPU-first packet acquisition under a declared GPU capacity ceiling."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT,supervise
from .capacity_support import REFERENCE_COMMIT,REFERENCE_SOURCE,LIMIT

CONFIG=ROOT/'configs/capacity-screen.json'
ORDER=[(0,'stream'),(0,'packet'),(1,'packet'),(1,'stream')]
PROMPTS=[
    'A train arrived at the mountain station before sunrise. The conductor carried a wooden box and a list of names. Nobody on the platform knew why the final name had been crossed out, but the stationmaster',
    'def unique_in_order(items):\n    seen = set()\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            yield item\n\n# The set records which values have already appeared.'
]
FROZEN=dict(protocol='docs/capacity-screen-protocol.md',repo='facebook/opt-1.3b',
    revision='3f5c25d0bc631cb57ac65913f76e22c2dfb61d62',
    artifact_manifest='results/relu-control-20260911/run/manifest.json',
    artifact_bytes={'config.json':653,'merges.txt':456318,'pytorch_model.bin':2631639353,
        'README.md':8824,'special_tokens_map.json':441,'tokenizer_config.json':685,'vocab.json':898822},
    prefix_tokens=16,generation_tokens=8,cpu_threads=4,worker_timeout_seconds=300,
    gpu_limit_mib=4800,host_floor_mib=2048,relative_l2_tolerance=1e-5,
    minimum_traffic_saving=.5,minimum_wall_saving=.2,maximum_packet_episode_seconds=5,
    useful_worker_seconds=60,reference_commit=REFERENCE_COMMIT,reference_source=REFERENCE_SOURCE,
    prompts=PROMPTS)


def validate_config(cfg):
    if cfg!=FROZEN: raise ValueError('Frozen capacity protocol changed')


def worker(output):
    from .sparse_screen import worker as run
    from .capacity_down import CapacityDown
    run(output,config_path=CONFIG,validator=validate_config,source_file=__file__,
        bank_type=CapacityDown,hybrid_order=ORDER,cold_capacity=True)


def pages_audit(pages,tensors,calls,episode):
    from .packet_analysis import audit_pages
    from .debt_analysis import demand
    if episode['condition']=='stream':
        demand(all(r.get('direct_contiguous_copy') is True for r in pages),'Dense control not contiguous')
    return audit_pages(pages,tensors,calls,episode)


def decision(conditions):
    a,b=conditions['packet'],conditions['stream']
    return dict(checks=dict(Hfaithfulness=True,Hcapacity=True,
        Hacquisition=2*a['h2d_bytes']<=b['h2d_bytes'],Hruntime=5*a['wall_seconds']<=4*b['wall_seconds']),
        savings=dict(h2d=1-a['h2d_bytes']/b['h2d_bytes'],wall=1-a['wall_seconds']/b['wall_seconds']))


def analyze(output):
    from .sparse_analysis import analyze as replay
    from .fault_screen import require_supervisor
    result=replay(output,config_relative='configs/capacity-screen.json',validator=validate_config,
        hybrid_order=ORDER,page_auditor=pages_audit,gate=decision,packet_bytes=67174400,cold_capacity=True)
    episodes=json.loads((Path(output)/'episodes.json').read_text())
    supervisor=require_supervisor(output)
    useful=(all(r['wall_seconds']<=5 for r in episodes if r['condition']=='packet')
        and supervisor['worker_wall_seconds']<=60)
    result['checks']['Hruntime']=result['checks']['Hruntime'] and useful
    result.update(useful_latency_passed=useful,declared_gpu_limit_bytes=LIMIT,
        full_fp32_parameters_exceed_budget=True,physical_smaller_gpu_tested=False,
        reference_reused=True,lower_precision_baseline_tested=False,
        decision='eligible_for_separately_frozen_followup' if all(result['checks'].values()) else 'stop_this_capacity_candidate',
        native_admission_evaluated=False,full_suite_launched=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--worker',action='store_true')
    p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.capacity_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

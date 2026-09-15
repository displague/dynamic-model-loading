"""Paired physical-grain screen: dense prefill plus exact scalar-decode packets."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT,supervise
from .debt_analysis import demand
from .context_screen import audit_workload
from .scale_screen import worker as scale_worker,analyze as scale_analyze,audit_pages as scale_audit
from .phase_rows import PhaseRows

CONFIG=ROOT/'configs/phase-screen.json'
ORDER=[(0,'packet'),(0,'phase'),(1,'phase'),(1,'packet'),(2,'packet'),(2,'phase')]
CONDITIONS=('packet','phase')


def audit_pages(pages,arrays,calls,condition,capacity):
    if condition!='phase': return scale_audit(pages,arrays,calls,condition,capacity)
    demand(capacity==0 and len(pages)==32*len(calls) and len(calls[0]['input_ids'])==512,'Wrong phase workload')
    normalized=[]
    for j,r in enumerate(pages):
        physical='stream' if j<32 else 'packet'
        demand(r['condition']=='phase' and r['physical_condition']==physical,'Wrong phase branch')
        if j>=32: demand(len(calls[j//32]['input_ids'])==1,'Non-scalar decode')
        normalized.append(dict(r,condition=physical,call=r['call']-(32 if j>=32 else 0)))
    prefill=scale_audit(normalized[:32],{},calls[:1],'stream',0)
    if len(calls)==1: return prefill
    activity={f'activity.{j-31}':arrays[f'activity.{j+1}'] for j in range(32,len(pages))}
    decode=scale_audit(normalized[32:],activity,calls[1:],'packet',0)
    total={k:prefill[k]+decode[k] for k in prefill}
    total['prefill_h2d_bytes']=prefill['weight_h2d_bytes']+prefill['metadata_h2d_bytes']
    total['decode_h2d_bytes']=decode['weight_h2d_bytes']+decode['metadata_h2d_bytes']
    return total


def decision(conditions):
    f,p,s=[conditions[k] for k in ('phase','packet','stream')]
    demand(min(p['h2d_bytes'],p['prefill_seconds'],p['wall_seconds'],s['h2d_bytes'],s['wall_seconds'])>0,'Missing phase denominator')
    return dict(checks=dict(Hfaithfulness=True,Hresources=True,
        Hprefill_gain=10*f['prefill_seconds']<=9*p['prefill_seconds'],
        Hwhole_episode_gain=20*f['wall_seconds']<=19*p['wall_seconds'],
        Htraffic_tradeoff=10*f['h2d_bytes']<=13*p['h2d_bytes'],
        Hstream_baseline=5*f['h2d_bytes']<=s['h2d_bytes'] and 2*f['wall_seconds']<=s['wall_seconds'],
        Hphysical_split=f['prefill_h2d_bytes']==s['prefill_h2d_bytes'] and f['decode_h2d_bytes']==p['decode_h2d_bytes']),
        h2d_saving=1-f['h2d_bytes']/s['h2d_bytes'],wall_saving=1-f['wall_seconds']/s['wall_seconds'],
        extra_bytes_vs_packet=f['h2d_bytes']/p['h2d_bytes']-1,
        prefill_gain_vs_packet=1-f['prefill_seconds']/p['prefill_seconds'],
        episode_gain_vs_packet=1-f['wall_seconds']/p['wall_seconds'])


def worker(output):
    audit_workload(json.loads(CONFIG.read_text()))
    scale_worker(output,config_path=CONFIG,source_file=__file__,hybrid_order=ORDER,
        bank_type=PhaseRows,bank_conditions=CONDITIONS)


def analyze(output):
    output=Path(output); audit_workload(json.loads((output/'config.json').read_text()))
    result=scale_analyze(output,config_relative='configs/phase-screen.json',hybrid_order=ORDER,
        bank_conditions=CONDITIONS,page_auditor=audit_pages,decider=decision)
    result.update(prefix_tokens=512,maximum_kv_positions=527,workload_is_new_holdout=False,
        all_offloaded_conditions_fit=True,decision='short_phase_component_pass_not_deployment'
        if all(result['checks'].values()) else 'stop_this_phase_candidate')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.phase_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

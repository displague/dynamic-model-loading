"""Frozen 512-token union/durability diagnostic for the CPU-first FP16 loader."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT,supervise
from .debt_analysis import demand
from .scale_screen import worker as scale_worker,analyze as scale_analyze

CONFIG=ROOT/'configs/context-screen.json'


def stage_decision(conditions):
    p,s=[conditions[k] for k in ('packet','stream')]
    demand(min(s[k] for k in ('prefill_h2d_bytes','decode_h2d_bytes','prefill_seconds','decode_seconds'))>0,'Missing stage denominator')
    return dict(checks=dict(Hprefill_traffic=2*p['prefill_h2d_bytes']<=s['prefill_h2d_bytes'],
        Hprefill_runtime=p['prefill_seconds']<=s['prefill_seconds'],
        Hdecode_traffic=10*p['decode_h2d_bytes']<=s['decode_h2d_bytes'],
        Hdecode_runtime=2*p['decode_seconds']<=s['decode_seconds']),
        prefill_traffic_fraction=p['prefill_h2d_bytes']/s['prefill_h2d_bytes'],
        decode_traffic_fraction=p['decode_h2d_bytes']/s['decode_h2d_bytes'],
        prefill_wall_ratio=p['prefill_seconds']/s['prefill_seconds'],
        decode_wall_ratio=p['decode_seconds']/s['decode_seconds'])


def audit_workload(cfg):
    import hashlib
    import subprocess
    raw=subprocess.check_output(['git','show',cfg['workload_origin']['commit']+':'+cfg['workload_origin']['path']],cwd=ROOT)
    demand(hashlib.sha256(raw).hexdigest()==cfg['workload_origin']['sha256'],'Origin corpus changed')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    groups=cfg['workload_origin']['record_groups']
    demand(groups==[[0,1,2],[3,4,5],[6,7,8],[9,10,11]],'Diagnostic selection changed')
    texts=['\n\n'.join(rows[i]['text'] for i in group) for group in groups]
    demand(texts==[*cfg['prompts'],cfg['warmup_prompt']],'Diagnostic concatenation changed')
    demand(cfg['prefix_tokens']==512 and cfg['generation_tokens']==16,'Context length changed')


def worker(output):
    audit_workload(json.loads(CONFIG.read_text()))
    scale_worker(output,config_path=CONFIG,source_file=__file__)


def analyze(output):
    output=Path(output); cfg=json.loads((output/'config.json').read_text()); audit_workload(cfg)
    result=scale_analyze(output,config_relative='configs/context-screen.json')
    stages=stage_decision(result['conditions']); result['checks'].update(stages.pop('checks')); result.update(stages)
    result.update(prefix_tokens=512,maximum_kv_positions=527,workload_is_new_holdout=False,
        decision='short_context_components_pass_not_validation' if all(result['checks'].values()) else 'stop_uniform_long_prefix_expansion')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.context_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

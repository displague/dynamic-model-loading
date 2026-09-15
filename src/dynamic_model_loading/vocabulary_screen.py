"""Full-vocabulary acquisition score tested against actual verified prefixes."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT, supervise
from .risk_screen import FROZEN as OLD, bundle as parent_bundle, worker as parent_worker

CONFIG=ROOT/'configs/vocabulary-risk-screen.json'
PROMPTS=[
    'A botanist catalogued specimens beside the river. Each envelope carried a date, a location, and a short note about the weather. The assistant noticed that several labels used the same number.',
    'def parse_record(raw):\n    fields = raw.strip().split(",")\n    if len(fields) != 3:\n        raise ValueError("expected three fields")\n    return fields\n\n# Validate the record before converting its values.'
]
FROZEN={k:v for k,v in OLD.items() if k not in ('protocol','diagnostic_indices','conditions')}
FROZEN.update(protocol='docs/vocabulary-risk-screen-protocol.md',prompts=PROMPTS,
    conditions=[['risk','fullrisk','all35'],['all35','fullrisk','risk']],
    monte_carlo_draws=16,monte_carlo_seed_base=20260918,minimum_charged_h2d_saving=.01,
    numpy_threads=4,license_bytes=11343,
    license_sha256='832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e')


def validate_config(cfg):
    if cfg!=FROZEN: raise ValueError('Frozen vocabulary-risk protocol changed')


def bundle(output):
    return parent_bundle(output,config_path=CONFIG,validator=validate_config,source_file=__file__)


def worker(output):
    from .risk_runtime import VocabularyReadout
    parent_worker(output,bundle_fn=bundle,conditions=FROZEN['conditions'],runtime_class=VocabularyReadout)


def screen_decision(conditions,target_seconds):
    c=conditions['fullrisk']; controls=[conditions[k] for k in ('risk','all35')]
    eligible=c['accepted']>0 and all(b['accepted']>0 for b in controls)
    acquisition=(eligible and c['acceptance']>=.5 and
        all(c['acceptance']>=b['acceptance'] and
            100*c['charged_h2d_bytes']*b['accepted']<=99*b['charged_h2d_bytes']*c['accepted'] for b in controls))
    runtime=all(c['charged_wall_seconds']<=b['charged_wall_seconds'] for b in controls)
    return dict(gates=dict(Hfaithfulness=True,Hacquisition=bool(acquisition),Hruntime=bool(runtime)),
        decision='eligible_for_separately_frozen_followup' if acquisition and runtime else 'stop_this_full_vocabulary_ranker',
        beats_resident_target_clock=c['charged_wall_seconds']<=target_seconds,
        native_admission_evaluated=False,full_suite_launched=False)


def analyze(output):
    from .risk_analysis import analyze as replay
    from .numpy_backend import set_blas_threads
    blas=set_blas_threads(4)
    manifest=json.loads((Path(output)/'manifest.json').read_text())
    if manifest.get('numpy_blas')!=blas: raise ValueError('NumPy backend/thread identity changed')
    return replay(output,config_relative='configs/vocabulary-risk-screen.json',validator=validate_config,
                  gate=screen_decision,condition_names=('risk','fullrisk','all35'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--worker',action='store_true')
    p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.vocabulary_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

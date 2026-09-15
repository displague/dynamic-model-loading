"""Bounded causal recursive/vector-coordinate component screen."""
import argparse,json,time
from pathlib import Path
from .fault_screen import supervise
from .field_replay import begin,finish,audit
from .recursive_field import recursive_predictions,recursive_summary

NAME='recursive-screen'
CONFIG=dict(protocol='docs/recursive-screen-protocol.md',parent_release='v0.26.0',
    fit_positions=96,diagnostic_positions=32,shrinkage=.5,observations=4,
    feature_variance_floor=1e-12,ar_clip=[0,.95],noise_fraction=1e-6,
    geometry_mse_ratio=.9,temporal_mse_ratio=.9,per_document_max_ratio=1.05,timeout_seconds=300)

def analyze(output):
    return audit(output,NAME,CONFIG,recursive_predictions,recursive_summary)

def worker(output):
    started=time.perf_counter(); inputs=begin(output,NAME,CONFIG)
    raw=recursive_predictions(inputs)
    finish(output,inputs,raw,recursive_summary(inputs,raw),time.perf_counter()-started)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze: parser.error('Conflicting modes')
    if args.worker: worker(args.output)
    else:
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.recursive_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'): raise SystemExit(1)

if __name__=='__main__': main()

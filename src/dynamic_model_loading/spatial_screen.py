"""One paid observation: does a covariance field help unobserved physical pages?"""
import argparse,json,time
from pathlib import Path
from .fault_screen import supervise
from .field_replay import begin,finish,audit
from .page_field import spatial_predictions,spatial_summary

NAME='spatial-screen'
CONFIG=dict(protocol='docs/spatial-screen-protocol.md',parent_release='v0.26.0',
    fit_positions=96,diagnostic_positions=32,shrinkage=.5,shuffle_seed=2701,
    budgets=[1,4],primary_budget=1,primary_mse_ratio=.9,per_document_max_ratio=1.05,
    timeout_seconds=300)


def compute(inputs):
    return spatial_predictions(inputs['z'])


def summarize(inputs,predictions):
    return spatial_summary(inputs['z'],predictions)


def analyze(output):
    return audit(output,NAME,CONFIG,compute,summarize)


def worker(output):
    started=time.perf_counter()
    inputs=begin(output,NAME,CONFIG)
    predictions=compute(inputs)
    summary=summarize(inputs,predictions)
    finish(output,inputs,predictions,summary,time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    if args.worker:
        worker(args.output)
    else:
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.spatial_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'):
            raise SystemExit(1)


if __name__=='__main__':
    main()

"""Decision-value acquisition and projected sufficient-observation CPU screen."""
import argparse,json,time
from pathlib import Path
from .fault_screen import supervise
from .field_replay import begin,finish,audit
from .decision_acquisition import acquisition_predictions,acquisition_summary

NAME='acquisition-screen'
CONFIG=dict(protocol='docs/acquisition-screen-protocol.md',parent_release='v0.26.0',
    fit_positions=96,diagnostic_positions=32,budget=17,quadrature=32,audit_quadrature=128,
    numerical_tolerance=1e-4,commutation_tolerance=1e-10,minimum_fewer_errors=2,
    maximum_document_extra_errors=1,timeout_seconds=300)

def analyze(output): return audit(output,NAME,CONFIG,acquisition_predictions,acquisition_summary)

def worker(output):
    started=time.perf_counter(); inputs=begin(output,NAME,CONFIG)
    raw=acquisition_predictions(inputs)
    finish(output,inputs,raw,acquisition_summary(inputs,raw),time.perf_counter()-started)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze: parser.error('Conflicting modes')
    if args.worker: worker(args.output)
    else:
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.acquisition_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'): raise SystemExit(1)

if __name__=='__main__': main()

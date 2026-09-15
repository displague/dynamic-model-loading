"""Single-packet exact-active-row acquisition under a separately frozen screen."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT, supervise
from .sparse_screen import worker as run_worker

CONFIG=ROOT/'configs/row-packet-screen.json'
ORDER=[(0,'stream'),(0,'sparse'),(0,'packet'),(1,'packet'),(1,'sparse'),(1,'stream')]
PACKET_BYTES=8192*(8+2048*4)


def validate_config(cfg):
    fixed=dict(protocol='docs/row-packet-screen-protocol.md',repo='facebook/opt-1.3b',
        revision='3f5c25d0bc631cb57ac65913f76e22c2dfb61d62',
        artifact_manifest='results/relu-control-20260911/run/manifest.json',
        prefix_tokens=16,generation_tokens=8,cpu_threads=4,worker_timeout_seconds=300,
        gpu_limit_mib=15000,host_floor_mib=2048,relative_l2_tolerance=1e-5,
        minimum_traffic_saving=.5,minimum_wall_saving=.2)
    if any(cfg.get(k)!=v for k,v in fixed.items()) or len(cfg.get('prompts',[]))!=2:
        raise ValueError('Frozen packet screen changed')


def worker(output):
    from .row_packet import RowPacket
    run_worker(output,config_path=CONFIG,validator=validate_config,source_file=__file__,
               bank_type=RowPacket,hybrid_order=ORDER)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--analyze',action='store_true'); args=parser.parse_args()
    from .packet_analysis import analyze
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        report=supervise(args.output,module='dynamic_model_loading.packet_screen',analyzer=analyze)
        print(json.dumps(report,indent=2))
        if report.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

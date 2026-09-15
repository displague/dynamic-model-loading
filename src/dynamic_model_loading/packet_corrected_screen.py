"""Fresh corrected packet screen preserving the original dense linear operator."""
import argparse
import json
from pathlib import Path
from .fault_screen import ROOT, supervise
from .packet_screen import validate_config as original_validator, ORDER, PACKET_BYTES
from .sparse_screen import worker as run_worker

CONFIG=ROOT/'configs/row-packet-corrected-screen.json'


def validate_config(cfg):
    if cfg.get('protocol')!='docs/row-packet-corrected-protocol.md' or cfg.get('original_workspace_layout') is not True:
        raise ValueError('Corrected numerical contract changed')
    base=dict(cfg,protocol='docs/row-packet-screen-protocol.md')
    base.pop('original_workspace_layout')
    original_validator(base)


def worker(output):
    from .row_packet import OriginalRowPacket
    run_worker(output,config_path=CONFIG,validator=validate_config,source_file=__file__,
               bank_type=OriginalRowPacket,hybrid_order=ORDER)


def analyze(output):
    from .sparse_analysis import analyze as replay
    from .packet_analysis import audit_pages, decision
    result=replay(output,config_relative='configs/row-packet-corrected-screen.json',validator=validate_config,
        hybrid_order=ORDER,page_auditor=audit_pages,gate=decision,packet_bytes=PACKET_BYTES)
    a=result['allocation']
    if not (a['original_workspace_layout'] is True and a['weight_workspace_contiguous'] is True
            and a['workspace_strides']==[1,8192]):
        raise ValueError('Original dense numerical layout not preserved')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--worker',action='store_true')
    p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.packet_corrected_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

"""Reconstruct packet contents and cost from observed exact activity, not labels."""
import math
import numpy as np
from .debt_analysis import demand
from .sparse_analysis import analyze as replay, audit_pages as extent_audit
from .packet_screen import validate_config, ORDER, PACKET_BYTES


def audit_pages(pages,tensors,calls,episode):
    if episode['condition']!='packet':
        return dict(extent_audit(pages,tensors,calls,episode),metadata_h2d_bytes=0)
    demand(len(pages)==len(calls)*24,'Missing packet calls')
    stats=dict(h2d_bytes=0,d2h_bytes=0,extents=0,prefill_bytes=0,decode_bytes=0,
               active_neurons=0,observed_neurons=0,metadata_h2d_bytes=0)
    previous=calls[0]['started']
    for i,row in enumerate(pages):
        step,layer=divmod(i,24); c=calls[step]; n=len(c['input_ids'])
        demand((row['call'],row['layer'],row['tokens'],row['episode'],row['condition'])==
            (i+1,layer,n,episode['episode'],'packet'),'Packet call sequence changed')
        times=[row[k] for k in ('started','selection_finished','packing_finished','transfer_finished',
                                'acquisition_finished','compute_finished')]
        demand(all(math.isfinite(t) for t in times) and times==sorted(times) and
            max(previous,c['started'])<=times[0]<=times[-1]<=c['finished'],'Packet serialization changed')
        previous=times[-1]
        packed=tensors[f'activity.{i+1}']
        demand(packed.dtype==np.uint8 and packed.shape==(n,1024),'Packet activity shape changed')
        activity=np.unpackbits(packed,axis=1).astype(bool)
        rows=np.flatnonzero(activity.any(0)).tolist(); count=len(rows)
        weight=count*8192; metadata=count*8; total=weight+metadata; d2h=n*8192+1
        demand(row['rows']==rows and row['packets']==int(count>0) and
            row['weight_h2d_bytes']==weight and row['metadata_h2d_bytes']==metadata and
            row['packet_h2d_bytes']==total and row['activity_d2h_bytes']==d2h,'Packet contents/copies changed')
        stats['h2d_bytes']+=total; stats['metadata_h2d_bytes']+=metadata; stats['d2h_bytes']+=d2h
        stats['extents']+=int(count>0); stats['prefill_bytes' if step==0 else 'decode_bytes']+=total
        stats['active_neurons']+=int(activity.sum()); stats['observed_neurons']+=activity.size
    return stats


def decision(conditions):
    candidate=conditions['packet']; controls=[conditions[k] for k in ('stream','sparse')]
    demand(all(c['h2d_bytes']>0 and c['wall_seconds']>0 for c in controls),'Missing control work')
    checks=dict(Htraffic=all(2*candidate['h2d_bytes']<=c['h2d_bytes'] for c in controls),
                Hruntime=all(5*candidate['wall_seconds']<=4*c['wall_seconds'] for c in controls))
    return dict(checks=checks,
        savings={k:dict(h2d=1-candidate['h2d_bytes']/conditions[k]['h2d_bytes'],
                       wall=1-candidate['wall_seconds']/conditions[k]['wall_seconds']) for k in ('stream','sparse')},
        decision='eligible_for_separately_frozen_followup' if all(checks.values()) else 'stop_this_row_packet_candidate',
        full_suite_launched=False,native_admission_evaluated=False)


def analyze(output):
    return replay(output,config_relative='configs/row-packet-screen.json',validator=validate_config,
        hybrid_order=ORDER,page_auditor=audit_pages,gate=decision,packet_bytes=PACKET_BYTES)

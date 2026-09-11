"""Validate the complete microbenchmark ledger without rerunning CUDA timing."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from .causal_analysis import require
from .experiment import digest, write_json
from .hardware_cost import GROUP_COUNTS, WORKLOADS, byte_count, reduce_timings, synthetic_inputs, input_receipt
from .provenance import frozen_environment, verify_snapshot


def analyze(path):
    root=Path(path)
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    cfg,manifest,summary=read('config.json'),read('manifest.json'),read('summary.json')
    require(summary['status']=='hardware_characterization_completed' and summary['runtime_claim'] is False
            and not (root/'failure.json').exists(),'Incomplete hardware characterization')
    require(cfg=={'group_counts':list(GROUP_COUNTS),'workloads':list(WORKLOADS),'width':8,'hidden':1536,'pool_groups':1120,
            'warmups':3,'measured_repetitions':10,'seed':1729,'dtype':'float32','rank_tokens':32,'rank_layers':28,'rank_groups':1120,'rank_keep':1008},'Changed hardware grid')
    require(digest(root/'config.json')==manifest['config_sha256'] and digest(root/'protocol.md')==manifest['protocol_sha256'],'Configuration/protocol hash mismatch')
    verify_snapshot(root,manifest,'configs/hardware-cost.json','docs/hardware-cost-protocol.md',__file__)
    for name,value in manifest['sources'].items():require(digest(root/'source'/name)==value,'Source checksum mismatch')
    env=manifest['environment'];frozen_environment(env)
    require(env['device']=='cuda' and env['cpu_threads']==4 and env['tf32_matmul'] is False and env['tf32_cudnn'] is False,'Hardware environment mismatch')
    inputs=read('inputs.json')
    require(inputs==input_receipt(*synthetic_inputs()),'Input receipt differs from frozen seeded recipe')
    rows=[json.loads(line) for line in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected=[]
    for count in GROUP_COUNTS:
        for workload in WORKLOADS:
            expected += [('integrity',count,workload)]+[('timing',count,workload,i) for i in range(13)]+[('aggregate',count,workload)]
    expected += [('ranking_integrity',)]
    for method in ('stable_argsort','topk'):
        expected += [('ranking_timing',method,i) for i in range(13)]+[('ranking_aggregate',method)]
    def key(r):
        kind=r['kind']
        if kind=='ranking_integrity':return (kind,)
        if kind.startswith('ranking_'):return (kind,r['method'],r['repetition']) if kind=='ranking_timing' else (kind,r['method'])
        return (kind,r['groups'],r['workload'],r['repetition']) if kind=='timing' else (kind,r['groups'],r['workload'])
    require([key(r) for r in rows]==expected,'Hardware ledger order/grid mismatch')
    results=[]
    for count in GROUP_COUNTS:
        for workload in WORKLOADS:
            integrity=next(r for r in rows if r['kind']=='integrity' and r['groups']==count and r['workload']==workload)
            compute=workload in ('resident_ffn','gather_transfer_pack_ffn')
            require(integrity['passed'] is True and integrity['payload_bytes']==byte_count(count),'Hardware integrity gate failed')
            require(type(integrity['relative_l2']) in (int,float) and math.isfinite(integrity['relative_l2']) and 0<=integrity['relative_l2']<=.01
                if compute else integrity['relative_l2'] is None,'Invalid integrity metric')
            # Pool and pinned pool, selected pageable/staging, order, CPU input and reference output.
            cpu_expected=2*byte_count(1120)+2*byte_count(count)+1120*8+2*1536*4
            gpu_expected=2*byte_count(count)+1536*4
            require(integrity['cpu_persistent_payload_bytes']==cpu_expected and integrity['gpu_preallocated_payload_bytes']==gpu_expected,'Payload accounting mismatch')
            timings=[r for r in rows if r['kind']=='timing' and r['groups']==count and r['workload']==workload]
            measured=next(r for r in rows if r['kind']=='aggregate' and r['groups']==count and r['workload']==workload)
            peak=measured['gpu_peak_allocated_bytes']
            require(type(peak) is int and peak>=gpu_expected,'Invalid timing allocator peak')
            value={'groups':count,'workload':workload,'payload_bytes':byte_count(count),'transfers':workload!='resident_ffn','medians':reduce_timings(timings),'gpu_peak_allocated_bytes':peak}
            require(measured=={'kind':'aggregate',**value},'Hardware median mismatch')
            results.append(value)
    ranks=next(r for r in rows if r['kind']=='ranking_integrity')
    require(ranks['unique_rows']==32*28 and ranks['unique_masks_equal'] is True,'Unique ranking gate failed')
    require(ranks['stable_tie_indices']==list(range(1008)) and len(ranks['topk_tie_indices'])==1008
        and ranks['topk_tie_indices']==sorted(set(ranks['topk_tie_indices']))
        and all(type(i) is int and 0<=i<1120 for i in ranks['topk_tie_indices'])
        and ranks['equal_score_masks_equal']==(ranks['stable_tie_indices']==ranks['topk_tie_indices']),'Ranking tie receipt mismatch')
    ranking=[]
    for method in ('stable_argsort','topk'):
        value={'method':method,'medians':reduce_timings([r for r in rows if r['kind']=='ranking_timing' and r['method']==method])}
        require(next(r for r in rows if r['kind']=='ranking_aggregate' and r['method']==method)=={'kind':'ranking_aggregate',**value},'Ranking median mismatch')
        ranking.append(value)
    require(summary['aggregates']==results and summary['ranking']==ranking,'Hardware summary mismatch')
    return {'status':'validated','row_counts':dict(Counter(r['kind'] for r in rows)),'aggregates':results,'ranking':ranking,
            'input_hashes':{name:digest(root/name) for name in ('config.json','manifest.json','results.jsonl','summary.json','inputs.json')},
            'runtime_claim':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'hardware_analysis.py')
    try:write_json(output/'summary.json',analyze(args.run))
    except (Exception,KeyboardInterrupt) as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':main()

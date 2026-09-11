"""Steady-state acquisition primitives; no pretrained-model runtime claim."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import torch
from torch.nn import functional as F

from .experiment import digest, environment, write_json
from .metrics import relative_l2
from .provenance import committed_inputs, owning_repository, frozen_environment


GROUP_COUNTS = (1,8,32,128,512,1120)
WORKLOADS = ('pinned_aggregate','pageable_staged','gather_staged','pinned_individual','resident_ffn','gather_transfer_pack_ffn')


def byte_count(groups, width=8, hidden=1536):
    if type(groups) is not int or groups <= 0:raise ValueError('Positive group count required')
    return 3*groups*width*hidden*4


def reduce_timings(rows):
    if [(r['repetition'],r['warmup']) for r in rows] != [(i,i<3) for i in range(13)]:
        raise ValueError('Incomplete timing grid')
    for row in rows:
        if any(type(row[k]) not in (int,float) or not 0<row[k]<float('inf') for k in ('cuda_ms','wall_ms')):
            raise ValueError('Invalid measured time')
    return {key:statistics.median(r[key] for r in rows if not r['warmup']) for key in ('cuda_ms','wall_ms')}


def rank(scores, method, count=1008):
    if method=='stable_argsort':indices=scores.argsort(descending=True,stable=True)[:count]
    elif method=='topk':indices=scores.topk(count,sorted=False).indices
    else:raise ValueError('Unknown ranking method')
    return torch.zeros_like(scores,dtype=torch.bool).scatter_(0,indices,True)


def synthetic_inputs(groups=1120, width=8, hidden=1536):
    generator=torch.Generator().manual_seed(1729)
    pool=torch.randn(groups,3,width,hidden,dtype=torch.float32,generator=generator)/hidden**.5
    order=torch.randperm(groups,generator=generator)
    x=torch.randn(1,hidden,dtype=torch.float32,generator=generator)
    return pool,order,x


def input_receipt(pool,order,x):
    return {'group_order':order.tolist(),'pool_shape':list(pool.shape),
            'pool_sha256':hashlib.sha256(pool.numpy().tobytes()).hexdigest(),'input':x.tolist()}


@torch.inference_mode()
def run(output_path, protocol_path):
    output=Path(output_path); output.mkdir(parents=True,exist_ok=False)
    stream=(output/'results.jsonl').open('x',encoding='utf-8')
    def record(kind,**values):
        row={'kind':kind,**values};stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno());return row
    try:
        if str(torch.__version__)!='2.10.0+cu130' or not torch.cuda.is_available():raise ValueError('Pinned CUDA baseline required')
        if Path(protocol_path).as_posix()!='docs/hardware-cost-protocol.md':raise ValueError('Use the frozen hardware protocol')
        repository=owning_repository(__file__)
        config_path=repository/'configs/hardware-cost.json'
        source,protocol=committed_inputs(__file__,config_path,protocol_path)
        torch.set_num_threads(4);torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        shutil.copyfile(protocol,output/'protocol.md')
        shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
        config={'group_counts':list(GROUP_COUNTS),'workloads':list(WORKLOADS),'width':8,'hidden':1536,'pool_groups':1120,
            'warmups':3,'measured_repetitions':10,'seed':1729,'dtype':'float32','rank_tokens':32,'rank_layers':28,'rank_groups':1120,'rank_keep':1008}
        if json.loads(config_path.read_text(encoding='utf-8'))!=config:raise ValueError('Changed hardware configuration')
        shutil.copyfile(config_path,output/'config.json')
        manifest={**source,'started_utc':datetime.now(timezone.utc).isoformat(),'environment':environment(torch.device('cuda')),
            'config_sha256':digest(output/'config.json'),'protocol_sha256':digest(output/'protocol.md'),
            'sources':{p.relative_to(output/'source').as_posix():digest(p) for p in (output/'source').rglob('*.py')}}
        write_json(output/'manifest.json',manifest)
        frozen_environment(manifest['environment'])
        pool,order,x_cpu=synthetic_inputs();pinned_pool=pool.pin_memory();x=x_cpu.cuda()
        write_json(output/'inputs.json',input_receipt(pool,order,x_cpu))
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();end.record();torch.cuda.synchronize()
        aggregates=[]
        for count in GROUP_COUNTS:
            indices=order[:count];group_ids=indices.tolist()
            pageable=pool.index_select(0,indices);staging=torch.empty_like(pageable,pin_memory=True)
            staging.copy_(pageable);gpu=torch.empty_like(pageable,device='cuda');gpu.copy_(staging)
            gate=torch.empty(count*8,1536,device='cuda',dtype=torch.float32);up=torch.empty_like(gate)
            down=torch.empty(1536,count*8,device='cuda',dtype=torch.float32)
            def pack():
                gate.view(count,8,1536).copy_(gpu[:,0])
                up.view(count,8,1536).copy_(gpu[:,1])
                down.view(1536,count,8).copy_(gpu[:,2].permute(2,0,1))
            def compute():return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
            pack()
            expected=F.linear(F.silu(F.linear(x_cpu,pageable[:,0].reshape(count*8,1536)))*
                F.linear(x_cpu,pageable[:,1].reshape(count*8,1536)),pageable[:,2].reshape(count*8,1536).T)
            def operation(workload):
                if workload=='pinned_aggregate':gpu.copy_(staging,non_blocking=True)
                elif workload=='pageable_staged':staging.copy_(pageable);gpu.copy_(staging,non_blocking=True)
                elif workload in ('gather_staged','gather_transfer_pack_ffn'):
                    torch.index_select(pool,0,indices,out=staging);gpu.copy_(staging,non_blocking=True)
                    if workload=='gather_transfer_pack_ffn':pack();return compute()
                elif workload=='pinned_individual':
                    for i,group in enumerate(group_ids):gpu[i].copy_(pinned_pool[group],non_blocking=True)
                elif workload=='resident_ffn':return compute()
                return gpu
            for workload in WORKLOADS:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                if workload!='resident_ffn':gpu.fill_(float('nan'))
                if workload in ('pageable_staged','gather_staged','gather_transfer_pack_ffn'):staging.fill_(float('nan'))
                if workload=='pinned_aggregate':staging.copy_(pageable)
                value=operation(workload);torch.cuda.synchronize()
                is_compute=workload in ('resident_ffn','gather_transfer_pack_ffn')
                error=relative_l2(expected,value.cpu()) if is_compute else None
                passed=error<=.01 if is_compute else torch.equal(pageable,value.cpu())
                record('integrity',groups=count,workload=workload,passed=passed,relative_l2=error,
                    payload_bytes=byte_count(count),cpu_persistent_payload_bytes=sum(t.numel()*t.element_size() for t in (pool,pinned_pool,pageable,staging,order,x_cpu,expected)),
                    gpu_preallocated_payload_bytes=sum(t.numel()*t.element_size() for t in (gpu,gate,up,down,x)))
                if not passed:raise ValueError('Hardware integrity gate failed')
                timings=[]
                for repetition in range(13):
                    torch.cuda.synchronize()
                    start=time.perf_counter();begin.record();value=operation(workload);end.record();torch.cuda.synchronize()
                    timings.append(record('timing',groups=count,workload=workload,repetition=repetition,warmup=repetition<3,
                        cuda_ms=begin.elapsed_time(end),wall_ms=(time.perf_counter()-start)*1000))
                value={'groups':count,'workload':workload,'payload_bytes':byte_count(count),
                       'transfers':workload!='resident_ffn','medians':reduce_timings(timings),
                       'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                record('aggregate',**value);aggregates.append(value)
            del gpu,gate,up,down,staging,pageable,value
            print('Completed hardware group count',count,flush=True)
        scores=torch.stack([torch.randperm(1120).float() for _ in range(32*28)]).to('cuda')
        for row in scores:
            if not torch.equal(rank(row,'stable_argsort'),rank(row,'topk')):raise ValueError('Unique-score ranking mismatch')
        equal=torch.ones(1120,device='cuda')
        record('ranking_integrity',unique_rows=len(scores),unique_masks_equal=True,
            equal_score_masks_equal=torch.equal(rank(equal,'stable_argsort'),rank(equal,'topk')),
            stable_tie_indices=torch.nonzero(rank(equal,'stable_argsort')).flatten().cpu().tolist(),
            topk_tie_indices=torch.nonzero(rank(equal,'topk')).flatten().cpu().tolist())
        ranking=[]
        for method in ('stable_argsort','topk'):
            timings=[]
            for repetition in range(13):
                torch.cuda.synchronize()
                start=time.perf_counter();begin.record()
                for row in scores:rank(row,method)
                end.record();torch.cuda.synchronize()
                timings.append(record('ranking_timing',method=method,repetition=repetition,warmup=repetition<3,
                    cuda_ms=begin.elapsed_time(end),wall_ms=(time.perf_counter()-start)*1000))
            value={'method':method,'medians':reduce_timings(timings)};record('ranking_aggregate',**value);ranking.append(value)
        summary={'status':'hardware_characterization_completed','aggregates':aggregates,'ranking':ranking,
                 'runtime_claim':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
        write_json(output/'summary.json',summary);return summary
    except (Exception,KeyboardInterrupt) as error:
        record('error',type=type(error).__name__,message=str(error));write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
    finally:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',default='docs/hardware-cost-protocol.md');args=parser.parse_args()
    print(run(args.output,args.protocol)['status'])


if __name__=='__main__':main()

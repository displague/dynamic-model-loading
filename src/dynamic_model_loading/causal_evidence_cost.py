"""Paired causal-controller primitives and explicitly serialized batch sensitivity."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from . import causal_evidence as ce
from .adapters import extract_ffns
from .causal import group_scores
from .causal_study import packed
from .causal_evidence_study import controllers,load_inputs,read
from .experiment import digest,environment,write_json
from .hardware_cost import synthetic_inputs,input_receipt
from .metrics import relative_l2
from .cache import cache_shape,hot_mask
from .provenance import committed_inputs,frozen_environment,verify_snapshot

TIMING_WORKLOADS=('dense_ffn','down_products','controller','output_additions')
TIMING_PROTOCOL='docs/causal-evidence-timing-correction.md'


def restore_fitted_layout(data,stored):
    """Recover the original solve layout without changing a fitted coefficient."""
    fitted=ce.fit_models(data)
    actual=ce.flatten_models(fitted)
    if set(actual)!=set(stored) or any(not torch.equal(actual[k],stored[k]) for k in actual):
        raise ValueError('Timing refit differs from the frozen coefficients')
    return fitted


def rounded_cost(count,workload,hardware):
    if count==0:return 0.
    candidates=sorted((r for r in hardware['aggregates'] if r['workload']==workload),key=lambda r:r['groups'])
    return next(r['medians']['wall_ms'] for r in candidates if r['groups']>=count)


def action_cost(receipt,hot,hardware,controller_ms_per_visit,output_add_ms_per_visit=0.):
    initial,extra=receipt['initial'].numpy(),receipt['additions'].numpy()
    resident=np.broadcast_to(hot.sum(-1),(initial.shape[0],initial.shape[1]))
    first=(initial&~hot).sum(-1);added=(extra&~hot).sum(-1)
    def total(counts,kind):
        values,counts_n=np.unique(counts,return_counts=True)
        return sum(int(n)*rounded_cost(int(v),kind,hardware) for v,n in zip(values,counts_n))
    movement=total(first,'gather_transfer_pack_ffn')+total(added,'gather_transfer_pack_ffn')
    resident_ms=total(resident,'resident_ffn')
    control=first.size*controller_ms_per_visit
    output_add=first.size*output_add_ms_per_visit
    return dict(batch_ms=movement,resident_ms=resident_ms,controller_ms=control,output_addition_ms=output_add,
                total_ms=movement+resident_ms+control+output_add)


def batch_counts(parent,cfg):
    counts={28,56,112}
    resident=parent['hot'].sum(-1)
    dense_hot=hot_mask(parent['importance'],parent['sizes'],cache_shape(parent['sizes'],cfg['budget_bytes']-cfg['acquisition_workspace_bytes'])['retained_slots'],'static_equal_layer')
    for value in resident:
        counts.add(int(value))
        for retained in (1008,1036,1064):counts.add(retained-int(value))
    for value in dense_hot.sum(-1):counts.update((int(value),1120-int(value)))
    return sorted(counts)


@torch.inference_mode()
def measure_batches(parent,cfg,output):
    pool,order,x_cpu=synthetic_inputs();x=x_cpu.cuda()
    write_json(output/'batch-inputs.json',input_receipt(pool,order,x_cpu))
    rows=[];aggregates=[]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();end.record();torch.cuda.synchronize()
    with (output/'batch-results.jsonl').open('x',encoding='utf-8') as stream:
        def record(row):rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        for count in batch_counts(parent,cfg):
            if 2*count*147456+1536*4+count*8*4>cfg['acquisition_workspace_bytes']:
                raise ValueError('Batch staging and canonical weights exceed charged workspace')
            indices=order[:count];pageable=pool.index_select(0,indices)
            staging=torch.empty_like(pageable,pin_memory=True);staging.copy_(pageable)
            gpu=torch.empty_like(pageable,device='cuda');gpu.copy_(staging)
            gate=torch.empty(count*8,1536,device='cuda');up=torch.empty_like(gate)
            down=torch.empty(1536,count*8,device='cuda')
            def pack():
                gate.view(count,8,1536).copy_(gpu[:,0]);up.view(count,8,1536).copy_(gpu[:,1])
                down.view(1536,count,8).copy_(gpu[:,2].permute(2,0,1))
            def compute():return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
            def operation(kind):
                if kind=='gather_transfer_pack_ffn':
                    torch.index_select(pool,0,indices,out=staging);gpu.copy_(staging,non_blocking=True);pack()
                return compute()
            pack()
            reference=F.linear(F.silu(F.linear(x_cpu,pageable[:,0].reshape(count*8,1536)))*
                F.linear(x_cpu,pageable[:,1].reshape(count*8,1536)),pageable[:,2].reshape(count*8,1536).T)
            for workload in ('resident_ffn','gather_transfer_pack_ffn'):
                if workload!='resident_ffn':gpu.fill_(float('nan'));staging.fill_(float('nan'))
                actual=operation(workload);torch.cuda.synchronize();error=relative_l2(reference,actual.cpu())
                record(dict(kind='integrity',groups=count,workload=workload,relative_l2=error,passed=error<=.01))
                if error>.01:raise ValueError('Exact-batch primitive integrity failed')
                samples=[]
                for repetition in range(13):
                    torch.cuda.synchronize();start=time.perf_counter();begin.record()
                    operation(workload);end.record();torch.cuda.synchronize()
                    row=dict(kind='timing',groups=count,workload=workload,repetition=repetition,warmup=repetition<3,
                        cuda_ms=begin.elapsed_time(end),wall_ms=(time.perf_counter()-start)*1000)
                    samples.append(row);record(row)
                aggregate=dict(groups=count,workload=workload,medians={key:statistics.median(r[key] for r in samples if not r['warmup']) for key in ('wall_ms','cuda_ms')})
                aggregates.append(aggregate)
            del gpu,gate,up,down,staging,pageable
    write_json(output/'batch-summary.json',{'aggregates':aggregates,'groups':batch_counts(parent,cfg)})
    return aggregates


@torch.inference_mode()
def run(config_path, run_path, output_path):
    cfg_path=Path(config_path).resolve();root=cfg_path.parent.parent;cfg=read(cfg_path)
    source,timing_protocol=committed_inputs(__file__,cfg_path,TIMING_PROTOCOL)
    run=Path(run_path);manifest=read(run/'manifest.json')
    verify_snapshot(run,manifest,'configs/causal-evidence.json',cfg['protocol'],__file__)
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',manifest['source_commit'],source['source_commit']],check=True)
    parent=load_inputs(root,cfg);rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    if len([r for r in rows if r['kind']=='wiki'])!=40:raise ValueError('Complete quality grid required before timing')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=environment(torch.device('cuda'));frozen_environment(env)
    output=Path(output_path);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(cfg_path,output/'config.json');shutil.copyfile(timing_protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(output/'manifest.json',{'environment':env,**source,'measurement_source_commit':manifest['source_commit'],
        'results_sha256':digest(run/'results.jsonl'),'hardware_sha256':cfg['hardware_sha256']})
    checkpoint=Path(snapshot_download(cfg['model']['model'],revision=cfg['model']['revision'],local_files_only=True))
    if {p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}!=read(run/'model.json')['checkpoint_files']:
        raise ValueError('Timing checkpoint mismatch')
    model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True,
        trust_remote_code=False).cuda().eval();mlps=extract_ffns(model)
    fit_row=next(r for r in rows if r['kind']=='fit' and r['pass_index']==2)
    def verified_tensor(receipt):
        path=run/receipt['file']
        if digest(path)!=receipt['sha256']:raise ValueError('Timing tensor receipt mismatch')
        return load_file(path)
    examples=[verified_tensor(r['examples']) for r in rows if r['kind']=='calibration' and r['pass_index']==2]
    fitted=restore_fitted_layout({key:torch.cat([d[key] for d in examples]) for key in examples[0]},
                                verified_tensor(fit_row['tensors']))
    write_json(output/'fit-layout.json',{'coefficient_bitwise_equal':True,'fit_sha256':fit_row['tensors']['sha256'],
        'strides':{k:list(v.stride()) for k,v in ce.flatten_models(fitted).items()}})
    models=ce.unflatten_models(ce.flatten_models(fitted),28,'cuda')
    del examples,fitted
    first_doc=parent['development'][0];measurements=[]
    stream=(output/'results.jsonl').open('x',encoding='utf-8')
    try:
        with packed(mlps,parent['layouts']['popularity']):
            for mode,extra in ce.CONDITIONS:
                row=next(r for r in rows if r['kind']=='wiki' and (r['mode'],r['extra'],r['document'])==(mode,extra,first_doc))
                trace=load_file(run/row['trace']['file']);n=32
                xs=trace['x'][:n].cuda();initial=trace['initial'][:n].cuda();additions=trace['additions'][:n].cuda()
                # Preparation is outside the primitive timers and explicitly recorded.
                zs=torch.stack([torch.stack([(m.act_fn(m.gate_proj(xs[t,i].reshape(1,1,-1)))*
                    m.up_proj(xs[t,i].reshape(1,1,-1))).reshape(-1) for i,m in enumerate(mlps)]) for t in range(n)])
                first_outputs=torch.stack([torch.stack([F.linear(
                    (zs[t,i]*initial[t,i].repeat_interleave(8)).reshape(1,1,-1),m.down_proj.weight).reshape(-1)
                    for i,m in enumerate(mlps)]) for t in range(n)])
                # The physical sensitivity combines resident and cold FFN outputs;
                # those vector additions are not inside either FFN primitive.
                hot=torch.tensor(parent['hot'],device='cuda')
                resident_outputs=torch.stack([torch.stack([F.linear(
                    (zs[t,i]*hot[i].repeat_interleave(8)).reshape(1,1,-1),m.down_proj.weight).reshape(-1)
                    for i,m in enumerate(mlps)]) for t in range(n)])
                cold_outputs=first_outputs-resident_outputs
                extra_outputs=torch.stack([torch.stack([F.linear(
                    (zs[t,i]*additions[t,i].repeat_interleave(8)).reshape(1,1,-1),m.down_proj.weight).reshape(-1)
                    for i,m in enumerate(mlps)]) for t in range(n)])
                norms=[m.down_proj.weight.detach().norm(dim=0) for m in mlps]
                cs=controllers(parent,models,mode,extra,'cuda')
                for token in range(n):
                    for layer,c in enumerate(cs):
                        mask=c.begin(xs[token,layer]);z=zs[token,layer]
                        observed=group_scores(z*mask.repeat_interleave(8),norms[layer],8).reshape(-1).div(c.scale).log1p()
                        chosen=c.acquire(observed,first_outputs[token,layer].norm())
                        if not torch.equal(mask,initial[token,layer]) or not torch.equal(chosen,additions[token,layer]):
                            raise ValueError('Timing fixture differs from the measured causal decisions')
                        final=mask|chosen
                        c.observe(group_scores(z*final.repeat_interleave(8),norms[layer],8).reshape(-1).div(c.scale).log1p(),final)
                for workload in TIMING_WORKLOADS:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();end.record();torch.cuda.synchronize()
                    for repetition in range(13):
                        cs=controllers(parent,models,mode,extra,'cuda')
                        torch.cuda.synchronize();start=time.perf_counter();begin.record()
                        for token in range(n):
                            for layer,(m,c) in enumerate(zip(mlps,cs,strict=True)):
                                x,z=xs[token,layer],zs[token,layer]
                                if workload=='dense_ffn':
                                    m(x.reshape(1,1,-1))
                                elif workload=='down_products':
                                    first=F.linear(z*initial[token,layer].repeat_interleave(8),m.down_proj.weight)
                                    if mode not in ('initial','one_shot','input_only'):
                                        second=F.linear(z*additions[token,layer].repeat_interleave(8),m.down_proj.weight)
                                        first+second
                                elif workload=='output_additions':
                                    combined=resident_outputs[token,layer]+cold_outputs[token,layer]
                                    if mode not in ('initial','one_shot','input_only'):
                                        combined+extra_outputs[token,layer]
                                else:
                                    mask=c.begin(x)
                                    # Dynamic acquisition starts as a GPU mask;
                                    # CPU index_select requires these host IDs.
                                    torch.nonzero(mask&~c.hot,as_tuple=False).flatten().cpu().contiguous()
                                    if mode in ('partial','fallback'):
                                        observed=group_scores(z*mask.repeat_interleave(8),norms[layer],8).reshape(-1).div(c.scale).log1p()
                                        first_norm=first_outputs[token,layer].norm()
                                    else:
                                        observed=torch.zeros_like(c.prior);first_norm=torch.zeros((),device='cuda')
                                    extra_mask=c.acquire(observed,first_norm);final=mask|extra_mask
                                    if mode not in ('initial','one_shot','input_only'):
                                        torch.nonzero(extra_mask&~c.hot,as_tuple=False).flatten().cpu().contiguous()
                                    final_scores=group_scores(z*final.repeat_interleave(8),norms[layer],8).reshape(-1).div(c.scale).log1p()
                                    c.observe(final_scores,final)
                        end.record();torch.cuda.synchronize()
                        value=dict(mode=mode,extra=extra,workload=workload,repetition=repetition,warmup=repetition<3,
                            visits=n*28,cuda_ms=begin.elapsed_time(end),wall_ms=(time.perf_counter()-start)*1000)
                        measurements.append(value);stream.write(json.dumps(value)+'\n');stream.flush()
                print('Completed cost fixture',mode,extra,flush=True)
        aggregates=[]
        for mode,extra in ce.CONDITIONS:
            for workload in TIMING_WORKLOADS:
                selected=[r for r in measurements if (r['mode'],r['extra'],r['workload'])==(mode,extra,workload) and not r['warmup']]
                aggregates.append(dict(mode=mode,extra=extra,workload=workload,
                    **{key:statistics.median(r[key] for r in selected) for key in ('wall_ms','cuda_ms')},visits=896))
        write_json(output/'summary.json',{'aggregates':aggregates,'status':'primitive_timings_completed','runtime_nomination':None})
    finally:stream.close()
    # Isolate transfer measurements from the resident model and restoration buffers.
    del model,mlps,models,cs,xs,zs,first_outputs,initial,additions,norms,m,c,x,z,resident_outputs,cold_outputs,extra_outputs
    gc.collect();torch.cuda.empty_cache()
    measure_batches(parent,cfg,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','run','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();run(args.config,args.run,args.output)


if __name__=='__main__':main()

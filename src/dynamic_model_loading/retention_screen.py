"""Bounded physical retention screen with immutable provenance and raw replay."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .fault_screen import ROOT,write,supervise,require_supervisor
from .specialist_screen import sha
from .debt_analysis import demand
from .retained_rows import retention_plan

CONFIG=ROOT/'configs/retention-screen.json'
ORDER=[(0,'packet'),(0,'retained'),(1,'retained'),(1,'packet'),(2,'packet'),(2,'retained')]


def decision(conditions):
    p,r=conditions['packet'],conditions['retained']
    demand(p['h2d_bytes']>0 and p['wall_seconds']>0,'Missing baseline denominator')
    return dict(checks=dict(Hfaithfulness=True,Hresources=True,
        Hacquisition=10*r['h2d_bytes']<=9*p['h2d_bytes'],
        Hruntime=20*r['wall_seconds']<=19*p['wall_seconds']),
        h2d_saving=1-r['h2d_bytes']/p['h2d_bytes'],wall_saving=1-r['wall_seconds']/p['wall_seconds'])


def worker(output,*,config_path=CONFIG,hybrid_order=ORDER,bank_type=None,
           bank_conditions=('packet','retained'),source_file=__file__,
           architecture=(24,2048,8192),cold_reference=False):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from safetensors.numpy import save_file
    from .provenance import committed_inputs,frozen_environment
    from .experiment import environment,cuda_memory
    from .fault_resources import Resources
    from .fault_generation import kv_bytes
    from .decision_field import kv_storage_bytes
    from .retained_rows import RetainedRows
    from .capacity_down import move_except_down
    from .relu import extract
    started=time.perf_counter(); cfg=json.loads(Path(config_path).read_text()); layer_count,hidden,neurons=architecture
    prefix_length=cfg['prefix_tokens']; generation_length=cfg['generation_tokens']
    manifest,protocol=committed_inputs(source_file,config_path,cfg['protocol'])
    demand(subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()==manifest['source_commit'],
           'Push reviewed source first')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config_path,output/'config.json'); shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent=ROOT/cfg['artifact_manifest']; shutil.copyfile(parent,output/'artifact-manifest.json')
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    artifacts=json.loads(parent.read_text())['checkpoint_files']
    for name,digest in artifacts.items(): demand(sha(path/name)==digest,'Checkpoint changed: '+name)
    torch.set_num_threads(4); torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    demand(cfg.get('dtype') in ('float32','float16'),'Unsupported precision')
    dtype=getattr(torch,cfg['dtype']); itemsize=4 if cfg['dtype']=='float32' else 2
    if itemsize==2: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
    demand(torch.cuda.is_available(),'CUDA required'); env=environment(torch.device('cuda:0')); frozen_environment(env)
    manifest.update(environment=env,cuda_execution=True,artifacts=artifacts); write(output/'manifest.json',manifest)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    prefixes=[tokenizer.encode(p,add_special_tokens=False)[:prefix_length] for p in [*cfg['prompts'],cfg['warmup_prompt']]]
    demand(all(len(p)==prefix_length for p in prefixes),'Short prompt'); write(output/'tokens.json',prefixes)
    write(output/'NOTICE.json',dict(checkpoint=cfg['repo'],license='OPT custom license; weights not redistributed',
        reference_gpu_startup=not cold_reference,capacity_claim=False))
    bank=None; episodes=[]; refs={}; limit=cfg['candidate_gpu_limit_mib']*2**20
    with torch.inference_mode(),Resources(output/'resources.jsonl',gpu_limit=cfg['reference_gpu_limit_mib']*2**20) as resources:
        setup=time.perf_counter()
        if cold_reference: torch.cuda.memory.set_per_process_memory_fraction(limit/env['gpu']['total_bytes'])
        model=AutoModelForCausalLM.from_pretrained(path,dtype=dtype,attn_implementation='sdpa',
            use_safetensors=False,weights_only=True,local_files_only=True,trust_remote_code=False).eval()
        if not cold_reference: model=model.to('cuda')
        torch.cuda.synchronize(); layers=extract(model)
        demand(len(layers)==layer_count and all(l.fc2.weight.shape==(hidden,neurons) for l in layers)
            and model.config.vocab_size==50272 and model.config.eos_token_id==2,'Wrong architecture')
        demand({p.dtype for p in model.parameters()}=={dtype},'Model precision changed')
        allocation=dict(resident_setup_seconds=time.perf_counter()-setup,
            resident_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            resident_cuda=cuda_memory(torch.device('cuda:0')))
        if cold_reference:
            from .scale_support import construct
            bank,allocation=construct(model,layers,bank_type,cfg,resources,setup)
            allocation['worker_started']=started

        def episode(doc,condition,warmup=False,reference=False):
            resources.boundary(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            begin=time.perf_counter(); key=f'episode-{len(episodes)}'; folder=output/key; folder.mkdir()
            arrays={}; pages=[]; calls=[]; ids=[]
            if bank is not None:
                bank.begin(key,condition)
                def record(row):
                    activity=row.pop('activity')
                    if activity is not None: arrays[f'activity.{row["call"]}']=activity
                    pages.append(row)
                bank.record=record
            cache=DynamicCache(config=model.config); tokens=torch.tensor([prefixes[doc]],device='cuda')
            h2d=prefix_length*8; d2h=0
            for step in range(generation_length):
                resources.check(); torch.cuda.synchronize(); callstart=time.perf_counter()
                actual=tokens.cpu().tolist()[0]; d2h+=len(actual)*8
                result=model(tokens,past_key_values=cache,use_cache=True)
                values=result.logits[0,-1].float().cpu().numpy().copy(); d2h+=values.nbytes
                arrays[f'logits.{step}']=values; token=int(values.argmax()); ids.append(token)
                torch.cuda.synchronize(); finish=time.perf_counter()
                calls.append(dict(step=step,input_ids=actual,token=token,started=callstart,finished=finish,
                    kv_length=cache.get_seq_length(),kv_bytes=kv_bytes(cache),kv_storage_bytes=kv_storage_bytes(cache)))
                del result
                if token==2 or len(ids)==generation_length: break
                tokens=torch.tensor([[token]],device='cuda'); h2d+=8
            write(folder/'calls.json',calls); write(folder/'pages.json',pages); save_file(arrays,folder/'tensors.safetensors')
            # FP16 footprint comparisons require a sample inside even a short
            # resident episode. This fixed-cost boundary is charged to all FP16 modes.
            if itemsize==2: resources.boundary()
            finish=time.perf_counter(); mem=cuda_memory(torch.device('cuda:0')); resources.boundary()
            # Resources.boundary records/checks but intentionally returns no sample.
            if bank is not None:
                demand(mem['peak_reserved_bytes']<=limit,'Candidate allocator limit exceeded')
                demand(resources.sample()['gpu_used']<=limit,'Candidate sampled GPU limit exceeded')
            row=dict(episode=key,document=doc,condition=condition,warmup=warmup,ids=ids,
                stop_reason='eos' if ids[-1]==2 else 'length',started=begin,finished=finish,
                wall_seconds=finish-begin,ttft_seconds=calls[0]['finished']-begin,
                explicit_copies=dict(h2d_bytes=h2d,d2h_bytes=d2h),cuda=mem)
            if cold_reference: row['reference']=reference
            write(folder/'episode.json',row); episodes.append(row)
            print(f'{key} doc{doc} {condition} warmup={warmup}: {len(ids)} tokens {row["wall_seconds"]:.3f}s',flush=True)
            logits=np.stack([arrays[f'logits.{s}'] for s in range(len(ids))])
            if condition=='resident' or reference: refs[doc]=(ids,row['stop_reason'],logits)
            else:
                reference=refs[doc]; demand((ids,row['stop_reason'])==reference[:2],'Greedy mismatch')
                demand(np.all(np.linalg.norm(logits.astype(np.float64)-reference[2],axis=1)/
                    np.maximum(np.linalg.norm(reference[2].astype(np.float64),axis=1),1e-30)<=1e-5),'Logit mismatch')

        reference_condition='stream' if cold_reference else 'resident'
        episode(3,reference_condition,True,reference=True)
        for d in range(3): episode(d,reference_condition,reference=True)
        if not cold_reference:
            setup=time.perf_counter(); model.cpu(); torch.cuda.empty_cache()
            allocation['before_candidate_cuda']=cuda_memory(torch.device('cuda:0'))
            bank=(bank_type or RetainedRows)(layers,capacity=cfg['cache_rows'])
            moved=move_except_down(model,layers); torch.cuda.synchronize()
            allocation.update(bank.allocation(),candidate_setup_seconds=time.perf_counter()-setup,
                construction_h2d_bytes=moved,reference_return_d2h_bytes=allocation['resident_parameter_bytes'],
                candidate_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda),
                candidate_cuda=cuda_memory(torch.device('cuda:0')))
        for c in bank_conditions: episode(3,c,True)
        for d,c in hybrid_order: episode(d,c)
        write(output/'episodes.json',episodes); write(output/'allocation.json',allocation); resources.boundary()
    write(output/'completion.json',dict(complete=True,worker_inner_seconds=time.perf_counter()-started,
        resources=resources.receipt(),full_suite_launched=False))
    write(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


def audit_pages(pages,arrays,calls,condition,capacity,*,itemsize=4,architecture=(24,2048,8192)):
    layer_count,hidden,neurons=architecture
    states=[OrderedDict() for _ in range(layer_count)]; totals=dict(weight_h2d_bytes=0,metadata_h2d_bytes=0,
        activity_d2h_bytes=0,hits=0,active=0,prefill_h2d_bytes=0,decode_h2d_bytes=0)
    demand(len(pages)==len(calls)*layer_count,'Missing layer calls')
    for j,r in enumerate(pages):
        step,layer=divmod(j,layer_count); n=len(calls[step]['input_ids']); packed=arrays[f'activity.{j+1}']
        demand(packed.dtype==np.uint8 and packed.shape==(n,neurons//8),'Activity shape changed')
        active=np.flatnonzero(np.unpackbits(packed,axis=1).any(axis=0)).tolist()
        hits,misses,inserts=retention_plan(states[layer],active,capacity if condition=='retained' else 0)
        demand(r['call']==j+1 and r['layer']==layer and r['condition']==condition and r['tokens']==n,'Call identity drift')
        demand(r['active']==active and r['hits']==[list(x) for x in hits] and r['misses']==misses
            and r['inserts']==[list(x) for x in inserts] and r['residency']==[list(x) for x in states[layer].items()],
            'Residency replay mismatch')
        weight=len(misses)*hidden*itemsize; meta=len(misses)*8+16*(len(hits)+len(inserts)); d2h=n*neurons+1
        for k,v in [('weight_h2d_bytes',weight),('metadata_h2d_bytes',meta),('activity_d2h_bytes',d2h)]:
            demand(r[k]==v,'Physical byte mismatch: '+k); totals[k]+=v
        clocks=[r[k] for k in ('started','selection_finished','acquisition_finished','finished')]
        demand(all(np.isfinite(clocks)) and clocks==sorted(clocks) and calls[step]['started']<=clocks[0]
            <=clocks[-1]<=calls[step]['finished'],'Physical clock mismatch')
        totals['hits']+=len(hits); totals['active']+=len(active)
        totals['prefill_h2d_bytes' if step==0 else 'decode_h2d_bytes']+=weight+meta
    return totals


def analyze(output,*,config_relative='configs/retention-screen.json',hybrid_order=ORDER,
            bank_conditions=('packet','retained'),page_auditor=audit_pages,decider=decision,
            architecture=(24,2048,8192),cold_reference=False):
    from safetensors.numpy import load_file
    from .provenance import verify_snapshot,frozen_environment
    from .specialist_analysis import audit_resources
    from .debt_analysis import read_rows
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    output=Path(output); supervisor=require_supervisor(output)
    cfg=json.loads((output/'config.json').read_text()); m=json.loads((output/'manifest.json').read_text())
    itemsize=4 if cfg['dtype']=='float32' else 2; layer_count,hidden,neurons=architecture
    prefix_length=cfg['prefix_tokens']; generation_length=cfg['generation_tokens']
    verify_snapshot(output,m,config_relative,cfg['protocol'],__file__); frozen_environment(m['environment'])
    demand(m['cuda_execution'] is True and m['environment']['cpu_threads']==4 and
        not m['environment']['tf32_matmul'] and not m['environment']['tf32_cudnn'],'Execution drift')
    inventory=json.loads((output/'files.json').read_text())
    demand(set(inventory)=={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='files.json'},'Inventory drift')
    for name,digest in inventory.items(): demand(sha(output/name)==digest,'Raw bytes changed')
    parent=subprocess.check_output(['git','show',m['source_commit']+':'+cfg['artifact_manifest']],cwd=ROOT)
    demand((output/'artifact-manifest.json').read_bytes()==parent and
        m['artifacts']==json.loads(parent)['checkpoint_files'],'Model identity drift')
    completion=json.loads((output/'completion.json').read_text()); samples=list(read_rows(output/'resources.jsonl'))
    audit_resources(samples,completion['resources']); demand(completion['complete'] and not completion['full_suite_launched']
        and 0<completion['worker_inner_seconds']<=supervisor['worker_wall_seconds'],'Incomplete run')
    demand(all(s['gpu_used']<=cfg['reference_gpu_limit_mib']*2**20 for s in samples),'Reference GPU allowance failed')
    episodes=json.loads((output/'episodes.json').read_text()); prefixes=json.loads((output/'tokens.json').read_text())
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    for name in ('config.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json'):
        demand(sha(path/name)==m['artifacts'][name],'Tokenizer changed')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    demand(prefixes==[tokenizer.encode(p,add_special_tokens=False)[:prefix_length]
        for p in [*cfg['prompts'],cfg['warmup_prompt']]],'Tokenization changed')
    reference_condition='stream' if cold_reference else 'resident'
    expected=[(3,reference_condition,True),*[(d,reference_condition,False) for d in range(3)],
        *[(3,c,True) for c in bank_conditions],*[(d,c,False) for d,c in hybrid_order]]
    demand([(r['document'],r['condition'],r['warmup']) for r in episodes]==expected,'Episode order drift')
    refs={}; metrics=[]; conditions={}; previous=0
    for i,r in enumerate(episodes):
        folder=output/f'episode-{i}'; demand(r==json.loads((folder/'episode.json').read_text()),'Episode changed')
        calls=json.loads((folder/'calls.json').read_text()); pages=json.loads((folder/'pages.json').read_text())
        arrays=load_file(folder/'tensors.safetensors'); ids=[]; n=len(calls); d2h=0
        demand(1<=n<=generation_length and r['started']>=previous and r['finished']>r['started']
            and r['wall_seconds']==r['finished']-r['started'],'Episode clocks invalid'); previous=r['finished']
        values=[]
        for j,c in enumerate(calls):
            a=arrays[f'logits.{j}']; demand(a.shape==(50272,) and a.dtype==np.float32 and np.isfinite(a).all(),'Bad logits')
            demand(c['step']==j and c['input_ids']==(prefixes[r['document']] if j==0 else [ids[-1]])
                and c['kv_length']==prefix_length+j and c['kv_bytes']==c['kv_storage_bytes']==(prefix_length+j)*2*layer_count*hidden*itemsize,'KV/trajectory drift')
            demand(r['started']<=c['started']<c['finished']<=r['finished'] and (j==0 or calls[j-1]['finished']<=c['started']),'Call clock drift')
            token=int(a.argmax()); demand(token==c['token'] and (j==n-1 or token!=2),'Token drift'); ids.append(token); values.append(a)
            d2h+=len(c['input_ids'])*8+a.nbytes
        stop='eos' if ids[-1]==2 else 'length'
        demand(ids==r['ids'] and r['stop_reason']==stop and (stop=='eos' or n==generation_length),'Generation changed')
        demand(r['explicit_copies']==dict(h2d_bytes=prefix_length*8+8*(n-1),d2h_bytes=d2h)
            and r['ttft_seconds']==calls[0]['finished']-r['started'],'Token copy/TTFT mismatch')
        values=np.stack(values)
        if r['condition']=='resident' or (cold_reference and i<4):
            if cold_reference:
                demand(r['reference'] is True and r['condition']=='stream','Wrong cold reference')
                stats=page_auditor(pages,arrays,calls,r['condition'],cfg['cache_rows'])
            else:
                demand(not pages,'Unexpected reference paging'); stats={}
            refs[r['document']]=(ids,stop,values)
        else:
            if cold_reference: demand(r['reference'] is False,'Candidate relabeled as reference')
            ref=refs[r['document']]; demand((ids,stop)==ref[:2],'Reference IDs differ')
            error=np.linalg.norm(values.astype(np.float64)-ref[2],axis=1)/np.maximum(np.linalg.norm(ref[2].astype(np.float64),axis=1),1e-30)
            demand(np.all(error<=1e-5),'Numerical contract failed')
            stats=page_auditor(pages,arrays,calls,r['condition'],cfg['cache_rows'])
            metrics.append(dict(episode=r['episode'],max_relative_l2=float(error.max()),exact=np.array_equal(values,ref[2])))
            demand(r['cuda']['peak_reserved_bytes']<=4800*2**20 and
                all(s['gpu_used']<=4800*2**20 for s in samples if r['started']<=s['monotonic']<=r['finished']), 'Candidate cap failed')
        demand(set(arrays)=={f'logits.{j}' for j in range(n)}|({f'activity.{j+1}' for j in range(len(pages))}
            if r['condition']!='stream' else set()),'Unaccounted tensors')
        if not r['warmup']:
            group=conditions.setdefault(r['condition'],dict(tokens=0,wall_seconds=0.,h2d_bytes=0,prefill_seconds=0.,decode_seconds=0.))
            group['tokens']+=n; group['wall_seconds']+=r['wall_seconds']; group['prefill_seconds']+=calls[0]['finished']-calls[0]['started']
            group['decode_seconds']+=sum(c['finished']-c['started'] for c in calls[1:])
            group['h2d_bytes']+=stats.get('weight_h2d_bytes',0)+stats.get('metadata_h2d_bytes',0)
            for k in ('hits','active','prefill_h2d_bytes','decode_h2d_bytes'): group[k]=group.get(k,0)+stats.get(k,0)
    a=json.loads((output/'allocation.json').read_text())
    demand(a['cache_bytes']==layer_count*cfg['cache_rows']*hidden*itemsize and a['workspace_bytes']==neurons*hidden*itemsize
        and a['packet_cuda_bytes']==a['pinned_bytes']==neurons*(8+hidden*itemsize)
        and a['candidate_parameter_bytes']==a['construction_h2d_bytes']==cfg.get('expected_candidate_parameters_bytes',913104896*itemsize),'Allocation mismatch')
    result=decider(conditions)
    return dict(conditions=conditions,metrics=metrics,**result,
        allocation=a,resources=completion['resources'],source_commit=m['source_commit'],
        decision='eligible_for_separate_followup' if all(result['checks'].values()) else 'stop_this_retention_candidate',
        full_suite_launched=False,native_admission_evaluated=False,capacity_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.retention_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

"""Ordinary resident FP16 as a bounded-access control, not a new algorithm."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .fault_screen import ROOT,write,supervise,require_supervisor
from .capacity_screen import FROZEN as CAPACITY
from .capacity_support import LIMIT,bounded,copy_references,load_references
from .specialist_screen import sha
from .debt_analysis import demand,read_rows

CONFIG=ROOT/'configs/resident-precision.json'
FROZEN={k:v for k,v in CAPACITY.items() if k not in
    ('protocol','minimum_traffic_saving','minimum_wall_saving','maximum_packet_episode_seconds')}
FROZEN.update(protocol='docs/resident-precision-protocol.md',parameter_dtype='float16',kv_dtype='float16',
    readback_dtype='float32',maximum_episode_seconds=5,fp16_reduced_precision_reduction=True,order=[0,1])


def validate_config(cfg):
    if cfg!=FROZEN: raise ValueError('Frozen resident precision control changed')


def aligned_metrics(reference,candidate):
    """Compare prediction states only while consumed greedy prefixes agree."""
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    demand(a.ndim==b.ndim==2 and a.shape[1]==b.shape[1] and 0<len(b)<=len(a)
           and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid precision comparison')
    wanted,actual=a.argmax(1),b.argmax(1)
    differences=np.flatnonzero(wanted[:len(b)]!=actual)
    first=int(differences[0]) if len(differences) else None
    aligned=first+1 if first is not None else len(b)
    errors=np.sqrt(np.sum((a[:aligned]-b[:aligned])**2,axis=1)/np.maximum(np.sum(a[:aligned]**2,axis=1),1e-30))
    same=first is None and len(a)==len(b)
    return dict(first_divergence=first,aligned_positions=aligned,relative_l2=errors.tolist(),
        all_ids_match=bool(same),fp32_numerical=bool(same and np.all(errors<=1e-5)))


def decision(metrics,episodes,worker_seconds):
    ids=all(m['all_ids_match'] for m in metrics)
    useful=all(r['wall_seconds']<=5 for r in episodes) and worker_seconds<=60
    return dict(checks=dict(Hordinary_access=ids and useful,
        Hfp32_numerical=all(m['fp32_numerical'] for m in metrics)),useful_latency_passed=useful,
        decision='ordinary_fp16_also_provides_tested_access' if ids and useful else 'ordinary_precision_access_unresolved',
        native_admission_evaluated=False,full_suite_launched=False,course_delivery_number=6,
        paired_packet_speedup_measured=False)


def audit_episode(row,calls,tensors,prefix,reference):
    n=len(calls)
    demand(1<=n<=8 and set(tensors)=={f'logits.{j}' for j in range(n)},'Missing generated rows')
    ids=[]; before=row['started']; d2h=0
    for j,c in enumerate(calls):
        logits=tensors[f'logits.{j}']
        demand(logits.shape==(50272,) and logits.dtype==np.float32 and np.isfinite(logits).all(),'Invalid logits')
        demand(c['step']==j and c['input_ids']==(prefix if j==0 else [ids[-1]])
            and c['kv_length']==16+j and c['kv_bytes']==c['kv_storage_bytes']==(16+j)*196608
            and c['kv_dtypes']==['torch.float16'] and c['logit_dtype']=='torch.float16','Own trajectory or precision drift')
        demand(before<=c['started']<c['finished']<=row['finished'],'Call clock invalid'); before=c['finished']
        ids.append(int(logits.argmax())); demand(c['token']==ids[-1] and (j==n-1 or ids[-1]!=2),'Greedy commit/EOS mismatch')
        d2h+=len(c['input_ids'])*8+logits.nbytes
    stop='eos' if ids[-1]==2 else 'length'
    demand(ids==row['ids'] and row['stop_reason']==stop and (stop=='eos' or n==8),'Output mismatch')
    demand(row['ttft_seconds']==calls[0]['finished']-row['started']
        and row['raw_tensor_bytes']==n*50272*4,'TTFT/raw byte drift')
    expected=dict(h2d_bytes=128+8*(n-1),d2h_bytes=d2h)
    demand(row['explicit_copies']==expected,'Transfer accounting changed')
    ref,values=reference
    metric=aligned_metrics(values,np.stack([tensors[f'logits.{j}'] for j in range(n)]))
    metric['all_ids_match']=metric['all_ids_match'] and stop==ref['stop_reason']
    metric['fp32_numerical']=metric['fp32_numerical'] and metric['all_ids_match']
    return metric,expected


def worker(output):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from safetensors.numpy import save_file
    from .provenance import committed_inputs,frozen_environment
    from .experiment import environment,cuda_memory
    from .fault_resources import Resources
    from .fault_generation import kv_bytes
    from .decision_field import kv_storage_bytes
    started=time.perf_counter(); cfg=json.loads(CONFIG.read_text()); validate_config(cfg)
    manifest,protocol=committed_inputs(__file__,CONFIG,cfg['protocol'])
    demand(subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()==manifest['source_commit'],
           'Push reviewed source before inference')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(CONFIG,output/'config.json'); shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent=ROOT/cfg['artifact_manifest']; shutil.copyfile(parent,output/'artifact-manifest.json')
    artifacts={n:dict(bytes=cfg['artifact_bytes'][n],sha256=s) for n,s in json.loads(parent.read_text())['checkpoint_files'].items()}
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    for name,expected in artifacts.items():
        demand((path/name).stat().st_size==expected['bytes'] and sha(path/name)==expected['sha256'],'Checkpoint changed')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    prefixes=[tokenizer.encode(p,add_special_tokens=False)[:16] for p in cfg['prompts']]
    demand(all(len(p)==16 for p in prefixes),'Short prompt'); write(output/'tokens.json',prefixes)
    copy_references(output); load_references(output,prefixes)
    write(output/'NOTICE.json',dict(checkpoint=cfg['repo'],revision=cfg['revision'],
        license='OPT custom license; checkpoint external, not redistributed',
        observations='Own-trajectory FP16 continuations and external FP32 reference logits'))
    torch.set_num_threads(4); torch.manual_seed(20260919)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
    demand(torch.cuda.is_available(),'CUDA required')
    env=environment(torch.device('cuda:0')); frozen_environment(env)
    manifest.update(environment=env,cuda_execution=True,artifacts=artifacts,artifact_manifest_sha256=sha(parent))
    write(output/'manifest.json',manifest); episodes=[]
    with torch.inference_mode(),Resources(output/'resources.jsonl',gpu_limit=LIMIT) as resources:
        resources.boundary(); setup=time.perf_counter()
        torch.cuda.memory.set_per_process_memory_fraction(LIMIT/env['gpu']['total_bytes'])
        before=cuda_memory(torch.device('cuda:0')); bounded(before)
        demand(before['allocated_bytes']==0,'Hidden startup CUDA tensors')
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16,attn_implementation='sdpa',
            use_safetensors=False,weights_only=True,local_files_only=True,trust_remote_code=False).eval()
        demand(all(p.device.type=='cpu' and p.dtype==torch.float16 for p in model.parameters()),'Wrong initial representation')
        parameters=sum(p.numel()*p.element_size() for p in model.parameters())
        buffers=sum(b.numel()*b.element_size() for b in model.buffers())
        demand(parameters==2631516160 and buffers==0 and model.config.eos_token_id==2
               and model.config.vocab_size==50272,'Unexpected resident architecture')
        model.to('cuda'); torch.cuda.synchronize()
        allocation=dict(parameters_bytes=parameters,buffers_bytes=buffers,construction_h2d_bytes=parameters+buffers,
            parameter_dtype='torch.float16',all_parameters_cuda=all(p.is_cuda for p in model.parameters()),
            tied_embeddings=model.lm_head.weight is model.model.decoder.embed_tokens.weight,
            allocator_fraction=torch.cuda.memory.get_per_process_memory_fraction(),before_cuda=before,
            resident_cuda=cuda_memory(torch.device('cuda:0')),setup_seconds=time.perf_counter()-setup)
        bounded(allocation['resident_cuda']); resources.boundary()
        for doc in (0,1):
            resources.boundary(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            begin=time.perf_counter(); folder=output/f'episode-{doc}'; folder.mkdir()
            cache=DynamicCache(config=model.config); tokens=torch.tensor([prefixes[doc]],device='cuda')
            tensors={}; calls=[]; ids=[]; h2d=128; d2h=0
            for step in range(8):
                resources.check(); torch.cuda.synchronize(); callstart=time.perf_counter()
                actual=tokens.cpu().tolist()[0]; d2h+=len(actual)*8
                result=model(tokens,past_key_values=cache,use_cache=True)
                original_dtype=str(result.logits.dtype)
                logits=result.logits[0,-1].float().cpu().numpy().copy(); d2h+=logits.nbytes
                tensors[f'logits.{step}']=logits; token=int(logits.argmax()); ids.append(token)
                torch.cuda.synchronize(); finish=time.perf_counter()
                calls.append(dict(step=step,input_ids=actual,token=token,started=callstart,finished=finish,
                    kv_length=cache.get_seq_length(),kv_bytes=kv_bytes(cache),kv_storage_bytes=kv_storage_bytes(cache),
                    kv_dtypes=sorted({str(t.dtype) for l in cache.layers for t in (l.keys,l.values)}),
                    logit_dtype=original_dtype))
                del result
                if token==2 or len(ids)==8: break
                tokens=torch.tensor([[token]],device='cuda'); h2d+=8
            write(folder/'calls.json',calls); save_file(tensors,folder/'tensors.safetensors')
            finish=time.perf_counter()
            row=dict(document=doc,condition='resident_fp16',ids=ids,stop_reason='eos' if ids[-1]==2 else 'length',
                started=begin,finished=finish,wall_seconds=finish-begin,ttft_seconds=calls[0]['finished']-begin,
                explicit_copies=dict(h2d_bytes=h2d,d2h_bytes=d2h),raw_tensor_bytes=sum(a.nbytes for a in tensors.values()),
                cuda=cuda_memory(torch.device('cuda:0')))
            write(folder/'episode.json',row); episodes.append(row); bounded(row['cuda']); resources.boundary()
            print(f'doc{doc} residentFP16: {len(ids)} tokens {row["wall_seconds"]:.3f}s',flush=True)
        write(output/'episodes.json',episodes); write(output/'allocation.json',allocation); resources.boundary()
    write(output/'completion.json',dict(complete=True,worker_inner_seconds=time.perf_counter()-started,
        resources=resources.receipt(),full_suite_launched=False))
    write(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


def analyze(output):
    from .provenance import verify_snapshot,frozen_environment
    from .specialist_analysis import audit_resources
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from safetensors.numpy import load_file
    output=Path(output); supervisor=require_supervisor(output)
    cfg=json.loads((output/'config.json').read_text()); validate_config(cfg)
    m=json.loads((output/'manifest.json').read_text())
    verify_snapshot(output,m,'configs/resident-precision.json',cfg['protocol'],__file__)
    env=m['environment']; frozen_environment(env)
    demand(m['cuda_execution'] is True and env['device']=='cuda:0' and env['cpu_threads']==4
        and not env['tf32_matmul'] and not env['tf32_cudnn'] and env['fp16_reduced_precision_reduction'] is True,'Execution drift')
    inventory=json.loads((output/'files.json').read_text())
    demand(set(inventory)=={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p!=output/'files.json'},'Inventory drift')
    for name,digest in inventory.items(): demand(sha(output/name)==digest,'Raw hash mismatch')
    parent=output/'artifact-manifest.json'
    original=subprocess.check_output(['git','show',m['source_commit']+':'+cfg['artifact_manifest']],cwd=ROOT)
    demand(parent.read_bytes()==original and sha(parent)==m['artifact_manifest_sha256'],'Parent artifact source changed')
    expected={n:dict(bytes=cfg['artifact_bytes'][n],sha256=s) for n,s in json.loads(original)['checkpoint_files'].items()}
    demand(m['artifacts']==expected,'Artifact inventory changed')
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    for name in ('config.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json'):
        demand(sha(path/name)==expected[name]['sha256'],'Tokenizer changed')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    prefixes=json.loads((output/'tokens.json').read_text())
    demand(prefixes==[tokenizer.encode(p,add_special_tokens=False)[:16] for p in cfg['prompts']],'Tokenization drift')
    refs=load_references(output,prefixes)
    completion=json.loads((output/'completion.json').read_text()); samples=list(read_rows(output/'resources.jsonl'))
    audit_resources(samples,completion['resources']); demand(all(r['gpu_used']<=LIMIT for r in samples),'GPU cap failed')
    demand(completion['complete'] is True and completion['full_suite_launched'] is False
        and 0<completion['worker_inner_seconds']<=supervisor['worker_wall_seconds'],'Incomplete worker')
    a=json.loads((output/'allocation.json').read_text())
    demand(a['parameters_bytes']==a['construction_h2d_bytes']==2631516160 and a['buffers_bytes']==0
        and a['parameter_dtype']=='torch.float16' and a['all_parameters_cuda'] is True and a['tied_embeddings'] is True
        and a['allocator_fraction']==LIMIT/env['gpu']['total_bytes'],'Resident representation accounting changed')
    for k in ('before_cuda','resident_cuda'): bounded(a[k])
    demand(a['before_cuda']['allocated_bytes']==0 and a['resident_cuda']['allocated_bytes']>=a['parameters_bytes']
        and math.isfinite(a['setup_seconds']) and 0<a['setup_seconds']<supervisor['worker_wall_seconds'],'Setup accounting invalid')
    episodes=json.loads((output/'episodes.json').read_text())
    demand([r['document'] for r in episodes]==[0,1],'Episode matrix changed')
    previous=samples[0]['monotonic']; metrics=[]; copies=dict(h2d_bytes=0,d2h_bytes=0)
    for doc,row in enumerate(episodes):
        folder=output/f'episode-{doc}'; demand(row==json.loads((folder/'episode.json').read_text()),'Episode drift')
        demand(row['condition']=='resident_fp16' and previous<=row['started']<row['finished']<=samples[-1]['monotonic']
            and row['wall_seconds']==row['finished']-row['started'],'Episode clock invalid')
        previous=row['finished']; bounded(row['cuda'])
        tensors=load_file(folder/'tensors.safetensors'); calls=json.loads((folder/'calls.json').read_text())
        metric,expected=audit_episode(row,calls,tensors,prefixes[doc],refs[doc])
        for k,v in expected.items(): copies[k]+=v
        metrics.append(metric)
    return dict(source_commit=m['source_commit'],metrics=metrics,episodes=episodes,allocation=a,
        resources=completion['resources'],episode_explicit_copies=copies,tokens=sum(len(r['ids']) for r in episodes),
        wall_seconds=sum(r['wall_seconds'] for r in episodes),reference_reused=True,
        **decision(metrics,episodes,supervisor['worker_wall_seconds']))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.resident_precision',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

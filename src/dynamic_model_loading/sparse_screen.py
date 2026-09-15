"""Frozen exact-zero outgoing-weight acquisition screen; no speculative target."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from .fault_screen import ROOT, write, supervise
from .specialist_screen import sha

CONFIG = ROOT/'configs/sparse-down-screen.json'


def validate_config(cfg):
    fixed = dict(protocol='docs/sparse-down-screen-protocol.md', repo='facebook/opt-1.3b',
        revision='3f5c25d0bc631cb57ac65913f76e22c2dfb61d62',
        artifact_manifest='results/relu-control-20260911/run/manifest.json',
        page_width=128,prefix_tokens=16,generation_tokens=8,cpu_threads=4,
        worker_timeout_seconds=300,gpu_limit_mib=15000,host_floor_mib=2048,
        relative_l2_tolerance=1e-5,minimum_traffic_saving=.1,minimum_wall_saving=.05)
    if any(cfg.get(k)!=v for k,v in fixed.items()) or len(cfg.get('prompts',[]))!=2:
        raise ValueError('Frozen sparse screen changed')


def worker(output):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from .experiment import environment, cuda_memory
    from .fault_generation import kv_bytes
    from .fault_resources import Resources
    from .provenance import committed_inputs, frozen_environment
    from .relu import extract
    from .sparse_down import SparseDown
    from .sparse_analysis import compare, bounded_cuda

    started=time.perf_counter()
    cfg=json.loads(CONFIG.read_text())
    validate_config(cfg)
    manifest,protocol=committed_inputs(__file__,CONFIG,cfg['protocol'])
    if subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()!=manifest['source_commit']:
        raise ValueError('Push reviewed source before inference')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(CONFIG,output/'config.json'); shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent=ROOT/cfg['artifact_manifest']
    shutil.copyfile(parent,output/'artifact-manifest.json')
    artifacts={name:dict(bytes=cfg['artifact_bytes'][name],sha256=digest)
               for name,digest in json.loads(parent.read_text())['checkpoint_files'].items()}
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    for name,expected in artifacts.items():
        if (path/name).stat().st_size!=expected['bytes'] or sha(path/name)!=expected['sha256']:
            raise ValueError('Checkpoint artifact changed: '+name)
    torch.set_num_threads(4); torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    env=environment(torch.device('cuda:0')); frozen_environment(env)
    manifest.update(environment=env,cuda_execution=True,artifacts=artifacts,
                    artifact_manifest_sha256=sha(parent))
    write(output/'manifest.json',manifest)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    prefixes=[tokenizer.encode(p,add_special_tokens=False)[:16] for p in cfg['prompts']]
    if any(len(p)!=16 for p in prefixes): raise ValueError('Short prompt')
    write(output/'tokens.json',prefixes)
    write(output/'NOTICE.json',dict(checkpoint=cfg['repo'],revision=cfg['revision'],
        license='OPT custom license; checkpoint external, not redistributed',
        observations='Authored prompt continuations and execution receipts; no checkpoint weights in archive'))
    episodes=[]; refs={}; bank=None
    with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
        resources.boundary()
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,
            attn_implementation='sdpa',use_safetensors=False,weights_only=True,
            local_files_only=True,trust_remote_code=False).eval().to('cuda')
        layers=extract(model)
        if (len(layers)!=24 or model.config.eos_token_id!=2 or model.config.vocab_size!=50272
            or any(layer.fc2.weight.shape!=(2048,8192) for layer in layers)
            or {p.dtype for p in model.parameters()}!={torch.float32}):
            raise ValueError('Frozen OPT architecture or dtype changed')
        allocation=dict(parameters_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            registered_buffers_bytes=sum(p.numel()*p.element_size() for p in model.buffers()))
        allocation['construction_h2d_bytes']=sum(allocation.values())
        allocation['resident_cuda']=cuda_memory(torch.device('cuda:0'))
        bounded_cuda(allocation['resident_cuda'])
        def episode(doc,condition):
            resources.boundary(); bounded_cuda(cuda_memory(torch.device('cuda:0')))
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            begin=time.perf_counter(); key=f'episode-{len(episodes)}'
            folder=output/key; folder.mkdir()
            tensors={}; calls=[]; pages=[]
            def record(row):
                activity=row.pop('activity')
                if activity is not None: tensors[f'activity.{row["call"]}']=activity
                pages.append(row)
            if bank is not None and condition in ('stream','sparse'):
                bank.record=record; bank.begin(key,condition)
            cache=DynamicCache(config=model.config)
            tokens=torch.tensor([prefixes[doc]],device='cuda')
            copies=dict(h2d_bytes=128,d2h_bytes=0)
            ids=[]; first=None
            for step in range(8):
                resources.check(); torch.cuda.synchronize(); callstart=time.perf_counter()
                actual=tokens.cpu().tolist()[0]; copies['d2h_bytes']+=len(actual)*8
                result=model(tokens,past_key_values=cache,use_cache=True)
                logits=result.logits[0,-1].cpu().numpy().copy()
                copies['d2h_bytes']+=logits.nbytes
                tensors[f'logits.{step}']=logits
                token=int(logits.argmax()); ids.append(token)
                torch.cuda.synchronize(); finish=time.perf_counter()
                calls.append(dict(step=step,input_ids=actual,token=token,kv_length=cache.get_seq_length(),
                    kv_bytes=kv_bytes(cache),started=callstart,finished=finish))
                if first is None: first=finish-begin
                del result
                if token==2 or len(ids)==8: break
                tokens=torch.tensor([[token]],device='cuda'); copies['h2d_bytes']+=8
            stop='eos' if ids[-1]==2 else 'length'
            write(folder/'calls.json',calls); write(folder/'pages.json',pages)
            save_file(tensors,folder/'tensors.safetensors')
            finish=time.perf_counter()
            row=dict(episode=key,document=doc,condition=condition,ids=ids,stop_reason=stop,
                started=begin,finished=finish,wall_seconds=finish-begin,ttft_seconds=first,
                explicit_copies=copies,weight_h2d_bytes=sum(r['weight_h2d_bytes'] for r in pages),
                raw_tensor_bytes=sum(a.nbytes for a in tensors.values()),
                activity_d2h_bytes=sum(r['activity_d2h_bytes'] for r in pages),
                load_extents=sum(len(r['extents']) for r in pages),cuda=cuda_memory(torch.device('cuda:0')))
            write(folder/'episode.json',row); episodes.append(row)
            bounded_cuda(row['cuda'])
            resources.boundary()
            print(f'{key} {condition}: {len(ids)} tokens {row["wall_seconds"]:.3f}s',flush=True)
            arrays=np.stack([tensors[f'logits.{s}'] for s in range(len(ids))])
            if condition=='resident': refs[doc]=(row,arrays)
            else:
                ref,a=refs[doc]
                if ids!=ref['ids'] or stop!=ref['stop_reason']: raise ValueError('Greedy reference mismatch')
                compare(a,arrays)
        episode(0,'resident'); episode(1,'resident')
        construction=time.perf_counter(); bank=SparseDown(layers,width=128)
        allocation.update(bank.allocation(),conversion_seconds=time.perf_counter()-construction,
            hybrid_cuda_parameters_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda),
            hybrid_cuda=cuda_memory(torch.device('cuda:0')))
        for doc,condition in [(0,'stream'),(0,'sparse'),(1,'sparse'),(1,'stream')]: episode(doc,condition)
        restore=time.perf_counter(); allocation['restore_h2d_bytes']=bank.restore()
        allocation['restore_seconds']=time.perf_counter()-restore
        allocation['restore_cuda']=cuda_memory(torch.device('cuda:0'))
        episode(0,'repeat'); episode(1,'repeat')
        write(output/'episodes.json',episodes); write(output/'allocation.json',allocation)
        resources.boundary()
    write(output/'completion.json',dict(complete=True,worker_inner_seconds=time.perf_counter()-started,
        resources=resources.receipt(),full_suite_launched=False))
    write(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--worker',action='store_true')
    p.add_argument('--analyze',action='store_true'); args=p.parse_args()
    from .sparse_analysis import analyze
    if args.worker: worker(args.output)
    elif args.analyze: print(json.dumps(analyze(args.output),indent=2))
    else:
        result=supervise(args.output,module='dynamic_model_loading.sparse_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout','interrupted'): raise SystemExit(1)


if __name__=='__main__': main()

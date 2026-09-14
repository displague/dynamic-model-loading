"""Supervised short screen of progressive precision and persistent increments."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .fault_screen import ROOT,supervise,write

CONFIG=ROOT/'configs/refinement-screen.json'
FROZEN=dict(protocol='docs/refinement-screen-protocol.md',parent='runs/fault-pager-20260914-v1',
    tokens_sha256='0ef8ea0ab583dd29f7645972623b75666c08cbf21b916e4957d186ce28727270',
    corpus_sha256='5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31',
    quant_group=128,cache_slots=8,threshold=.02,extra_cuda_mib=1024,prefix_tokens=4,
    generation_tokens=4,conditions=['q4','q6','q8','adaptive','retained','static'],worker_timeout_seconds=300)


def validate_config(cfg):
    if json.dumps(cfg,sort_keys=True)!=json.dumps(FROZEN,sort_keys=True):
        raise ValueError('Changed frozen refinement screen')


def worker(output):
    import shutil
    import torch
    from torch.nn import functional as F
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from .adapters import extract_ffns
    from .experiment import digest,environment,cuda_memory,load_corpus
    from .fault_generation import generate
    from .fault_pager_run import Ledger,move_non_ffn,unique_tensor_bytes
    from .fault_pager import metadata_bytes
    from .fault_resources import Resources
    from .ffn import dimensions
    from .metrics import relative_l2
    from .provenance import committed_inputs,frozen_environment
    from .progressive_precision import ProgressiveDraft
    from .debt_screen import charged_generate

    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance,protocol=committed_inputs(__file__,CONFIG,cfg['protocol'])
    if provenance['source_commit']!=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip():
        raise ValueError('Push reviewed source before inference')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(CONFIG,output/'config.json')
    shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent=ROOT/cfg['parent']
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        if digest(parent/name)!=cfg[key]:
            raise ValueError('Parent artifact differs')
        shutil.copyfile(parent/name,output/name)
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    env=environment(torch.device('cuda:0'))
    frozen_environment(env)
    snapshot=Path(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
        revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=True))
    prior=ROOT/'runs/qwen15b-packing-pilot-20260911/manifest.json'
    if digest(prior)!='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a':
        raise ValueError('Checkpoint manifest differs')
    checkpoint=json.loads(prior.read_text(encoding='utf-8'))['checkpoint']['files']
    for name,expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size,sha256=digest(snapshot/name))!=expected:
            raise ValueError('Checkpoint differs: '+name)
    shutil.copyfile(prior,output/'parent-manifest.json')
    write(output/'manifest.json',dict(**provenance,environment=env,checkpoint=checkpoint))
    corpus=load_corpus(output/'corpus.jsonl')
    ds=[r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    cal=next(r['id'] for r in corpus if r['split']=='calibration')
    tokens=json.loads((output/'token-ids.json').read_text(encoding='utf-8'))
    def load_model():
        return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).eval()
    raw=Ledger(output/'episodes.jsonl')
    try:
        with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
            print('Load reference and CPU draft',flush=True)
            target=load_model().to('cuda')
            draft=load_model()
            move_non_ffn(draft,'cuda')
            target_ffns,draft_ffns=extract_ffns(target),extract_ffns(draft)
            eos=tuple(target.generation_config.eos_token_id)
            if eos!=(151645,151643) or [dimensions(m) for m in draft_ffns]!=[(1536,8960)]*28:
                raise ValueError('Unexpected architecture')
            tp,dp=list(target.parameters()),list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp}&{p.untyped_storage().data_ptr() for p in dp}:
                raise ValueError('Unexpected shared weights')
            known=unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            baseline=cuda_memory(torch.device('cuda'))
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            pager=ProgressiveDraft(draft_ffns,group=cfg['quant_group'],slots=cfg['cache_slots'],
                                   threshold=cfg['threshold'],check=resources.boundary)
            torch.cuda.synchronize()
            construction_seconds=time.perf_counter()-start
            print('Snapshot embedded representation',flush=True)
            start=time.perf_counter()
            constructed=output/'constructed'
            constructed.mkdir()
            for i,layer in enumerate(pager.layers):
                data={f'{j}.{name}':getattr(q,name).cpu() for j,q in enumerate(layer) for name in ('base','minimum','scale')}
                data.update({f'increment.{s}':v for s,v in enumerate(pager.host[i])})
                save_file(data,constructed/f'layer-{i:02}.safetensors')
                del data
                resources.boundary()
            hashes={p.name:digest(p) for p in constructed.iterdir()}
            account=dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),shared_bytes=0,
                charged_baseline_bytes=known,baseline_cuda=baseline,resident_bytes=pager.resident_bytes,
                workspace_bytes=pager.workspace_bytes,cache_pool_bytes=pager.cache.pool.numel(),
                staging_bytes=pager.cache.staging.numel(),host_increments_bytes=sum(p.numel() for pair in pager.host for p in pair),
                construction_h2d_bytes=pager.construction_h2d_bytes,construction_d2h_bytes=pager.construction_d2h_bytes,
                construction_wall_seconds=construction_seconds,snapshot_wall_seconds=time.perf_counter()-start,
                snapshot_d2h_bytes=pager.resident_bytes,construction_cuda=cuda_memory(torch.device('cuda')),
                constructed_sha256=hashes)
            write(output/'allocation.json',account)
            if account['construction_cuda']['peak_allocated_bytes']-known>cfg['extra_cuda_mib']*2**20:
                raise RuntimeError('Construction extra CUDA exceeds allowance')
            # Freeze all constructed artifacts before diagnostics or scored references.
            ids={d:torch.tensor([tokens[d]],device='cuda') for d in [cal,*ds]}
            mechanics={}
            handles=[m.register_forward_pre_hook(lambda m,a,i=i:mechanics.update({f'input.{i}':a[0].detach().cpu()}))
                     for i,m in enumerate(target_ffns)]
            try:
                target(ids[ds[0]][:,:1],use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            print('Check independent eight-bit reconstruction and local errors',flush=True)
            for i,(dense_m,draft_m) in enumerate(zip(target_ffns,draft_ffns,strict=True)):
                x=mechanics[f'input.{i}']
                mechanics[f'fp32.{i}']=dense_m(x.to('cuda')).cpu()
                quantized=[]
                for w in (draft_m.gate_proj.weight,draft_m.up_proj.weight,draft_m.down_proj.weight.T):
                    groups=w.detach().reshape(8960,-1,128)
                    lo,hi=groups.amin(-1,keepdim=True),groups.amax(-1,keepdim=True)
                    scale=(hi-lo)/255
                    q=((groups-lo)/torch.where(scale>0,scale,torch.ones_like(scale))).round().clamp(0,255)*scale+lo
                    quantized.append(q.reshape_as(w))
                flat=x.reshape(1,1536)
                mechanics[f'independent8.{i}']=F.linear(draft_m.act_fn(F.linear(flat,quantized[0]))*
                    F.linear(flat,quantized[1]),quantized[2].T).reshape_as(x)
                del quantized,groups,q,w,lo,hi,scale
                for mode in ('q4','q6','q8'):
                    pager.mode=mode
                    pager.begin_token()
                    mechanics[f'{mode}.{i}']=draft_m(x.to('cuda')).cpu()
                resources.boundary()
            errs=[relative_l2(mechanics[f'independent8.{i}'],mechanics[f'q8.{i}']) for i in range(28)]
            save_file({k:v.contiguous() for k,v in mechanics.items()},output/'mechanics.safetensors')
            write(output/'mechanics.json',dict(relative_l2=errs,passed=max(errs)<=.01,
                h2d_bytes=pager.cache.stats['h2d_bytes'],cuda=cuda_memory(torch.device('cuda'))))
            del mechanics
            if max(errs)>.01 or torch.cuda.max_memory_allocated()-known>cfg['extra_cuda_mib']*2**20:
                raise RuntimeError('Independent reconstruction or mechanics resource gate failed')
            refs={}
            for d in [cal,*ds]:
                warm=d==cal
                torch.cuda.reset_peak_memory_stats()
                result=charged_generate(lambda:generate(target,ids[d][:,:2 if warm else 4],eos,
                    cap=1 if warm else 4,check=resources.check))
                refs[d]=result
                cuda=cuda_memory(torch.device('cuda'))
                raw.record(dict(mode='target',document=d,warmup=warm,**result,cuda=cuda,
                                extra_cuda_peak_bytes=cuda['peak_allocated_bytes']-known))
                raw.flush()
                if cuda['peak_allocated_bytes']-known>cfg['extra_cuda_mib']*2**20:
                    raise RuntimeError('Target-reference extra CUDA exceeds allowance')
            pairs=[(cal,m) for m in cfg['conditions']]
            pairs +=[(d,m) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
            for number,(d,mode) in enumerate(pairs):
                warm=d==cal
                folder=output/f'episode-{number:02}'
                folder.mkdir()
                pages=Ledger(folder/'pages.jsonl.gz',True)
                policy=Ledger(folder/'policy.jsonl.gz',True)
                rounds=Ledger(folder/'rounds.jsonl')
                pager.mode=mode
                pager.cache.sink,pager.policy_sink=pages.record,policy.record
                resources.boundary()
                torch.cuda.reset_peak_memory_stats()
                print(f'Start {number}: {mode}, warmup={warm}',flush=True)
                try:
                    def cleanup():
                        pages.flush()
                        policy.flush()
                        rounds.flush()
                    def execute():
                        pager.reset()
                        return generate(target,ids[d][:,:2 if warm else 4],eos,
                            cap=1 if warm else 4,draft=draft,pager=pager,record=rounds.record,check=resources.check)
                    result=charged_generate(execute,cleanup)
                    resources.boundary()
                    cuda=cuda_memory(torch.device('cuda'))
                    match=result['ids']==refs[d]['ids'] and result['stop_reason']==refs[d]['stop_reason']
                    row=dict(episode=folder.name,mode=mode,document=d,warmup=warm,reference_match=match,
                        **result,cache=dict(pager.cache.stats),consumed_draft_tokens=pager.cache.token,cuda=cuda,
                        extra_cuda_peak_bytes=cuda['peak_allocated_bytes']-known,
                        host_controller_metadata_bytes=metadata_bytes(pager.cache.entries,pager.cache.scores,pager.cache.admitted))
                    raw.record(row)
                    raw.flush()
                    write(folder/'episode.json',row)
                    print(f'End {number}: {result["accepted"]}/{result["attempted"]}, {result["wall_seconds"]:.3f}s',flush=True)
                    if not match or row['extra_cuda_peak_bytes']>cfg['extra_cuda_mib']*2**20:
                        raise RuntimeError('Reference or resource gate failed')
                finally:
                    pages.close()
                    policy.close()
                    rounds.close()
        write(output/'completion.json',dict(resources=resources.receipt()))
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        raw.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    if args.worker:
        worker(args.output.resolve())
    else:
        from .refinement_analysis import analyze
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.refinement_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('timeout','error'):
            raise SystemExit(1)


if __name__=='__main__':
    main()

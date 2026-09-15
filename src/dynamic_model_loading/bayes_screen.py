"""Bounded physical screen of calibrated Bayesian error acquisition and prefill."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .fault_screen import ROOT,supervise,write
from .refinement_screen import FROZEN as PRIOR

CONFIG=ROOT/'configs/bayes-screen.json'
FROZEN=dict(PRIOR,protocol='docs/bayes-screen-protocol.md',
    representation_parent='runs/refinement-screen-20260914-v1/worker',
    representation_files_sha256='f693d26dd4ca7bdcc4a66d1cfce8af4f3e165f48e2b1da8c806e91d478421dde',
    threshold=.06,prefix_tokens=8,fit_documents=8,calibration_documents=9,shadow_tokens=4,
    conditions=['q6','p8d6','p8d8','p8mean','p8gp','p8constant'])


def validate_config(cfg):
    if json.dumps(cfg,sort_keys=True)!=json.dumps(FROZEN,sort_keys=True):
        raise ValueError('Changed frozen Bayesian screen')


def worker(output):
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file,save_file
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from .adapters import extract_ffns
    from .experiment import digest,environment,cuda_memory,load_corpus
    from .fault_generation import generate,step,kv_bytes
    from .fault_pager_run import Ledger,move_non_ffn,unique_tensor_bytes
    from .fault_pager import metadata_bytes
    from .fault_resources import Resources
    from .ffn import dimensions
    from .metrics import relative_l2
    from .provenance import committed_inputs,frozen_environment
    from .calibrated_error import CalibratedDraft,ErrorGP
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
    prior_rep=ROOT/cfg['representation_parent']
    if digest(prior_rep/'files.json')!=cfg['representation_files_sha256']:
        raise ValueError('Representation parent inventory differs')
    prior_files=json.loads((prior_rep/'files.json').read_text(encoding='utf-8'))
    shutil.copyfile(prior_rep/'files.json',output/'representation-files.json')
    for name in ['mechanics.safetensors',*[f'constructed/layer-{i:02}.safetensors' for i in range(28)]]:
        if digest(prior_rep/name)!=prior_files[name]:
            raise ValueError('Representation parent differs: '+name)
    shutil.copyfile(prior_rep/'mechanics.safetensors',output/'parent-mechanics.safetensors')
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
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
    cal=[r['id'] for r in corpus if r['split']=='calibration'][:17]
    tokens=json.loads((output/'token-ids.json').read_text(encoding='utf-8'))
    def load_model():
        return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).eval()
    raw=Ledger(output/'episodes.jsonl')
    phases=Ledger(output/'phases.jsonl')
    try:
        with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
            print('Load reference and CPU draft',flush=True)
            target=load_model().to('cuda')
            draft=load_model()
            move_non_ffn(draft,'cuda')
            ffns=extract_ffns(draft)
            eos=tuple(target.generation_config.eos_token_id)
            if eos!=(151645,151643) or [dimensions(m) for m in ffns]!=[(1536,8960)]*28:
                raise ValueError('Unexpected architecture')
            tp,dp=list(target.parameters()),list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp}&{p.untyped_storage().data_ptr() for p in dp}:
                raise ValueError('Unexpected shared weights')
            known=unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            baseline=cuda_memory(torch.device('cuda'))
            if baseline['peak_allocated_bytes']-known>1024*2**20:
                raise RuntimeError('Model-loading CUDA allowance')
            def phase(name,start,**extra):
                resources.boundary()
                cuda=cuda_memory(torch.device('cuda'))
                row=dict(phase=name,wall_seconds=time.perf_counter()-start,cuda=cuda,
                         extra_cuda_peak_bytes=cuda['peak_allocated_bytes']-known,**extra)
                phases.record(row)
                phases.flush()
                if row['extra_cuda_peak_bytes']>1024*2**20:
                    raise RuntimeError('Extra CUDA allowance exceeded: '+name)
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            pager=CalibratedDraft(ffns,group=128,slots=8,threshold=.06,check=resources.boundary)
            torch.cuda.synchronize()
            phase('construction',start)
            start=time.perf_counter()
            constructed=output/'constructed'
            constructed.mkdir()
            for i,layer in enumerate(pager.layers):
                data={f'{j}.{name}':getattr(q,name).cpu() for j,q in enumerate(layer) for name in ('base','minimum','scale')}
                data.update({f'increment.{s}':v for s,v in enumerate(pager.host[i])})
                prior_data=load_file(prior_rep/f'constructed/layer-{i:02}.safetensors')
                if set(data)!=set(prior_data) or any(not torch.equal(v,prior_data[k]) for k,v in data.items()):
                    raise ValueError('Changed embedded representation')
                save_file(data,constructed/f'layer-{i:02}.safetensors')
                del data,prior_data
                resources.boundary()
            phase('snapshot',start,d2h_bytes=pager.resident_bytes)
            account=dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),shared_bytes=0,
                charged_baseline_bytes=known,baseline_cuda=baseline,resident_bytes=pager.resident_bytes,
                workspace_bytes=pager.workspace_bytes,cache_pool_bytes=pager.cache.pool.numel(),
                staging_bytes=pager.cache.staging.numel(),host_increments_bytes=sum(p.numel() for pair in pager.host for p in pair),
                construction_h2d_bytes=pager.construction_h2d_bytes+pager.projection.numel()*4,
                projection_cuda_bytes=pager.projection.numel()*4,
                constructed_sha256={p.name:digest(p) for p in constructed.iterdir()})
            write(output/'allocation.json',account)
            start=time.perf_counter()
            mechanics=load_file(output/'parent-mechanics.safetensors')
            checks={}
            pager.mode='p8d8'
            for i,m in enumerate(ffns):
                pager.begin_token()
                checks[f'q8.{i}']=m(mechanics[f'input.{i}'].to('cuda')).cpu()
            errs=[relative_l2(mechanics[f'q8.{i}'],checks[f'q8.{i}']) for i in range(28)]
            save_file(checks,output/'mechanics.safetensors')
            if max(errs)>.01:
                raise RuntimeError('Changed eight-bit FFN arithmetic')
            phase('mechanics',start,h2d_bytes=pager.cache.stats['h2d_bytes'],relative_l2=errs)
            del checks,mechanics
            ids={d:torch.tensor([tokens[d]],device='cuda') for d in [*cal,*ds]}
            observations={}
            def shadow(d,split,index):
                folder=output/f'shadow-{index:02}'
                folder.mkdir()
                pages,policy,kv=(Ledger(folder/name,True) for name in ('pages.jsonl.gz','policy.jsonl.gz','kv.jsonl.gz'))
                rows=[]
                def policy_row(row):
                    policy.record(row)
                    if row['phase']=='decode':
                        rows.append(row)
                pager.mode='shadow'
                pager.prefill_tokens=8
                pager.cache.sink,pager.policy_sink,pager.kv_sink=pages.record,policy_row,kv.record
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                try:
                    pager.reset()
                    cache=DynamicCache(config=draft.config)
                    for t in ids[d][:,:12].split(1,1):
                        step(draft,t,cache,pager)
                    torch.cuda.synchronize()
                    pages.flush(); policy.flush(); kv.flush()
                    phase('shadow',start,document=d,split=split,folder=folder.name,
                          cache=dict(pager.cache.stats),consumed_draft_tokens=pager.cache.token,
                          kv_bytes=kv_bytes(cache),controller_seconds=pager.controller_seconds)
                    observations[d]=rows
                    print(f'Shadow {index}: {split}',flush=True)
                finally:
                    pages.close(); policy.close(); kv.close()
            for i,d in enumerate(cal):
                shadow(d,'fit' if i<8 else 'calibration',i)
            start=time.perf_counter()
            def matrices(documents,layer):
                rows=[r for d in documents for r in observations[d] if r['layer']==layer]
                return torch.tensor([r['feature'] for r in rows],dtype=torch.float64),torch.tensor([r['observed_log'] for r in rows],dtype=torch.float64)
            pager.models=[ErrorGP(*matrices(cal[:8],i)) for i in range(28)]
            for i,model in enumerate(pager.models):
                model.calibrate([matrices([d],i) for d in cal[8:]])
            fitted={f'{i}.{k}':v for i,m in enumerate(pager.models) for k,v in m.tensors().items()}
            fitted['projection']=pager.projection.cpu()
            save_file(fitted,output/'gp.safetensors')
            write(output/'fit.json',dict(fit_documents=cal[:8],calibration_documents=cal[8:],
                gp_sha256=digest(output/'gp.safetensors'),host_model_tensor_bytes=sum(v.numel()*v.element_size() for k,v in fitted.items() if k!='projection'),
                host_model_metadata_bytes=metadata_bytes(*[vars(m) for m in pager.models]),
                observation_metadata_bytes=metadata_bytes(observations),projection_d2h_bytes=pager.projection.numel()*4))
            phase('fit',start)
            for i,d in enumerate(ds,17):
                shadow(d,'diagnostic',i)
            # Drop raw observation lists before scored generation; frozen CPU GP remains.
            observations.clear()
            pager.cache.sink=pager.policy_sink=pager.kv_sink=lambda r:None
            refs={}
            for d in [cal[0],*ds]:
                warm=d==cal[0]
                torch.cuda.reset_peak_memory_stats()
                result=charged_generate(lambda:generate(target,ids[d][:,:2 if warm else 8],eos,
                    cap=1 if warm else 4,check=resources.check))
                refs[d]=result
                cuda=cuda_memory(torch.device('cuda'))
                raw.record(dict(mode='target',document=d,warmup=warm,**result,cuda=cuda,
                                extra_cuda_peak_bytes=cuda['peak_allocated_bytes']-known))
                raw.flush()
                if cuda['peak_allocated_bytes']-known>1024*2**20:
                    raise RuntimeError('Target reference allowance')
            pairs=[(cal[0],m) for m in cfg['conditions']]
            pairs +=[(d,m) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
            for number,(d,mode) in enumerate(pairs):
                warm=d==cal[0]
                folder=output/f'episode-{number:02}'
                folder.mkdir()
                pages,policy,kv=(Ledger(folder/name,True) for name in ('pages.jsonl.gz','policy.jsonl.gz','kv.jsonl.gz'))
                rounds=Ledger(folder/'rounds.jsonl')
                pager.mode=mode
                pager.prefill_tokens=2 if warm else 8
                pager.cache.sink,pager.policy_sink,pager.kv_sink=pages.record,policy.record,kv.record
                resources.boundary()
                torch.cuda.reset_peak_memory_stats()
                print(f'Start {number}: {mode}, warmup={warm}',flush=True)
                try:
                    def cleanup():
                        pages.flush(); policy.flush(); kv.flush(); rounds.flush()
                    def execute():
                        pager.reset()
                        return generate(target,ids[d][:,:2 if warm else 8],eos,
                            cap=1 if warm else 4,draft=draft,pager=pager,record=rounds.record,check=resources.check)
                    result=charged_generate(execute,cleanup)
                    resources.boundary()
                    cuda=cuda_memory(torch.device('cuda'))
                    match=result['ids']==refs[d]['ids'] and result['stop_reason']==refs[d]['stop_reason']
                    row=dict(episode=folder.name,mode=mode,document=d,warmup=warm,reference_match=match,
                        **result,cache=dict(pager.cache.stats),consumed_draft_tokens=pager.cache.token,cuda=cuda,
                        extra_cuda_peak_bytes=cuda['peak_allocated_bytes']-known,controller_seconds=pager.controller_seconds,
                        host_controller_metadata_bytes=metadata_bytes(pager.cache.entries,pager.cache.scores,pager.cache.admitted))
                    raw.record(row); raw.flush()
                    write(folder/'episode.json',row)
                    print(f'End {number}: {result["accepted"]}/{result["attempted"]}, {result["wall_seconds"]:.3f}s',flush=True)
                    if not match or row['extra_cuda_peak_bytes']>1024*2**20:
                        raise RuntimeError('Reference or resource gate failed')
                finally:
                    pages.close(); policy.close(); kv.close(); rounds.close()
            if digest(output/'gp.safetensors')!=json.loads((output/'fit.json').read_text(encoding='utf-8'))['gp_sha256']:
                raise RuntimeError('Fitted artifact changed')
            fitted=load_file(output/'gp.safetensors')
            for i,m in enumerate(pager.models):
                if any(not torch.equal(v,fitted[f'{i}.{k}']) for k,v in m.tensors().items()):
                    raise RuntimeError('Fitted parameters mutated')
        write(output/'completion.json',dict(resources=resources.receipt()))
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        raw.close(); phases.close()
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
        from .bayes_analysis import analyze
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.bayes_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('timeout','error'):
            raise SystemExit(1)


if __name__=='__main__':
    main()

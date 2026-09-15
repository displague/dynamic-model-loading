"""Bounded CUDA screen of risk-selected pages on generated draft trajectories."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
from .fault_screen import ROOT,supervise,write
from .output_sensors import FROZEN as SENSOR

CONFIG=ROOT/'configs/risk-screen.json'
FROZEN={k:v for k,v in SENSOR.items() if k not in ('fit_documents','diagnostic_documents','positions','oracle_pages')}
FROZEN.update(protocol='docs/risk-screen-protocol.md',diagnostic_indices=[2,3],generation_tokens=8,
    conditions=[['fixed','risk','all35'],['all35','risk','fixed']],
    index_parent='runs/acquisition-screen-20260915-v1/worker/predictions.safetensors',
    index_parent_sha256='6bf1319d2d5bc79fd5e9da1af187a3355bfee903a8453f1825825ef9acf3e877',
    page_budget=17,extra_cuda_mib=1024)

def validate_config(cfg):
    if cfg!=FROZEN: raise ValueError('Physical risk-screen freeze changed')

def bundle(output,*,config_path=CONFIG,validator=validate_config,source_file=__file__):
    import torch
    from huggingface_hub import snapshot_download
    from .experiment import digest,environment,load_corpus
    from .provenance import committed_inputs,frozen_environment
    cfg=json.loads(Path(config_path).read_text()); validator(cfg)
    provenance,protocol=committed_inputs(source_file,config_path,cfg['protocol'])
    if 'prompts' in cfg:
        from .numpy_backend import set_blas_threads
        provenance['numpy_blas']=set_blas_threads(4)
    if provenance['source_commit']!=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip():
        raise ValueError('Push source before checkpoint inference')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config_path,output/'config.json'); shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent,rep=ROOT/cfg['parent'],ROOT/cfg['representation_parent']
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        if digest(parent/name)!=cfg[key]: raise ValueError('Corpus binding changed')
        shutil.copyfile(parent/name,output/name)
    if digest(rep/'files.json')!=cfg['representation_files_sha256']: raise ValueError('Representation inventory changed')
    inventory=json.loads((rep/'files.json').read_text())
    for name in ['mechanics.safetensors',*[f'constructed/layer-{i:02}.safetensors' for i in range(28)]]:
        if digest(rep/name)!=inventory[name]: raise ValueError('Representation changed')
    shutil.copyfile(rep/'files.json',output/'representation-files.json')
    shutil.copyfile(rep/'mechanics.safetensors',output/'parent-mechanics.safetensors')
    index=ROOT/cfg['index_parent']
    if digest(index)!=cfg['index_parent_sha256']: raise ValueError('Frozen predictor changed')
    shutil.copyfile(index,output/'index-parent.safetensors')
    torch.set_num_threads(4); torch.manual_seed(20260915)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    env=environment(torch.device('cuda:0')); frozen_environment(env)
    snapshot=Path(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=True))
    parent_manifest=ROOT/'runs/qwen15b-packing-pilot-20260911/manifest.json'
    if digest(parent_manifest)!='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a':
        raise ValueError('Checkpoint manifest changed')
    checkpoint=json.loads(parent_manifest.read_text())['checkpoint']['files']
    for name,expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size,sha256=digest(snapshot/name))!=expected:
            raise ValueError('Checkpoint changed')
    shutil.copyfile(parent_manifest,output/'parent-manifest.json')
    write(output/'manifest.json',dict(**provenance,environment=env,checkpoint=checkpoint))
    if 'prompts' in cfg:
        from transformers import AutoTokenizer
        if (snapshot/'LICENSE').stat().st_size!=cfg['license_bytes'] or digest(snapshot/'LICENSE')!=cfg['license_sha256']:
            raise ValueError('Pinned Qwen license changed')
        shutil.copyfile(snapshot/'LICENSE',output/'LICENSE-Qwen.txt')
        write(output/'NOTICE.json',dict(model='Qwen/Qwen2.5-1.5B-Instruct',
            revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
            changes='Experimental quantized representations, acquired corrections and observations; full HF weights external.',
            corpus='Parent corpus archived only for provenance; inference uses the new authored config prompts.'))
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        docs=[f'authored-{i}' for i in range(len(cfg['prompts']))]
        tokens={d:tokenizer.encode(p,add_special_tokens=False) for d,p in zip(docs,cfg['prompts'],strict=True)}
        if any(len(t)<4 for t in tokens.values()): raise ValueError('Short authored prompt')
        write(output/'authored-token-ids.json',tokens)
        return snapshot,rep,docs,tokens
    docs=[r['id'] for r in load_corpus(output/'corpus.jsonl') if r['split']=='diagnostic']
    return snapshot,rep,[docs[i] for i in cfg['diagnostic_indices']],json.loads((output/'token-ids.json').read_text())

def worker(output,*,bundle_fn=bundle,conditions=None,runtime_class=None):
    import torch
    import numpy as np
    from safetensors.torch import load_file,save_file
    from safetensors.numpy import load_file as load_numpy
    from transformers import AutoModelForCausalLM
    from .adapters import extract_ffns
    from .experiment import digest,cuda_memory
    from .fault_generation import generate
    from .fault_pager import metadata_bytes
    from .fault_pager_run import Ledger,move_non_ffn,unique_tensor_bytes
    from .fault_resources import Resources
    from .metrics import relative_l2
    from .output_pages import OutputPageDraft
    from .risk_runtime import RiskReadout
    snapshot,rep,docs,tokens=bundle_fn(output)
    pages=Ledger(output/'pages.jsonl.gz',True); phases=Ledger(output/'phases.jsonl')
    context={}; runtime=None
    try:
        with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
            started=time.perf_counter()
            def load(): return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
            target,draft=load().to('cuda'),load(); move_non_ffn(draft,'cuda')
            tp,dp=list(target.parameters()),list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp}&{p.untyped_storage().data_ptr() for p in dp}: raise ValueError('Shared model storage')
            known=unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            def phase(name,start,**extra):
                torch.cuda.synchronize(); resources.boundary(); mem=cuda_memory(torch.device('cuda'))
                end=time.perf_counter()
                row=dict(phase=name,started_monotonic=start,finished_monotonic=end,wall_seconds=end-start,cuda=mem,extra_cuda_peak_bytes=mem['peak_allocated_bytes']-known,**extra)
                phases.record(row); phases.flush()
                if row['extra_cuda_peak_bytes']>1024*2**20: raise RuntimeError('Extra CUDA budget exceeded')
            phase('model_loading',started)
            started=time.perf_counter()
            pager=OutputPageDraft(extract_ffns(draft),group=128,slots=1,page_width=256,check=resources.boundary)
            pager.cache.sink=lambda r:pages.record(dict(**context,cache='layer',**r))
            pager.page_cache.sink=lambda r:pages.record(dict(**context,cache='page',**r))
            if draft.lm_head.bias is not None or draft.model.norm.variance_epsilon!=1e-6: raise ValueError('Readout assumptions changed')
            equality=[]
            for i,layer in enumerate(pager.layers):
                old=load_file(rep/f'constructed/layer-{i:02}.safetensors')
                new={f'{j}.{name}':getattr(q,name).cpu() for j,q in enumerate(layer) for name in ('base','minimum','scale')}
                new.update({f'increment.{s}':v for s,v in enumerate(pager.host[i])})
                equality.append(set(old)==set(new) and all(torch.equal(v,old[k]) for k,v in new.items()))
                del old,new
            if not all(equality): raise ValueError('Parent representation mismatch')
            planes={f'low.{j}':v.cpu() for j,v in enumerate(pager.low)}
            planes.update({f'next.{j}':pager.page_host[0][j] for j in range(3)})
            planes.update({f'parent4.{j}':q.base for j,q in enumerate(pager.layers[-1])})
            save_file(planes,output/'final-planes.safetensors'); del planes
            index_all=load_numpy(output/'index-parent.safetensors')
            index={k:index_all[k].copy() for k in ('vector_prior','vector_covariance','feature_variance')}; del index_all
            phase('construction',started,equal_layers=equality,construction_h2d_bytes=pager.construction_h2d_bytes,
                construction_d2h_bytes=pager.construction_d2h_bytes,equality_d2h_bytes=pager.resident_bytes-3*8960*384,
                plane_snapshot_d2h_bytes=3*8960*384)
            write(output/'allocation.json',dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),charged_baseline_bytes=known,shared_bytes=0,
                resident_bytes=pager.resident_bytes,workspace_bytes=pager.workspace_bytes,
                layer_pool_bytes=pager.cache.pool.numel(),page_pool_bytes=pager.page_cache.pool.numel(),
                layer_staging_bytes=pager.cache.staging.numel(),page_staging_bytes=pager.page_cache.staging.numel(),
                original_host_increment_bytes=sum(v.numel() for pair in pager.host for v in pair),
                final_host_base4_bytes=sum(q.base.numel() for q in pager.layers[-1]),new_host_plane_bytes=pager.page_host[0].numel(),
                new_device_base2_bytes=sum(v.numel() for v in pager.low),index_host_array_bytes=sum(a.nbytes for a in index.values()),
                host_metadata_bytes=metadata_bytes(vars(pager),vars(pager.cache),vars(pager.page_cache),*[vars(q) for layer in pager.layers for q in layer])))
            started=time.perf_counter(); mech=load_file(output/'parent-mechanics.safetensors')
            pager.prefill=True; context.update(episode='mechanics',call=None,prefill=True)
            checks={}
            for i,m in enumerate(extract_ffns(draft)):
                pager.begin_token(); checks[f'q8.{i}']=m(mech[f'input.{i}'].to('cuda')).cpu()
            errors=[relative_l2(mech[f'q8.{i}'],checks[f'q8.{i}']) for i in range(28)]
            save_file(checks,output/'mechanics.safetensors')
            if max(errors)>.01: raise RuntimeError('Eight-bit faithfulness failed')
            phase('mechanics',started,relative_l2=errors,input_h2d_bytes=28*1536*4,output_d2h_bytes=28*1536*4,correction_scalar_d2h_bytes=27*8)
            del mech,checks
            for di,doc in enumerate(docs):
                prefix=torch.tensor([tokens[doc][:4]],device='cuda')
                started=time.perf_counter(); reference=generate(target,prefix,[151643,151645],cap=8,check=resources.check,copy_receipt=True)
                write(output/f'reference-{di}.json',dict(document=doc,**reference))
                phase('target_reference',started,document=doc,prefix_h2d_bytes=32)
                for condition in (conditions or FROZEN['conditions'])[di]:
                    name=f'episode-{di}-{condition}'; folder=output/name; folder.mkdir()
                    calls=Ledger(folder/'calls.jsonl.gz',True); rounds=Ledger(folder/'rounds.jsonl')
                    context.update(episode=name,call=None,prefill=None)
                    runtime=(runtime_class or RiskReadout)(draft,pager,index,lambda r:(calls.record(r),calls.flush()),context)
                    runtime.reset(condition)
                    started=time.perf_counter()
                    try:
                        result=generate(target,prefix,[151643,151645],cap=8,draft=draft,pager=runtime,record=rounds.record,check=resources.check,copy_receipt=True)
                        pages.flush(); calls.flush(); rounds.flush()
                        ended=time.perf_counter()
                        result.update(charged_started_monotonic=started,charged_finished_monotonic=ended,charged_wall_seconds=ended-started)
                        result.update(document=doc,condition=condition,layer_cache=dict(pager.cache.stats),page_cache=dict(pager.page_cache.stats),
                            draft_calls=runtime.calls,output_matches_reference=result['ids']==reference['ids'])
                        write(folder/'episode.json',result)
                        if result['ids']!=reference['ids']: raise RuntimeError('Committed output differs from scalar target')
                        phase('episode',started,episode=name)
                    finally:
                        runtime.close(); runtime=None; calls.close(); rounds.close()
                started=time.perf_counter(); repeated=generate(target,prefix,[151643,151645],cap=8,check=resources.check,copy_receipt=True)
                write(output/f'reference-repeat-{di}.json',dict(document=doc,**repeated))
                if repeated['ids']!=reference['ids']: raise RuntimeError('Scalar target reference changed')
                phase('target_repeat',started,document=doc)
            write(output/'completion.json',dict(complete=True,cuda_execution=True))
        write(output/'final-resources.json',resources.receipt())
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc))); raise
    finally:
        if runtime is not None: runtime.close()
        pages.close(); phases.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze: parser.error('Conflicting modes')
    from .risk_analysis import analyze
    if args.worker: worker(args.output)
    else:
        result=analyze(args.output) if args.analyze else supervise(args.output,module='dynamic_model_loading.risk_screen',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'): raise SystemExit(1)

if __name__=='__main__': main()

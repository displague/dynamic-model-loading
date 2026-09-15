"""Bounded checkpoint screen for physical final-FFN output-margin observations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from .decision_field import FROZEN as PARENT
from .fault_screen import ROOT,supervise,write

CONFIG = ROOT/'configs/output-sensors.json'
FROZEN = {k:v for k,v in PARENT.items() if k!='sites'}
FROZEN.update(protocol='docs/output-sensors-protocol.md',positions=16,page_width=256,oracle_pages=17)
FROZEN['final_base_sha256']=['8a461dfcc154e9e92df2f4dcdfca1f70c779f7397d3f58fc45ddc43929e23578',
    '4513fcfbe9000824ae32980c85ee9d0b0c2c6fd0e7ee490aa4860024ca2069a0',
    '847e4b143e7bd49ff3d20f467ec75f06564e37059c5f7ff555dd4d42a05c3f4a']


def validate_config(cfg):
    if json.dumps(cfg,sort_keys=True)!=json.dumps(FROZEN,sort_keys=True):
        raise ValueError('Changed output-sensor screen')


def bundle(output):
    """Validate inputs and archive Git-bound source before any checkpoint execution."""
    import torch
    from huggingface_hub import snapshot_download
    from .experiment import digest,environment,load_corpus
    from .provenance import committed_inputs,frozen_environment
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance,protocol = committed_inputs(__file__,CONFIG,cfg['protocol'])
    if provenance['source_commit']!=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip():
        raise ValueError('Push reviewed source first')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(CONFIG,output/'config.json')
    shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent,rep = ROOT/cfg['parent'],ROOT/cfg['representation_parent']
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        if digest(parent/name)!=cfg[key]:
            raise ValueError('Parent corpus drift')
        shutil.copyfile(parent/name,output/name)
    if digest(rep/'files.json')!=cfg['representation_files_sha256']:
        raise ValueError('Representation inventory drift')
    inventory=json.loads((rep/'files.json').read_text(encoding='utf-8'))
    for name in ['mechanics.safetensors',*[f'constructed/layer-{i:02}.safetensors' for i in range(28)]]:
        if digest(rep/name)!=inventory[name]:
            raise ValueError('Parent representation drift')
    shutil.copyfile(rep/'files.json',output/'representation-files.json')
    shutil.copyfile(rep/'mechanics.safetensors',output/'parent-mechanics.safetensors')
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
        raise ValueError('Checkpoint manifest drift')
    checkpoint=json.loads(prior.read_text(encoding='utf-8'))['checkpoint']['files']
    for name,expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size,sha256=digest(snapshot/name))!=expected:
            raise ValueError('Checkpoint drift')
    shutil.copyfile(prior,output/'parent-manifest.json')
    write(output/'manifest.json',dict(**provenance,environment=env,checkpoint=checkpoint))
    corpus=load_corpus(output/'corpus.jsonl')
    docs=[(r['id'],'fit') for r in corpus if r['split']=='calibration'][:6]
    docs += [(r['id'],'diagnostic') for r in corpus if r['split']=='diagnostic'][:2]
    return snapshot,rep,docs,json.loads((output/'token-ids.json').read_text(encoding='utf-8'))


def worker(output):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file,save_file
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from .adapters import extract_ffns
    from .experiment import digest,cuda_memory
    from .fault_generation import step,kv_bytes
    from .fault_pager import metadata_bytes
    from .fault_pager_run import Ledger,move_non_ffn,unique_tensor_bytes
    from .fault_resources import Resources
    from .decision_field import fingerprint,kv_storage_bytes
    from .metrics import relative_l2
    from .output_pages import OutputPageDraft,projected_margin

    snapshot,rep,docs,tokens=bundle(output)
    pages,raw,phases=Ledger(output/'pages.jsonl.gz',True),Ledger(output/'frames.jsonl'),Ledger(output/'phases.jsonl')
    context={}
    try:
        with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
            started=time.perf_counter()
            def load():
                return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,
                    attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
            target,draft=load().to('cuda'),load()
            move_non_ffn(draft,'cuda')
            tp,dp=list(target.parameters()),list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp}&{p.untyped_storage().data_ptr() for p in dp}:
                raise ValueError('Shared model parameters')
            known=unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            def phase(name,start,**extra):
                torch.cuda.synchronize(); resources.boundary()
                mem=cuda_memory(torch.device('cuda'))
                row=dict(phase=name,wall_seconds=time.perf_counter()-start,cuda=mem,
                    extra_cuda_peak_bytes=mem['peak_allocated_bytes']-known,**extra)
                phases.record(row); phases.flush()
                if row['extra_cuda_peak_bytes']>1024*2**20:
                    raise RuntimeError('Extra CUDA cap')
            phase('model_loading',started)
            started=time.perf_counter()
            pager=OutputPageDraft(extract_ffns(draft),group=128,slots=1,page_width=256,check=resources.boundary)
            pager.cache.sink=lambda r:pages.record(dict(**context,cache='layer',**r))
            pager.page_cache.sink=lambda r:pages.record(dict(**context,cache='page',**r))
            if draft.lm_head.bias is not None or draft.model.norm.variance_epsilon!=1e-6:
                raise ValueError('Readout assumptions changed')
            captured={}
            draft.model.norm.register_forward_pre_hook(lambda m,a:captured.update(h=a[0].detach().clone()))
            draft.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(
                lambda m,a:captured.update(residual=a[0].detach().clone()))
            equality=[]
            for i,layer in enumerate(pager.layers):
                old=load_file(rep/f'constructed/layer-{i:02}.safetensors')
                new={f'{j}.{name}':getattr(q,name).cpu() for j,q in enumerate(layer) for name in ('base','minimum','scale')}
                new.update({f'increment.{s}':v for s,v in enumerate(pager.host[i])})
                equality.append(set(old)==set(new) and all(torch.equal(v,old[k]) for k,v in new.items()))
                del old,new
            if not all(equality):
                raise ValueError('Parent integer representation drift')
            planes={f'low.{j}':v.cpu() for j,v in enumerate(pager.low)}
            planes.update({f'next.{j}':pager.page_host[0][j] for j in range(3)})
            planes.update({f'parent4.{j}':q.base for j,q in enumerate(pager.layers[-1])})
            integer_planes=[]
            for j in range(3):
                if hashlib.sha256(planes[f'parent4.{j}'].numpy().tobytes()).hexdigest()!=FROZEN['final_base_sha256'][j]:
                    raise ValueError('Final parent base content drift')
                def expand(value,bits):
                    return torch.stack([(value>>(bits*k))&(2**bits-1) for k in range(8//bits)],-1).reshape(value.shape[0],-1)
                integer_planes.append(torch.equal(4*expand(planes[f'low.{j}'],2)+expand(planes[f'next.{j}'],2),
                                                  expand(planes[f'parent4.{j}'],4)))
            if not all(integer_planes):
                raise ValueError('New integer-plane identity')
            save_file(planes,output/'final-planes.safetensors')
            del planes
            phase('construction',started,equal_layers=equality,integer_planes=integer_planes,
                construction_h2d_bytes=pager.construction_h2d_bytes,
                construction_d2h_bytes=pager.construction_d2h_bytes,
                equality_d2h_bytes=pager.resident_bytes-3*8960*384,
                plane_snapshot_d2h_bytes=3*8960*384)
            account=dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),
                charged_baseline_bytes=known,shared_bytes=0,resident_bytes=pager.resident_bytes,
                workspace_bytes=pager.workspace_bytes,layer_pool_bytes=pager.cache.pool.numel(),
                page_pool_bytes=pager.page_cache.pool.numel(),layer_staging_bytes=pager.cache.staging.numel(),
                page_staging_bytes=pager.page_cache.staging.numel(),
                original_host_increment_bytes=sum(v.numel() for pair in pager.host for v in pair),
                final_host_base4_bytes=sum(q.base.numel() for q in pager.layers[-1]),
                new_host_plane_bytes=pager.page_host[0].numel(),new_device_base2_bytes=sum(v.numel() for v in pager.low),
                host_metadata_bytes=metadata_bytes(vars(pager),vars(pager.cache),vars(pager.page_cache),
                    *[vars(q) for layer in pager.layers for q in layer]))
            write(output/'allocation.json',account)
            started=time.perf_counter()
            mech=load_file(output/'parent-mechanics.safetensors')
            pager.prefill=True
            context.update(phase='mechanics',document=None,position=None,action=None)
            checks={}
            for i,m in enumerate(extract_ffns(draft)):
                pager.begin_token()
                checks[f'q8.{i}']=m(mech[f'input.{i}'].to('cuda')).cpu()
            errors=[relative_l2(mech[f'q8.{i}'],checks[f'q8.{i}']) for i in range(28)]
            save_file(checks,output/'mechanics.safetensors')
            if max(errors)>.01:
                raise RuntimeError('Eight-bit numerical control')
            phase('mechanics',started,relative_l2=errors,input_h2d_bytes=28*1536*4,
                output_d2h_bytes=28*1536*4,correction_scalar_d2h_bytes=27*8)
            del mech,checks
            for di,(doc,split) in enumerate(docs):
                started=time.perf_counter()
                ids=torch.tensor([tokens[doc][:20]],device='cuda')
                cache,tcache=DynamicCache(config=draft.config),DynamicCache(config=target.config)
                pager.reset(); pager.prefill=True; pager.force_full=False
                context.update(phase='prefill',document=doc,position=None,action=None)
                for token in ids[:,:4].split(1,1):
                    step(draft,token,cache,pager); step(target,token,tcache)
                prefill_sha,prefill_readback=fingerprint(cache,4)
                phase('prefill',started,document=doc,token_h2d_bytes=20*8,
                    kv_storage_bytes=kv_storage_bytes(cache)+kv_storage_bytes(tcache),
                    kv_sha=prefill_sha,kv_fingerprint_d2h_bytes=prefill_readback,
                    correction_scalar_d2h_bytes=4*27*8)
                pager.prefill=False
                for pos in range(16):
                    started=time.perf_counter()
                    folder=output/f'frame-{di:02}-{pos:02}'; folder.mkdir()
                    length=cache.get_seq_length()
                    prior_sha,prior_size=fingerprint(cache,length)
                    context.update(phase='decode',document=doc,position=pos,action='base')
                    t0=time.perf_counter()
                    target_logits=step(target,ids[:,4+pos:5+pos],tcache)[0,-1].cpu()
                    target_seconds=time.perf_counter()-t0
                    t0=time.perf_counter()
                    base=step(draft,ids[:,4+pos:5+pos],cache,pager)[0,-1].cpu()
                    base_seconds=time.perf_counter()-t0
                    h=captured['h'].flatten().clone()
                    residual=captured['residual'].flatten().clone()
                    base_y=pager.base_y.flatten().clone()
                    post_sha,post_size=fingerprint(cache,length+1)
                    u=int(base.argmax())
                    other=base.clone(); other[u]=-torch.inf
                    v=int(other.argmax()); del other
                    headrows=torch.stack((draft.lm_head.weight[u],draft.lm_head.weight[v]))
                    gain=draft.model.norm.weight
                    basis=(headrows[0]-headrows[1])*gain
                    geometry=dict(h=h.cpu(),residual=residual.cpu(),gain=gain.cpu(),basis=basis.cpu(),headrows=headrows.cpu())
                    base_y_cpu=base_y.cpu()
                    eps=draft.model.norm.variance_epsilon
                    rms=torch.sqrt(h.square().mean()+eps)
                    observations=[]; deltas=[]
                    context['action']='observe'
                    for page in range(35):
                        t0=time.perf_counter()
                        delta=pager.acquire(page).flatten()
                        predicted=projected_margin(basis,h,delta,eps)
                        direct=F.linear(draft.model.norm.forward((h+delta)[None]),headrows)[0]
                        values=torch.stack((predicted,direct[0]-direct[1],(basis*delta).sum()/rms)).cpu().tolist()
                        deltas.append(delta.cpu())
                        pages.flush()
                        observations.append(dict(page=page,predicted=values[0],actual=values[1],utility=values[2],
                            wall_seconds=time.perf_counter()-t0,h2d_bytes=3*294912,d2h_bytes=1536*4+12))
                    delta_matrix=torch.stack(deltas)
                    t0=time.perf_counter()
                    canonical_y=pager._compute().flatten().clone()
                    canonical_y_cpu=canonical_y.cpu()
                    def readout(y):
                        return draft.lm_head(draft.model.norm.forward((residual+y)[None]))[0].cpu()
                    high=readout(canonical_y)
                    winner=int(high.argmax())
                    high_readout_seconds=time.perf_counter()-t0
                    t0=time.perf_counter()
                    oracle_basis=(draft.lm_head.weight[winner]-draft.lm_head.weight[u])*gain
                    oracle_basis_cpu=oracle_basis.cpu()
                    oracle_row_cpu=draft.lm_head.weight[winner].cpu()
                    # Observation field is paid and privileged, never hidden in a causal policy.
                    utilities=(delta_matrix@oracle_basis_cpu)/float(rms.cpu())
                    selected=sorted(range(35),key=lambda p:(-float(utilities[p]),p))[:17]
                    oracle_selection_seconds=time.perf_counter()-t0
                    t0=time.perf_counter()
                    context['action']='fixed'
                    fixed_y=pager.partial(range(17)).flatten()
                    fixed=readout(fixed_y)
                    fixed_y_cpu=fixed_y.cpu()
                    pages.flush()
                    fixed_seconds=time.perf_counter()-t0
                    t0=time.perf_counter()
                    context['action']='oracle'
                    oracle_y=pager.partial(selected).flatten()
                    oracle=readout(oracle_y)
                    oracle_y_cpu=oracle_y.cpu()
                    pages.flush()
                    oracle_seconds=time.perf_counter()-t0
                    saved=dict(base=base,high=high,fixed=fixed,oracle=oracle,target=target_logits,
                        **geometry,base_y=base_y_cpu,canonical_y=canonical_y_cpu,deltas=delta_matrix,
                        oracle_basis=oracle_basis_cpu,oracle_row=oracle_row_cpu,oracle_utilities=utilities,
                        fixed_y=fixed_y_cpu,oracle_y=oracle_y_cpu)
                    old_after,old_size=fingerprint(cache,length)
                    after,after_size=fingerprint(cache,length+1)
                    if old_after!=prior_sha or after!=post_sha:
                        raise RuntimeError('Page observations changed KV')
                    replay=None
                    if pos==0:
                        t0=time.perf_counter()
                        context['action']='full_replay'
                        cache.crop(length); pager.force_full=True
                        replayed=step(draft,ids[:,4+pos:5+pos],cache,pager)[0,-1].cpu()
                        pager.force_full=False
                        replay_sha,replay_size=fingerprint(cache,length+1)
                        replay_error=relative_l2(high,replayed)
                        replay=dict(wall_seconds=time.perf_counter()-t0,relative_l2=replay_error,
                            argmax_match=int(high.argmax())==int(replayed.argmax()),kv_sha=replay_sha,
                            kv_d2h_bytes=replay_size,logit_d2h_bytes=151936*4)
                        saved['replay']=replayed
                        if replay_error>.01 or not replay['argmax_match'] or replay_sha!=post_sha:
                            raise RuntimeError('Full replay/readout/KV identity')
                    summed=base_y_cpu+delta_matrix.sum(0)
                    grouped_error=relative_l2(canonical_y_cpu,summed)
                    if grouped_error>.01:
                        raise RuntimeError('Grouped final FFN reconstruction')
                    row=dict(document=doc,split=split,position=pos,consumed_token=tokens[doc][4+pos],
                        token_counter=pager.cache.token,base_length=length,end_length=cache.get_seq_length(),
                        prior_sha=prior_sha,prior_after_sha=old_after,post_sha=post_sha,post_after_sha=after,
                        kv_fingerprint_d2h_bytes=prior_size+post_size+old_size+after_size,
                        kv_logical_bytes=kv_bytes(cache)+kv_bytes(tcache),
                        kv_storage_bytes=kv_storage_bytes(cache)+kv_storage_bytes(tcache),
                        base_ids=[u,v],high_id=winner,oracle_pages=selected,epsilon=eps,
                        base_seconds=base_seconds,target_seconds=target_seconds,observations=observations,
                        high_readout_seconds=high_readout_seconds,oracle_selection_seconds=oracle_selection_seconds,
                        fixed_seconds=fixed_seconds,oracle_seconds=oracle_seconds,
                        replay=replay,grouped_relative_l2=grouped_error,
                        full_logit_d2h_bytes=5*151936*4,correction_scalar_d2h_bytes=27*4*(1+int(pos==0)),
                        auxiliary_d2h_bytes=12*1536*4+4,
                        delta_d2h_bytes=35*1536*4)
                    save_file(saved,folder/'observations.safetensors')
                    raw.record(row); raw.flush(); pages.flush()
                    phase('frame',started,document=doc,split=split,position=pos)
                    del saved,delta_matrix,deltas,h,residual,base_y,canonical_y,fixed_y,oracle_y
                    if pos%4==3:
                        print(f'Document {di}, positions {pos-2}..{pos+1} complete ({split})',flush=True)
                del ids,cache,tcache
            write(output/'completion.json',dict(complete=True))
        write(output/'final-resources.json',resources.receipt())
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        pages.close(); raw.close(); phases.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    from .output_sensors_analysis import analyze
    if args.worker:
        worker(args.output.resolve())
    else:
        result=analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.output_sensors',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'):
            raise SystemExit(1)


if __name__=='__main__':
    main()

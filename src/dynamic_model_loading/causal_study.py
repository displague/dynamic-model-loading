"""Incremental Qwen selector development control; dense diagnostic execution only."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load, load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import extract_ffns
from . import causal
from .cache import dense_static_frontier, group_arrays, replay
from .experiment import digest, environment, load_corpus, write_json
from .ffn import dimensions, grouped_forward, repack_
from .metrics import aggregate, compare_logits, relative_l2


@torch.inference_mode()
def incremental(model, ids):
    past, values = None, []
    for index in range(ids.shape[1]):
        result = model(input_ids=ids[:,index:index+1],past_key_values=past,use_cache=True)
        past = result.past_key_values
        values.append(result.logits.cpu())
    return torch.cat(values,1)


@torch.inference_mode()
def generate(model, prompt, count):
    past = None
    for index in range(prompt.shape[1]):
        result = model(input_ids=prompt[:,index:index+1],past_key_values=past,use_cache=True)
        past = result.past_key_values
    generated=[]
    for index in range(count):
        token=result.logits[:,-1].argmax(-1,keepdim=True)
        generated.append(token)
        if index+1<count:
            result=model(input_ids=token,past_key_values=past,use_cache=True)
            past=result.past_key_values
    return torch.cat((prompt,*generated),1).cpu()


@contextmanager
def packed(mlps, orders):
    # Share the reviewed immutable-byte restoration routine delivered with v0.6.
    from .relu import restore_originals
    originals=[]
    try:
        for mlp, order in zip(mlps,orders,strict=True):
            targets=(mlp.gate_proj.weight,mlp.up_proj.weight,mlp.down_proj.weight)
            originals.append((targets,[t.detach().clone() for t in targets]))
            repack_(mlp,torch.tensor(order,dtype=torch.long))
        yield
    finally:
        restore_originals(originals)


def numerical_pass(metrics):
    return metrics['logit_relative_l2']<=.01 and metrics['mean_kl_dense_to_candidate']<=.001


def timing(mlps, selectors, fixtures, cfg):
    if not torch.cuda.is_available():
        raise ValueError('CUDA timing requires CUDA')
    output=[]
    for kind in ('dense','selector'):
        for repetition in range(cfg['timing_warmups']+cfg['timing_repetitions']):
            for selector in selectors:
                selector.reset()
            torch.cuda.synchronize()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();begin.record()
            for token in range(cfg['timing_tokens']):
                for mlp,selector,(x,z) in zip(mlps,selectors,fixtures,strict=True):
                    if kind=='dense':
                        mlp(x[token:token+1])
                    else:
                        mask=selector.query(x[token:token+1])
                        if selector.state is not None:
                            selected=z[token:token+1]*mask.repeat_interleave(selector.width)[:z.shape[-1]]
                            selector.observe_selected(selected,mask)
            end.record();torch.cuda.synchronize()
            output.append({'workload':kind,'repetition':repetition,'warmup':repetition<cfg['timing_warmups'],
                           'cuda_ms':begin.elapsed_time(end),'wall_ms':(time.perf_counter()-start)*1000})
    medians={kind:{metric:statistics.median(r[metric] for r in output if r['workload']==kind and not r['warmup'])
                   for metric in ('cuda_ms','wall_ms')} for kind in ('dense','selector')}
    return output,medians


@torch.inference_mode()
def run(config_path, parent_path, output_path, device='cuda'):
    output,config_path,parent=Path(output_path),Path(config_path),Path(parent_path)
    output.mkdir(parents=True,exist_ok=False)
    stream=(output/'results.jsonl').open('x',encoding='utf-8')
    def record(kind,**values):
        stream.write(json.dumps({'kind':kind,**values},allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    try:
        cfg=json.loads(config_path.read_text(encoding='utf-8'))
        if (cfg['group_width'],cfg['keep'],cfg['budget_bytes'],cfg['ridge'],cfg['features'],cfg['alpha']) != (8,.9,2**31,.01,64,.1):
            raise ValueError('Changed frozen operating point')
        if cfg['modes'] != ['static','recency','ema','learned','hindsight'] or cfg['expected_torch'] != str(torch.__version__):
            raise ValueError('Policy/environment mismatch')
        if (cfg['screen_relative_ppl_max'],cfg['screen_warm_savings_min'],cfg['screen_cost_ratio_max']) != (1.01,.1,.1):
            raise ValueError('Changed screen')
        for name,expected in cfg['parent_hashes'].items():
            if digest(parent/name)!=expected:raise ValueError('Parent hash mismatch: '+name)
        if device not in ('cuda','cpu') or device=='cuda' and not torch.cuda.is_available():
            raise ValueError('Requested device unavailable; no fallback')
        torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for source,name in ((config_path,'config.json'),(Path(cfg['protocol']),'protocol.md')):
            shutil.copyfile(source,output/name)
        shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('corpus.jsonl','token-ids.json','layouts.json.gz','calibration-importance.safetensors.gz'):
            shutil.copyfile(parent/name,output/name)
        started={'started_utc':datetime.now(timezone.utc).isoformat(),'environment':environment(torch.device(device)),
                 'config_sha256':digest(config_path),'protocol_sha256':digest(output/'protocol.md'),
                 'sources':{p.name:digest(p) for p in (output/'source').glob('*.py')},'parent_hashes':cfg['parent_hashes']}
        write_json(output/'started.json',started)
        corpus=load_corpus(parent/'corpus.jsonl')
        ids=json.loads((parent/'token-ids.json').read_text())
        tokens={name:torch.tensor(value,dtype=torch.long,device=device) for name,value in ids.items()}
        cal=[r['id'] for r in corpus if r['split']=='calibration']; docs=[r['id'] for r in corpus if r['split']=='diagnostic']
        if [len(cal),len(docs)] != cfg['document_counts']:raise ValueError('Changed corpus split')
        layouts=json.loads(gzip.decompress((parent/'layouts.json.gz').read_bytes()))
        saved_means=load(gzip.decompress((parent/'calibration-importance.safetensors.gz').read_bytes()))
        means=torch.stack([saved_means[str(i)] for i in range(cfg['layers'])]).numpy()
        sizes,importance=group_arrays(means,np.asarray(layouts['popularity']),cfg['group_width'],3*cfg['hidden']*4)
        print('Loading Qwen for incremental controls',flush=True)
        snapshot=Path(snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True))
        parent_manifest=json.loads((parent/'manifest.json').read_text())
        checkpoint={p.name:digest(p) for p in snapshot.iterdir() if p.is_file()}
        if checkpoint != {k:v['sha256'] for k,v in parent_manifest['checkpoint']['files'].items()}:
            raise ValueError('Checkpoint catalog differs from parent')
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).to(device).eval()
        mlps=extract_ffns(model)
        if [dimensions(m) for m in mlps] != [(cfg['hidden'],cfg['neurons'])]*cfg['layers']:raise ValueError('Shape mismatch')
        manifest={**started,'checkpoint_files':checkpoint,'model_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                  'groupable_ffn_bytes':int(sizes.sum()),'diagnostic_backup_bytes':int(sizes.sum()),
                  'attention_implementation':model.config._attn_implementation}
        write_json(output/'manifest.json',manifest)
        for name in ('reference_logits','traces','generation'):(output/name).mkdir()
        refs={}
        episodes=[(name,tokens[name],None) for name in docs]
        episodes.append(('transition',torch.cat((tokens[docs[0]],tokens[docs[1]]),1),tokens[docs[0]].shape[1]))
        for index,(name,value,boundary) in enumerate(episodes):
            dense=incremental(model,value); path=output/'reference_logits'/f'{index:03d}.safetensors'
            save_file({'logits':dense},path);refs[name]=path
            record('dense_reference',document=name,input_tokens=value.shape[1],logits_sha256=digest(path))
            if boundary is None:
                prefill=model(input_ids=value,use_cache=False).logits.cpu()
                record('prefill_incremental_drift',document=name,**compare_logits(prefill,dense,value.cpu()))
        dense_generations={}
        for name in docs[:cfg['generation_documents']]:
            dense_generations[name]=generate(model,tokens[name][:,:cfg['prompt_tokens']],cfg['generation_tokens'])
            value=dense_generations[name][0].tolist()
            record('dense_generation',document=name,token_ids=value,text=tokenizer.decode(value,skip_special_tokens=False))
        def reference(name):return load_file(refs[name])['logits']
        gate=True
        with packed(mlps,layouts['popularity']):
            captured=[]; handles=[]
            try:
                for mlp in mlps:
                    handles.append(mlp.register_forward_pre_hook(lambda module,args:captured.append(args[0].reshape(-1,cfg['hidden'])[:2].clone())))
                model(input_ids=tokens[cal[0]],use_cache=False)
            finally:
                for handle in handles:handle.remove()
            for index,(mlp,x) in enumerate(zip(mlps,captured,strict=True)):
                error=relative_l2(mlp(x),grouped_forward(mlp,x,cfg['group_width']))
                record('group_reconstruction',layer=index,relative_l2=error,passed=error<=.01);gate &= error<=.01
            del captured
            for name,value,boundary in episodes:
                metrics=compare_logits(reference(name),incremental(model,value),value.cpu())
                passed=numerical_pass(metrics);gate &= passed
                record('layout_correctness',document=name,passed=passed,**metrics)
            if not gate:
                summary={'status':'correctness_gate_failed','correctness_passed':False}
                write_json(output/'summary.json',summary);return summary
            print('Calibration-only ridge fitting',flush=True)
            fits=[causal.RidgeFit(cfg['hidden'],importance.shape[1],cfg['features'],cfg['seed']+i,device) for i in range(len(mlps))]
            norms=[m.down_proj.weight.detach().float().norm(dim=0) for m in mlps]
            pending={};fixtures={};handles=[]
            def before(index,args):pending[index]=args[0].reshape(-1,cfg['hidden'])
            def observe(index,args):
                x=pending.pop(index);z=args[0].reshape(-1,cfg['neurons'])
                fits[index].observe(x,causal.group_scores(z,norms[index],cfg['group_width']))
                if index not in fixtures:fixtures[index]=(x[:cfg['timing_tokens']].clone(),z[:cfg['timing_tokens']].clone())
            try:
                for index,mlp in enumerate(mlps):
                    handles.append(mlp.register_forward_pre_hook(lambda m,a,i=index:before(i,a)))
                    handles.append(mlp.down_proj.register_forward_pre_hook(lambda m,a,i=index:observe(i,a)))
                for name in cal:model(input_ids=tokens[name],use_cache=False)
            finally:
                for handle in handles:handle.remove()
            learned=[fit.finish(cfg['ridge']) for fit in fits]
            save_file({f'{i}.{k}':v for i,fit in enumerate(fits) for k,v in fit.tensors().items()},output/'calibration-statistics.safetensors')
            save_file({f'{i}.{k}':v for i,fit in enumerate(learned) for k,v in fit.items()},output/'learned.safetensors')
            record('calibration_fit',documents=cal,observations=[f.count for f in fits],
                   statistics_sha256=digest(output/'calibration-statistics.safetensors'),learned_sha256=digest(output/'learned.safetensors'),
                   diagnostic_statistics_bytes=sum(v.numel()*v.element_size() for f in fits for v in f.tensors().values()),
                   timing_fixture_bytes=sum(t.numel()*t.element_size() for pair in fixtures.values() for t in pair))
            del fits
            priors=[torch.tensor(row,dtype=torch.float32,device=device) for row in importance]
            results=[]
            for mode in cfg['modes']:
                print('Incremental selector: '+mode,flush=True)
                selectors=[causal.Selector(mode,p,n,cfg['group_width'],learned=l,keep=cfg['keep'],alpha=cfg['alpha'])
                           for p,n,l in zip(priors,norms,learned,strict=True)] if mode!='hindsight' else [None]*len(mlps)
                selector_bytes=sum(s.bytes() for s in selectors if s is not None)
                record('selector_storage',mode=mode,persistent_bytes=selector_bytes,
                       tensors=[{k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for k,v in s.tensors().items()}
                                for s in selectors if s is not None],
                       hindsight_cost_excluded=mode=='hindsight')
                measured=[];traffic=[];all_audits=[]
                for index,(name,value,boundary) in enumerate(episodes):
                    for selector in selectors:
                        if selector is not None:selector.reset()
                    observers=[causal.AppliedProbe(m,s,cfg['group_width'],cfg['keep']) for m,s in zip(mlps,selectors,strict=True)]
                    with causal.probes(mlps,observers):candidate=incremental(model,value)
                    masks,audits=zip(*(observer.take() for observer in observers))
                    mask=torch.stack(masks,1).numpy();audit=torch.stack(audits,1).numpy()
                    trace_path=output/'traces'/f'{mode}-{index:03d}.npz'
                    np.savez_compressed(trace_path,bits=np.packbits(mask,axis=-1,bitorder='little'),shape=np.asarray(mask.shape),audits=audit)
                    metrics=compare_logits(reference(name),candidate,value.cpu())
                    replayed=replay(mask,sizes,importance,cfg['budget_bytes']-selector_bytes,'static_equal_layer')
                    record('selector_document',mode=mode,document=name,**metrics,trace_file=trace_path.relative_to(output).as_posix(),
                           trace_sha256=digest(trace_path),replay=replayed,selector_bytes=selector_bytes)
                    if boundary is None:
                        measured.append(metrics);traffic.append(replayed);all_audits.append(audit)
                    else:
                        record('transition_after_boundary',mode=mode,boundary=boundary,
                            **compare_logits(reference(name)[:,boundary-1:],candidate[:,boundary-1:],value.cpu()[:,boundary-1:]),
                            replay=replay(mask[boundary:],sizes,importance,cfg['budget_bytes']-selector_bytes,'static_equal_layer'),
                            note='Post-boundary cold row is a separate preload counterfactual; use warm demand for carried static cache.')
                if mode!='hindsight' and device=='cuda':
                    timed,medians=timing(mlps,selectors,[fixtures[i] for i in range(len(mlps))],cfg)
                    for row in timed:record('selector_timing',mode=mode,**row)
                    cost={key:medians['selector'][key]/medians['dense'][key] for key in ('cuda_ms','wall_ms')}
                else:medians=cost=None
                for name in docs[:cfg['generation_documents']]:
                    for selector in selectors:
                        if selector is not None:selector.reset()
                    observers=[causal.AppliedProbe(m,s,cfg['group_width'],cfg['keep']) for m,s in zip(mlps,selectors,strict=True)]
                    with causal.probes(mlps,observers):value=generate(model,tokens[name][:,:cfg['prompt_tokens']],cfg['generation_tokens'])
                    masks,audits=zip(*(observer.take() for observer in observers))
                    generation_trace=output/'generation'/f'{mode}-{docs.index(name):03d}.npz'
                    mask=torch.stack(masks,1).numpy()
                    np.savez_compressed(generation_trace,bits=np.packbits(mask,axis=-1,bitorder='little'),
                                        shape=np.asarray(mask.shape),audits=torch.stack(audits,1).numpy())
                    payload={'mode':mode,'document':name,'token_ids':value[0].tolist(),
                             'text':tokenizer.decode(value[0].tolist(),skip_special_tokens=False),
                             'generated_token_agreement_with_dense':float((value[:,cfg['prompt_tokens']:]==dense_generations[name][:,cfg['prompt_tokens']:]).float().mean()),
                             'trace_file':generation_trace.relative_to(output).as_posix(), 'trace_sha256':digest(generation_trace)}
                    generation_path=output/'generation'/f'{mode}-{docs.index(name):03d}.json';write_json(generation_path,payload)
                    record('selector_generation',**payload,artifact_sha256=digest(generation_path))
                quality=aggregate(measured)
                baseline=dense_static_frontier(means,layouts,[8,32,128],[cfg['budget_bytes']],3*cfg['hidden']*4,[tokens[d].shape[1] for d in docs])
                warm=sum(next(r['total_bytes'] for r in rows if r['state']=='warm') for rows in traffic)
                dense_bytes=baseline[(cfg['budget_bytes'],'warm')]['total_bytes']
                saving=1-warm/dense_bytes if dense_bytes else None
                audits=np.concatenate([a.reshape(-1,3) for a in all_audits],0)
                result={'mode':mode,**quality,'selector_bytes':selector_bytes,'warm_bytes':warm,'warm_saving':saving,
                        'dense_baseline':baseline[(cfg['budget_bytes'],'warm')],'cost_ratios':cost,'timing_medians':medians,
                        'audit_mean':audits.mean(0).tolist(),'audit_p95':np.quantile(audits,.95,axis=0).tolist(),
                        'audit_local_error_above_point1_fraction':float((audits[:,0]>.1).mean()),
                        'passes_screen':mode!='hindsight' and cost is not None and saving is not None
                            and quality['relative_perplexity']<=1.01 and saving>=.1 and max(cost.values())<=.1}
                record('selector_aggregate',**result);results.append(result)
            summary={'status':'causal_control_completed' if device=='cuda' else 'cpu_fixture_completed',
                     'correctness_passed':True,'selectors':results,
                     'nominated_selector':next(iter(sorted((r['mode'] for r in results if r['passes_screen']),
                         key=lambda name:next((r['warm_bytes'],r['mean_kl_dense_to_candidate'],name) for r in results if r['mode']==name))),None),
                     'finished_utc':datetime.now(timezone.utc).isoformat()}
        # Complete restoration before publishing a successful summary.
        write_json(output/'summary.json',summary);return summary
    except (Exception,KeyboardInterrupt) as error:
        record('error',error_type=type(error).__name__,message=str(error))
        write_json(output/'failure.json',{'error_type':type(error).__name__,'message':str(error)})
        raise
    finally:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','parent','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=run(args.config,args.parent,args.output)
    print(json.dumps({'status':result['status']}))
    return 0 if result['status']=='causal_control_completed' else 2


if __name__=='__main__':raise SystemExit(main())

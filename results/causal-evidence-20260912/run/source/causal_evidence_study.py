"""Bounded calibration and own-trajectory evaluation of one-round causal repair."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
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

from . import causal_evidence as ce
from .adapters import extract_ffns
from .balanced_control import quality_metrics, quality_aggregate
from .cache import cache_shape, dense_static_frontier, group_arrays, hot_mask
from .causal_study import incremental, numerical_pass, packed
from .dense_interface import aligned_metrics, decoding, incremental_generate, prepare, score
from .experiment import digest, environment, write_json
from .provenance import committed_inputs, frozen_environment


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def load_inputs(root, cfg):
    parent = root/cfg['parent']
    for name,sha in cfg['parent_hashes'].items():
        if digest(parent/name)!=sha: raise ValueError('Parent identity mismatch: '+name)
    for path,key in [(cfg['hardware'],'hardware_sha256'),(cfg['interface_config'],'interface_sha256'),(cfg['task_corpus'],'task_sha256')]:
        if digest(root/path)!=cfg[key]: raise ValueError('Frozen input mismatch: '+path)
    corpus = [json.loads(line) for line in (parent/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    tokens = read(parent/'token-ids.json')
    calibration = [r['id'] for r in corpus if r['split']=='calibration'][:4]
    development = [r['id'] for r in corpus if r['split']=='diagnostic'][:4]
    if len(calibration)!=4 or len(development)!=4 or set(calibration)&set(development):
        raise ValueError('Calibration/development split mismatch')
    ids = {name:torch.tensor(tokens[name],dtype=torch.long)[:,:128] for name in calibration+development}
    if any(t.shape!=(1,128) for t in ids.values()): raise ValueError('Token prefix mismatch')
    means_tensors = load(gzip.decompress((parent/'calibration-importance.safetensors.gz').read_bytes()))
    means = torch.stack([means_tensors[str(i)] for i in range(28)]).numpy()
    layouts = json.loads(gzip.decompress((parent/'layouts.json.gz').read_bytes()))
    sizes, importance = group_arrays(means,np.asarray(layouts['popularity']),8,3*1536*4)
    budget = cfg['budget_bytes']-cfg['controller_reservation_bytes']-cfg['acquisition_workspace_bytes']
    hot = hot_mask(importance,sizes,cache_shape(sizes,budget)['retained_slots'],'static_equal_layer')
    dense_budget=cfg['budget_bytes']-cfg['acquisition_workspace_bytes']
    baseline = dense_static_frontier(means,layouts,[8,32,128],[dense_budget],3*1536*4,[128]*4)[(dense_budget,'warm')]
    tasks = [json.loads(line) for line in (root/cfg['task_corpus']).read_text(encoding='utf-8').splitlines()]
    return dict(calibration=calibration,development=development,ids=ids,means=means,layouts=layouts,
                importance=importance,sizes=sizes,hot=hot,baseline=baseline,tasks=tasks)


def save_tensors(output, label, tensors):
    path = output/'tensors'/f'{label}.safetensors'
    save_file({key:value.detach().cpu().clone().contiguous() for key,value in tensors.items()},path)
    return {'file':path.relative_to(output).as_posix(),'sha256':digest(path)}


def controllers(parent, models, mode, extra, device):
    return [ce.Controller(torch.tensor(parent['importance'][i],dtype=torch.float32,device=device),
                torch.tensor(parent['hot'][i],device=device),ce.projections(1536,1120,1729+i,device),
                None if models is None else models[i],mode,extra) for i in range(28)]


@contextmanager
def finite_observer(model, output, label):
    calls, failures = [], []
    def check(module,args,result):
        good = bool(torch.isfinite(result.logits).all()); calls.append(good)
        if not good: failures.append(save_tensors(output,f'{label}-nonfinite-{len(calls)}',{'logits':result.logits}))
    handle = model.register_forward_hook(check)
    try: yield {'calls':calls,'failures':failures}
    finally: handle.remove()


def execute(model, mlps, parent, models, mode, extra, ids, config=None, collect=False, label=None, forced=None):
    control = controllers(parent,models,mode,extra,ids.device)
    observers = [ce.Probe(m,c,collect) for m,c in zip(mlps,control,strict=True)]
    for layer,probe in enumerate(observers):probe.layer=layer
    consumed=[]
    def remember(module,args,kwargs):
        consumed.append(kwargs['input_ids'].detach().cpu().clone())
    handle=model.register_forward_pre_hook(remember,with_kwargs=True)
    try:
        with ce.apply(mlps,observers):
            value = incremental(model,ids) if config is None else incremental_generate(model,ids,config,forced=forced)
    except ce.ExecutionFailure as error:
        error.context.update(label=label,generated=config is not None)
        error.tensors['token_inputs']=ids.detach().cpu()
        error.tensors['consumed_token_inputs']=torch.cat(consumed,1)
        for layer,probe in enumerate(observers):
            if probe.receipts:
                for key in probe.receipts[0]:
                    error.tensors[f'partial_trace.{layer}.{key}']=torch.stack([r[key] for r in probe.receipts])
        raise
    finally:handle.remove()
    return value, ce.payload(observers), ce.payload(observers,True) if collect else None


def model_bytes(models, parent):
    tensors = list(ce.flatten_models(models).values())
    total = sum(v.numel()*v.element_size() for v in tensors)
    # Explicitly reserve mutable state, projected inputs, prior copies, norms,
    # boolean masks, all model/projection tensors, preserved input and FFN scratch.
    projection = 28*(1536*8+1120*8)*4
    persistent = 28*(1120*6*4+8960*4+1120*4+8*4)
    scratch = 28*(1536*4*3+8960*4*3+1120*4*6+128*4)
    return {'fitted_tensor_bytes':total,'projection_bytes':projection,'state_norm_mask_bytes':persistent,
            'preserved_input_and_scratch_bytes':scratch,'total':total+projection+persistent+scratch}


def policy_summary(rows, baseline_bytes):
    result = []
    for mode,extra in ce.CONDITIONS:
        selected = [r for r in rows if r['kind']=='wiki' and r['mode']==mode and r['extra']==extra]
        quality = quality_aggregate([r['quality'] for r in selected])
        warm = sum(r['traffic']['initial_cold_bytes']+r['traffic']['repair_cold_bytes']+r['traffic']['dispatch_bytes'] for r in selected)
        saving = 1-warm/baseline_bytes
        result.append({'mode':mode,'extra':extra,'quality':quality,'warm_bytes':warm,'warm_saving':saving,
            'individual_quality_passes':sum(r['quality'] is not None and r['quality']['relative_perplexity']<=1.01 for r in selected),
            'gate_b_wiki':quality is not None and quality['relative_perplexity']<=1.01 and all(
                r['finite'] and r['quality'] is not None and r['quality']['relative_perplexity']<=1.01 for r in selected),
            'traffic_pass':saving>=.1,
            'round_fraction':sum(r['traffic']['rounds'] for r in selected)/sum(r['traffic']['visits'] for r in selected),
            'fallback_fraction':sum(r['traffic']['fallbacks'] for r in selected)/sum(r['traffic']['visits'] for r in selected)})
    return result


@torch.inference_mode()
def run(config_path, output_path):
    config_path = Path(config_path).resolve(); root = config_path.parent.parent
    cfg = read(config_path); source,protocol = committed_inputs(__file__,config_path,cfg['protocol'])
    parent = load_inputs(root,cfg)
    if not torch.cuda.is_available(): raise ValueError('CUDA required')
    torch.set_num_threads(4);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=environment(torch.device('cuda'));frozen_environment(env)
    output=Path(output_path).resolve();output.mkdir(parents=True,exist_ok=False);(output/'tensors').mkdir()
    shutil.copyfile(config_path,output/'config.json');shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(output/'manifest.json',{**source,'environment':env,'started_utc':datetime.now(timezone.utc).isoformat()})
    write_json(output/'inputs.json',{'calibration':parent['calibration'],'development':parent['development'],
        'tokens':{k:v.tolist() for k,v in parent['ids'].items()},'tasks':parent['tasks'],'hot':parent['hot'].tolist(),
        'dense_baseline':parent['baseline']})
    stream=(output/'results.jsonl').open('x',encoding='utf-8');rows=[]
    def record(kind,**values):
        row={'kind':kind,**values};rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    try:
        spec=cfg['model']; checkpoint=Path(snapshot_download(spec['model'],revision=spec['revision'],local_files_only=True))
        catalog={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
        if catalog!=read(root/spec['parent']/'manifest.json')['checkpoint_files']:raise ValueError('Checkpoint mismatch')
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        config=decoding(checkpoint)
        model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True,
            trust_remote_code=False).cuda().eval();mlps=extract_ffns(model)
        write_json(output/'model.json',{'checkpoint_files':catalog,'generation_config':config.to_dict(),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'diagnostic_backup_bytes':4624220160})
        refs={};task_inputs={}
        for name in parent['development']:
            value=incremental(model,parent['ids'][name].cuda());refs[name]=save_tensors(output,'native-'+name,{'logits':value})
            record('native_wiki',document=name,tensors=refs[name],finite=bool(torch.isfinite(value).all()))
        for task in parent['tasks']:
            name=task['id'];prompt=prepare(tokenizer,task,'chat');answer=tokenizer(task['answer'],add_special_tokens=False,return_tensors='pt')['input_ids']
            ids=torch.cat((prompt,answer),1);task_inputs[name]=(prompt,ids)
            value=incremental(model,ids.cuda())
            with finite_observer(model,output,'native-'+name) as observed:
                generated,gen_logits=incremental_generate(model,prompt.cuda(),config)
            refs[name]=save_tensors(output,'native-'+name,{'logits':value,'generation_logits':gen_logits})
            text=tokenizer.decode(generated,skip_special_tokens=True)
            record('native_task',document=name,domain=task['domain'],prompt_ids=prompt[0].tolist(),teacher_ids=ids[0].tolist(),
                token_ids=generated,text=text,tensors=refs[name],finite=bool(torch.isfinite(value).all()) and all(observed['calls']),
                generation=observed,**score(text,task))
        with packed(mlps,parent['layouts']['popularity']):
            models=None
            for pass_index in (1,2):
                collected=[]
                for name in parent['calibration']:
                    mode='predetermined' if pass_index==1 else 'partial'
                    _,receipt,data=execute(model,mlps,parent,models,mode,28,parent['ids'][name].cuda(),collect=True,label=f'calibration-{pass_index}-{name}')
                    collected.append(data)
                    record('calibration',pass_index=pass_index,document=name,
                        examples=save_tensors(output,f'calibration-{pass_index}-{name}',data),
                        trace=save_tensors(output,f'calibration-trace-{pass_index}-{name}',receipt))
                combined={key:torch.cat([d[key] for d in collected],0) for key in collected[0]}
                fitted=ce.fit_models(combined)
                fitted_file=save_tensors(output,f'fitted-{pass_index}',ce.flatten_models(fitted))
                record('fit',pass_index=pass_index,tensors=fitted_file,examples=512,storage=model_bytes(fitted,parent))
                models=ce.unflatten_models(ce.flatten_models(fitted),28,'cuda')
                if model_bytes(fitted,parent)['total']>cfg['controller_reservation_bytes']:
                    raise ValueError('Controller exceeds common reservation')
                print('Completed calibration pass',pass_index,flush=True)
            numerical=True
            # Full completion must reconstruct before any partial-policy evaluation.
            for name in parent['development']:
                value,receipt,_=execute(model,mlps,parent,models,'full',112,parent['ids'][name].cuda(),label='completion-'+name)
                reference=load_file(output/refs[name]['file'])['logits'];quality=quality_metrics(reference,value,parent['ids'][name])
                passed=quality is not None and numerical_pass(quality);numerical &= passed
                record('completion_control',document=name,quality=quality,passed=passed,
                    tensors=save_tensors(output,'completion-'+name,{'logits':value}),trace=save_tensors(output,'completion-trace-'+name,receipt))
            for task in parent['tasks']:
                name=task['id'];prompt,ids=task_inputs[name]
                value,receipt,_=execute(model,mlps,parent,models,'full',112,ids.cuda(),label='completion-'+name)
                with finite_observer(model,output,'completion-'+name) as observed:
                    (generated,gen_logits),gen_receipt,_=execute(model,mlps,parent,models,'full',112,prompt.cuda(),config,label='completion-generation-'+name)
                ref=load_file(output/refs[name]['file']);quality=quality_metrics(ref['logits'],value,ids)
                native_row=next(r for r in rows if r['kind']=='native_task' and r['document']==name)
                aligned_logits,aligned_trace=gen_logits,gen_receipt
                aligned_observed=observed
                if generated!=native_row['token_ids']:
                    with finite_observer(model,output,'completion-aligned-'+name) as aligned_observed:
                        (_,aligned_logits),aligned_trace,_=execute(model,mlps,parent,models,'full',112,prompt.cuda(),config,
                            label='completion-aligned-'+name,forced=native_row['token_ids'])
                gen_quality=aligned_metrics(ref['generation_logits'],aligned_logits)
                passed=(quality is not None and numerical_pass(quality) and generated==native_row['token_ids']
                    and all(observed['calls']) and all(aligned_observed['calls']) and gen_quality is not None
                    and gen_quality['relative_l2']<=.01 and gen_quality['mean_kl']<=.001)
                numerical &= passed
                gen_trace_file=save_tensors(output,'completion-gen-trace-'+name,gen_receipt)
                aligned_trace_file=gen_trace_file if aligned_trace is gen_receipt else save_tensors(output,'completion-aligned-trace-'+name,aligned_trace)
                record('task_completion_control',document=name,quality=quality,passed=passed,token_ids=generated,generation=observed,
                    generation_quality=gen_quality,aligned_generation=aligned_observed,
                    aligned_generation_trace=aligned_trace_file,
                    tensors=save_tensors(output,'completion-'+name,{'logits':value,'generation_logits':gen_logits,'aligned_generation_logits':aligned_logits}),
                    trace=save_tensors(output,'completion-trace-'+name,receipt),generation_trace=gen_trace_file)
            if not numerical: raise ValueError('Full-completion numerical gate failed')
            for mode,extra in ce.CONDITIONS:
                condition=f'{mode}-{extra}'
                for name in parent['development']:
                    value,receipt,_=execute(model,mlps,parent,models,mode,extra,parent['ids'][name].cuda(),label=condition+'-'+name)
                    ref=load_file(output/refs[name]['file'])['logits'];quality=quality_metrics(ref,value,parent['ids'][name])
                    record('wiki',mode=mode,extra=extra,document=name,quality=quality,finite=quality is not None,
                        tensors=save_tensors(output,condition+'-'+name,{'logits':value}),
                        trace=save_tensors(output,condition+'-trace-'+name,receipt),traffic=ce.traffic(receipt,parent['hot']))
                if extra==28 and mode in ('one_shot','predetermined','partial'):
                    for task in parent['tasks']:
                        name=task['id'];prompt,ids=task_inputs[name]
                        value,receipt,_=execute(model,mlps,parent,models,mode,extra,ids.cuda(),label=condition+'-'+name)
                        with finite_observer(model,output,condition+'-'+name) as observed:
                            (generated,gen_logits),gen_receipt,_=execute(model,mlps,parent,models,mode,extra,prompt.cuda(),config,label=condition+'-generation-'+name)
                        ref=load_file(output/refs[name]['file']);quality=quality_metrics(ref['logits'],value,ids)
                        text=tokenizer.decode(generated,skip_special_tokens=True)
                        record('task',mode=mode,extra=extra,document=name,domain=task['domain'],quality=quality,
                            finite=quality is not None and all(observed['calls']),token_ids=generated,text=text,generation=observed,**score(text,task),
                            answer_quality=quality_metrics(ref['logits'][:,prompt.shape[1]-1:],value[:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:]),
                            tensors=save_tensors(output,condition+'-'+name,{'logits':value,'generation_logits':gen_logits}),
                            trace=save_tensors(output,condition+'-trace-'+name,receipt),generation_trace=save_tensors(output,condition+'-gen-trace-'+name,gen_receipt),
                            traffic=ce.traffic(receipt,parent['hot']),generation_traffic=ce.traffic(gen_receipt,parent['hot']))
                print('Completed causal condition',condition,flush=True)
            # Timing is a separate command: all evaluation artifacts survive it,
            # and the caller can ensure tests/reviews are idle before timing.
        summary={'status':'causal_evidence_measured','conditions':policy_summary(rows,parent['baseline']['total_bytes']),
                 'runtime_nomination':None,'timing_pending':True}
        write_json(output/'summary.json',summary);return summary
    except (Exception,KeyboardInterrupt) as error:
        evidence={}
        if isinstance(error,ce.ExecutionFailure):
            evidence={'context':error.context,'tensors':save_tensors(output,'failure-evidence',error.tensors)}
        record('error',type=type(error).__name__,message=str(error),**evidence)
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error),**evidence});raise
    finally:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.config,args.output),indent=2))


if __name__=='__main__':main()

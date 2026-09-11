"""Small frozen task control: each checkpoint is compared with its own dense output."""

import argparse
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
import gc
import gzip
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import relu
from .adapters import extract_ffns
from .causal import group_scores, rank_mask
from .causal_study import generate, incremental, numerical_pass, packed
from .experiment import digest, environment, write_json
from .ffn import dimensions, grouped_forward
from .metrics import aggregate, compare_logits, relative_l2
from .provenance import committed_inputs, frozen_environment


PREFIX='Return only the requested answer on one line, without explanation.\n'


def prepare_ids(tokenizer, task):
    prompt=PREFIX+task['task']+'\nAnswer:'
    first=tokenizer(prompt,add_special_tokens=False,return_tensors='pt')['input_ids']
    answer=tokenizer(task['answer'],add_special_tokens=False,return_tensors='pt')['input_ids']
    if first.shape[1]<1 or answer.shape[1]<1 or first.shape[1]+answer.shape[1]>256:
        raise ValueError('Frozen task token bound failed')
    return first,torch.cat((first,answer),1)


def score_generation(tokenizer, generated, expected):
    eos=tokenizer.eos_token_id
    scored=generated[:generated.index(eos)] if eos is not None and eos in generated else generated
    text=tokenizer.decode(scored,skip_special_tokens=True)
    line=next((line.strip() for line in text.splitlines() if line.strip()),'')
    return {'scored_text':text,'first_nonempty_line':line,'exact_target_match':text.strip()==expected}


def ffn_volume(key, layers, hidden, neurons, visits, selected_groups, width=8):
    if key not in ('qwen','opt') or neurons%width or not 0<=selected_groups<=visits*layers*(neurons//width):
        raise ValueError('Invalid FFN accounting dimensions')
    bytes_per_neuron=(2*hidden+1)*4 if key=='opt' else 3*hidden*4
    bias_bytes=layers*hidden*4 if key=='opt' else 0
    return (visits*(layers*neurons*bytes_per_neuron+bias_bytes),
            selected_groups*width*bytes_per_neuron+visits*bias_bytes)


def validate_config(cfg):
    if cfg['protocol']!='docs/balanced-development-protocol.md' or cfg['corpus']!='data/balanced-development.jsonl' or set(cfg['models'])!={'qwen','opt'}:
        raise ValueError('Changed balanced-control configuration')
    for key,name,revision,shape,layers in (
        ('qwen','Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306',[1536,8960],28),
        ('opt','facebook/opt-1.3b','3f5c25d0bc631cb57ac65913f76e22c2dfb61d62',[2048,8192],24)):
        value=cfg['models'][key]
        if (value['model'],value['revision'],value['shapes'])!=(name,revision,[shape]*layers):
            raise ValueError('Changed model pair')


class Probe:
    def __init__(self, down, mode, width=8):
        if mode not in ('static','hindsight'):raise ValueError('Unknown balanced-control mask')
        self.down,self.mode,self.width=down,mode,width
        self.neurons=down.weight.shape[1];self.groups=math.ceil(self.neurons/width)
        self.norms=None
        if mode=='hindsight':self.norms=down.weight.detach().float().norm(dim=0)
        self.fixed=torch.arange(self.groups,device=down.weight.device)<math.ceil(.9*self.groups)
        self.masks=[]
    def __call__(self,module,args):
        z=args[0]
        if z.numel()!=self.neurons:raise ValueError('Incremental single-token control required')
        mask=self.fixed if self.mode=='static' else rank_mask(group_scores(z,self.norms,self.width),.9).reshape(-1)
        self.masks.append(mask.detach().cpu())
        return (z*mask.repeat_interleave(self.width)[:self.neurons],*args[1:])


@contextmanager
def masks(downs, mode):
    observers=[Probe(down,mode) for down in downs];handles=[]
    try:
        for down,observer in zip(downs,observers,strict=True):handles.append(down.register_forward_pre_hook(observer))
        yield observers
    finally:
        for handle in handles:handle.remove()


def save_masks(path, observers):
    value=np.stack([torch.stack(p.masks).numpy() for p in observers],axis=1)
    np.savez_compressed(path,bits=np.packbits(value,axis=-1,bitorder='little'),shape=np.asarray(value.shape))
    return {'file':path.name,'sha256':digest(path),'shape':list(value.shape),'selected_groups':int(value.sum())}


def save_logits(root,name,logits):
    path=root/'references'/f'{name}.safetensors';save_file({'logits':logits},path)
    return {'file':path.relative_to(root).as_posix(),'sha256':digest(path)}


def reconstruction_metric(reference,candidate):
    reference,candidate=reference.cpu(),candidate.cpu()
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return None,'nonfinite_reconstruction'
    error=relative_l2(reference,candidate)
    return (error,None) if math.isfinite(error) else (None,'nonfinite_reconstruction_metric')


def quality_metrics(reference,candidate,ids):
    if (reference.shape!=candidate.shape or reference.ndim!=3 or reference.shape[0]!=1
            or ids.shape!=reference.shape[:2] or ids.shape[1]<2
            or reference.dtype!=torch.float32 or candidate.dtype!=torch.float32):
        raise ValueError('Invalid FP32 teacher-forced tensor dimensions')
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():return None
    return compare_logits(reference,candidate,ids)


def quality_aggregate(rows):
    return aggregate(rows) if all(row is not None for row in rows) else None


def generate_checked(model,prompt,output,label):
    finite=[];failures=[]
    def observe(module,args,result):
        logits=result.logits;good=bool(torch.isfinite(logits).all());step=len(finite);finite.append(good)
        if not good:
            failures.append({'step':step,**save_logits(output,f'{label}-nonfinite-{step}',logits.detach().cpu())})
    handle=model.register_forward_hook(observe)
    try:ids=generate(model,prompt,32)[0].tolist()
    finally:handle.remove()
    if len(finite)!=prompt.shape[1]+31:raise ValueError('Incomplete generation observation grid')
    return ids,{'finite_steps':finite,'nonfinite':failures}


@torch.inference_mode()
def run_model(key, spec, tasks, output, record):
    parent=Path(spec['parent'])
    for name,expected in spec['parent_hashes'].items():
        if digest(parent/name)!=expected:raise ValueError('Balanced-control parent mismatch')
    parent_manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    checkpoint=Path(snapshot_download(spec['model'],revision=spec['revision'],local_files_only=True))
    catalog={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
    if catalog!=parent_manifest['checkpoint_files']:raise ValueError('Balanced-control checkpoint mismatch')
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    options={'use_safetensors':False,'weights_only':True} if key=='opt' else {}
    model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,attn_implementation='sdpa',
        local_files_only=True,trust_remote_code=False,**options).cuda().eval()
    layers=relu.extract(model) if key=='opt' else extract_ffns(model)
    shape_function=relu.dimensions if key=='opt' else dimensions
    shapes=[list(shape_function(layer)) for layer in layers]
    if shapes!=spec['shapes']:raise ValueError('Balanced-control model shapes mismatch')
    downs=[layer.fc2 if key=='opt' else layer.down_proj for layer in layers]
    inputs={task['id']:prepare_ids(tokenizer,task) for task in tasks}
    write_json(output/f'{key}-token-ids.json',{name:{'prompt':p.tolist(),'teacher_forcing':v.tolist()} for name,(p,v) in inputs.items()})
    record('model_manifest',model=key,checkpoint_files=catalog,shapes=shapes,
        vocabulary_size=model.config.vocab_size,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        restoration_bytes=parent_manifest['diagnostic_layout_backup_bytes'] if key=='opt' else parent_manifest['diagnostic_backup_bytes'],
        token_ids_sha256=digest(output/f'{key}-token-ids.json'),attention_implementation=model.config._attn_implementation)
    native={};generations={};scores=[];native_valid=True
    for task in tasks:
        name=task['id'];prompt,ids=inputs[name];logits=incremental(model,ids.cuda());native[name]=logits
        path=output/'references'/f'{key}-{name}.safetensors';save_file({'logits':logits},path)
        generated,generation=generate_checked(model,prompt.cuda(),output,f'{key}-dense-{name}');generations[name]=generated
        native_valid &= all(generation['finite_steps']) and bool(torch.isfinite(logits).all())
        scored=score_generation(tokenizer,generated[prompt.shape[1]:],task['answer']);scores.append(scored)
        record('dense_task',model=key,document=name,reference_file=path.relative_to(output).as_posix(),reference_sha256=digest(path),
            prompt_tokens=prompt.shape[1],answer_tokens=ids.shape[1]-prompt.shape[1],token_ids=generated,generation=generation,
            text=tokenizer.decode(generated,skip_special_tokens=False),**scored,
            numerical_error=None if torch.isfinite(logits).all() else 'nonfinite_logits',
            answer_metrics=quality_metrics(logits[:,prompt.shape[1]-1:],logits[:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:]),
            full_metrics=quality_metrics(logits,logits,ids))
    orders=json.loads(gzip.decompress((parent/'layouts.json.gz').read_bytes()))['popularity']
    layout=relu.layout(layers,[torch.tensor(order,dtype=torch.long) for order in orders]) if key=='opt' else packed(layers,orders)
    gate=native_valid;condition_rows=[];masked_generation_valid=True;masked_teacher_valid=True
    with layout:
        captured=[];handles=[]
        try:
            for layer in layers:
                source=layer.fc1 if key=='opt' else layer
                handles.append(source.register_forward_pre_hook(lambda module,args:captured.append(args[0].reshape(-1,shapes[0][0])[:2].clone())))
            model(input_ids=inputs[tasks[0]['id']][1].cuda(),use_cache=False)
        finally:
            for handle in handles:handle.remove()
        for index,(layer,x) in enumerate(zip(layers,captured,strict=True)):
            reference=relu.dense_forward(layer,x) if key=='opt' else layer(x)
            grouped=relu.grouped_forward(layer,x,8) if key=='opt' else grouped_forward(layer,x,8)
            path=output/'references'/f'{key}-group-{index}.safetensors'
            save_file({'reference':reference.cpu(),'candidate':grouped.cpu()},path)
            saved={'file':path.relative_to(output).as_posix(),'sha256':digest(path)}
            error,failure=reconstruction_metric(reference,grouped);passed=failure is None and error<=.01;gate &= passed
            record('group_reconstruction',model=key,layer=index,relative_l2=error,passed=passed,numerical_error=failure,tensors=saved)
        del captured
        for task in tasks:
            name=task['id'];prompt,ids=inputs[name]
            value=incremental(model,ids.cuda())
            measured=quality_metrics(native[name],value,ids);finite=measured is not None
            metrics=measured if finite else {}
            saved=save_logits(output,f'{key}-layout-{name}',value)
            generated,generation=generate_checked(model,prompt.cuda(),output,f'{key}-layout-{name}');equal=generated==generations[name]
            passed=finite and numerical_pass(metrics) and equal and all(generation['finite_steps']);gate &= passed
            record('layout_correctness',model=key,document=name,passed=passed,generated_equal=equal,token_ids=generated,logits=saved,
                numerical_error=None if finite else 'nonfinite_logits',generation=generation,**metrics)
        if gate:
            for mode in ('static','hindsight'):
                for task in tasks:
                    name=task['id'];prompt,ids=inputs[name]
                    with masks(downs,mode) as observers:logits=incremental(model,ids.cuda())
                    masked_teacher_valid &= bool(torch.isfinite(logits).all())
                    saved=save_logits(output,f'{key}-{mode}-{name}',logits)
                    trace=save_masks(output/'traces'/f'{key}-{mode}-{name}-teacher.npz',observers)
                    with masks(downs,mode) as observers:generated,generation=generate_checked(model,prompt.cuda(),output,f'{key}-{mode}-{name}')
                    masked_generation_valid &= all(generation['finite_steps'])
                    gen_trace=save_masks(output/'traces'/f'{key}-{mode}-{name}-generated.npz',observers)
                    scored=score_generation(tokenizer,generated[prompt.shape[1]:],task['answer'])
                    hidden,neurons=shapes[0]
                    full_bytes,selected_bytes=ffn_volume(key,len(layers),hidden,neurons,trace['shape'][0],trace['selected_groups'])
                    gen_full,gen_selected=ffn_volume(key,len(layers),hidden,neurons,gen_trace['shape'][0],gen_trace['selected_groups'])
                    row={'model':key,'mode':mode,'document':name,'trace':trace,'generation_trace':gen_trace,'logits':saved,'generation':generation,
                        'token_ids':generated,'text':tokenizer.decode(generated,skip_special_tokens=False),**scored,
                        'generated_token_agreement':float(np.equal(generated[prompt.shape[1]:],generations[name][prompt.shape[1]:]).mean()),
                        'full_ffn_bytes':full_bytes,'selected_ffn_bytes':selected_bytes,
                        'generation_full_ffn_bytes':gen_full,'generation_selected_ffn_bytes':gen_selected,
                        'numerical_error':None if torch.isfinite(logits).all() else 'nonfinite_logits',
                        'answer_metrics':quality_metrics(native[name][:,prompt.shape[1]-1:],logits[:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:]),
                        'full_metrics':quality_metrics(native[name],logits,ids)}
                    record('masked_task',**row);condition_rows.append(row)
                print('Completed balanced control',key,mode,flush=True)
    results=[]
    for mode in ('static','hindsight') if gate else ():
        rows=[r for r in condition_rows if r['mode']==mode]
        value={'model':key,'mode':mode,'tasks':len(rows),'exact_targets':sum(r['exact_target_match'] for r in rows),
            'dense_exact_targets':sum(r['exact_target_match'] for r in scores),
            'answer_metrics':quality_aggregate([r['answer_metrics'] for r in rows]),'full_metrics':quality_aggregate([r['full_metrics'] for r in rows]),
            'selected_ffn_fraction':sum(r['selected_ffn_bytes'] for r in rows)/sum(r['full_ffn_bytes'] for r in rows)}
        record('aggregate',**value);results.append(value)
    status=('completed' if masked_generation_valid else 'masked_generation_failed') if gate else 'numerical_gate_failed'
    if gate and not masked_teacher_valid:status='masked_numerical_failed'
    return {'model':key,'status':status,'aggregates':results,
            'dense_exact_targets':sum(r['exact_target_match'] for r in scores),'tasks':len(tasks)}


@torch.inference_mode()
def run(config_path, output_path):
    config_path=Path(config_path);output=Path(output_path);output.mkdir(parents=True,exist_ok=False)
    stream=(output/'results.jsonl').open('x',encoding='utf-8')
    def record(kind,**values):
        stream.write(json.dumps({'kind':kind,**values},allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    try:
        cfg=json.loads(config_path.read_text(encoding='utf-8'))
        validate_config(cfg)
        source,protocol=committed_inputs(__file__,config_path,cfg['protocol'])
        if 'configs/balanced-development.json' not in source['committed_files']:raise ValueError('Use the committed balanced configuration path')
        corpus=Path(cfg['corpus'])
        if digest(corpus)!=cfg['corpus_sha256']:raise ValueError('Frozen task corpus mismatch')
        tasks=[json.loads(line) for line in corpus.read_text(encoding='utf-8').splitlines()]
        if (len(tasks)!=10 or len({r['id'] for r in tasks})!=10
                or Counter(r['domain'] for r in tasks)!={d:2 for d in ('code','extraction','arithmetic','copying','topic_changes')}
                or any(not r['task'].strip() or not r['answer'].strip() or '\n' in r['answer'] for r in tasks)):
            raise ValueError('Task grid mismatch')
        if not torch.cuda.is_available():raise ValueError('CUDA required; no fallback')
        torch.set_num_threads(4);torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        shutil.copyfile(config_path,output/'config.json');shutil.copyfile(protocol,output/'protocol.md');shutil.copyfile(corpus,output/'corpus.jsonl')
        shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
        manifest={**source,'started_utc':datetime.now(timezone.utc).isoformat(),'environment':environment(torch.device('cuda')),
            'config_sha256':digest(output/'config.json'),'protocol_sha256':digest(output/'protocol.md'),'corpus_sha256':digest(output/'corpus.jsonl'),
            'sources':{p.relative_to(output/'source').as_posix():digest(p) for p in (output/'source').rglob('*.py')}}
        write_json(output/'manifest.json',manifest);frozen_environment(manifest['environment'])
        for name in ('references','traces'):(output/name).mkdir()
        models=[]
        for key in ('qwen','opt'):
            models.append(run_model(key,cfg['models'][key],tasks,output,record))
            gc.collect();torch.cuda.empty_cache()
        summary={'status':'balanced_control_finished','models':models,'runtime_nomination':None,'finished_utc':datetime.now(timezone.utc).isoformat()}
        write_json(output/'summary.json',summary);return summary
    except (Exception,KeyboardInterrupt) as error:
        record('error',type=type(error).__name__,message=str(error));write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
    finally:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(run(args.config,args.output)['status'])


if __name__=='__main__':main()

"""Recompute the balanced development control from logits, IDs and applied masks."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .balanced_control import ffn_volume, prepare_ids, score_generation, validate_config, reconstruction_metric
from .balanced_control import quality_metrics, quality_aggregate
from .causal_analysis import require
from .causal_study import numerical_pass
from .experiment import digest, write_json
from .metrics import aggregate, compare_logits
from .provenance import frozen_environment, owning_repository, verify_snapshot


def checked_file(root, relative, expected):
    path=Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,'Unsafe evidence path')
    path=Path(root)/path
    require(digest(path)==expected,'Evidence checksum mismatch: '+relative)
    return path


def checked_masks(root,receipt,steps,layers,groups,mode):
    require(Path(receipt['file']).name==receipt['file'],'Trace must use a basename')
    path=checked_file(root/'traces',receipt['file'],receipt['sha256'])
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)=={'bits','shape'},'Trace fields mismatch')
        shape=data['shape'].tolist();bits=data['bits']
    require(shape==receipt['shape']==[steps,layers,groups] and bits.dtype==np.uint8
            and bits.shape==(steps,layers,math.ceil(groups/8)),'Trace shape mismatch')
    unpacked=np.unpackbits(bits,axis=-1,bitorder='little')
    require(not unpacked[:,:,groups:].any(),'Trace padding mismatch')
    mask=unpacked[:,:,:groups].astype(bool);keep=math.ceil(.9*groups)
    require(np.all(mask.sum(-1)==keep) and int(mask.sum())==receipt['selected_groups'],'Mask retention mismatch')
    if mode=='static':require(np.array_equal(mask,np.broadcast_to(np.arange(groups)<keep,mask.shape)),'Static mask changed')
    return int(mask.sum())


def generated_tokens(ids,prompt,vocabulary_size):
    first=prompt[0].tolist()
    require(all(type(i) is int and 0<=i<vocabulary_size for i in ids)
            and len(ids)==len(first)+32 and ids[:len(first)]==first,'Generated token grid mismatch')
    return ids[len(first):]


def check_generation(row,prompt,tokenizer,answer,vocabulary_size):
    ids=row['token_ids'];generated=generated_tokens(ids,prompt,vocabulary_size)
    expected=score_generation(tokenizer,generated,answer)
    require(all(row[key]==value for key,value in expected.items()),'Exact-answer scoring mismatch')
    require(row['text']==tokenizer.decode(ids,skip_special_tokens=False),'Decoded generation mismatch')
    return generated


def generation_valid(root,receipt,steps,vocabulary_size):
    flags=receipt['finite_steps']
    require(len(flags)==steps and all(type(v) is bool for v in flags),'Generation observation grid mismatch')
    require([r['step'] for r in receipt['nonfinite']]==[i for i,v in enumerate(flags) if not v],'Nonfinite generation inventory mismatch')
    for row in receipt['nonfinite']:
        logits=load_file(checked_file(root,row['file'],row['sha256']))['logits']
        require(logits.dtype==torch.float32 and logits.shape==(1,1,vocabulary_size)
                and not torch.isfinite(logits).all(),'Nonfinite generation evidence mismatch')
    return all(flags)


def analyze(path,repository=None):
    root=Path(path);repository=Path(repository) if repository else owning_repository(__file__)
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    cfg,manifest,summary=read('config.json'),read('manifest.json'),read('summary.json')
    validate_config(cfg)
    require(summary['status']=='balanced_control_finished' and summary['runtime_nomination'] is None
            and not (root/'failure.json').exists(),'Incomplete balanced control')
    for local,key in (('config.json','config_sha256'),('protocol.md','protocol_sha256'),('corpus.jsonl','corpus_sha256')):
        require(digest(root/local)==manifest[key],'Frozen input checksum mismatch')
    require(manifest['corpus_sha256']==cfg['corpus_sha256'],'Task corpus differs from committed configuration')
    verify_snapshot(root,manifest,'configs/balanced-development.json','docs/balanced-development-protocol.md',__file__)
    for name,value in manifest['sources'].items():require(digest(root/'source'/name)==value,'Source checksum mismatch')
    env=manifest['environment'];frozen_environment(env)
    require(env['device']=='cuda' and env['cpu_threads']==4 and env['tf32_matmul'] is False and env['tf32_cudnn'] is False,'Control environment mismatch')
    torch.set_num_threads(4)
    tasks=[json.loads(line) for line in (root/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(tasks)==10 and len({r['id'] for r in tasks})==10
            and Counter(r['domain'] for r in tasks)=={d:2 for d in ('code','extraction','arithmetic','copying','topic_changes')},'Task grid mismatch')
    rows=[json.loads(line) for line in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_order=[];models=[];domains=[]
    for key in ('qwen','opt'):
        spec=cfg['models'][key];parent=repository/spec['parent']
        for name,value in spec['parent_hashes'].items():require(digest(parent/name)==value,'Parent checksum mismatch')
        parent_manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
        checkpoint=Path(snapshot_download(spec['model'],revision=spec['revision'],local_files_only=True))
        catalog={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
        require(catalog==parent_manifest['checkpoint_files'],'Tokenizer/checkpoint catalog changed')
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        vocabulary_size=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))['vocab_size']
        group=[r for r in rows if r['model']==key];header=group[0]
        require(header['kind']=='model_manifest' and header['checkpoint_files']==catalog
                and header['shapes']==spec['shapes'] and header['attention_implementation']=='sdpa'
                and header['vocabulary_size']==vocabulary_size,'Model manifest mismatch')
        require(header['parameter_bytes']==parent_manifest['model_parameter_bytes']
                and header['restoration_bytes']==parent_manifest['diagnostic_layout_backup_bytes' if key=='opt' else 'diagnostic_backup_bytes'],'Model memory receipt mismatch')
        checked_file(root,f'{key}-token-ids.json',header['token_ids_sha256']);saved_ids=read(f'{key}-token-ids.json')
        inputs={task['id']:prepare_ids(tokenizer,task) for task in tasks}
        require(saved_ids=={name:{'prompt':p.tolist(),'teacher_forcing':v.tolist()} for name,(p,v) in inputs.items()},'Frozen tokenization mismatch')
        expected_order.append(('model_manifest',key));dense={};dense_rows={};generated={};native_valid=True
        for task in tasks:
            name=task['id'];prompt,ids=inputs[name]
            row=next(r for r in group if r['kind']=='dense_task' and r['document']==name)
            logits=load_file(checked_file(root,row['reference_file'],row['reference_sha256']))['logits']
            require(logits.dtype==torch.float32 and logits.shape==(1,ids.shape[1],vocabulary_size),'Dense logits shape mismatch')
            require(row['prompt_tokens']==prompt.shape[1] and row['answer_tokens']==ids.shape[1]-prompt.shape[1],'Answer boundary mismatch')
            require(row['full_metrics']==quality_metrics(logits,logits,ids)
                    and row['answer_metrics']==quality_metrics(logits[:,prompt.shape[1]-1:],logits[:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:])
                    and row['numerical_error']==(None if torch.isfinite(logits).all() else 'nonfinite_logits'),'Dense metric mismatch')
            generated[name]=check_generation(row,prompt,tokenizer,task['answer'],vocabulary_size);dense[name]=logits;dense_rows[name]=row
            native_valid &= generation_valid(root,row['generation'],prompt.shape[1]+31,vocabulary_size) and bool(torch.isfinite(logits).all())
            expected_order.append(('dense_task',key,name))
        hidden,neurons=spec['shapes'][0];layers=len(spec['shapes']);gate=native_valid
        for index in range(layers):
            row=next(r for r in group if r['kind']=='group_reconstruction' and r['layer']==index)
            pair=load_file(checked_file(root,row['tensors']['file'],row['tensors']['sha256']))
            require(set(pair)=={'reference','candidate'} and pair['reference'].numel()==2*hidden
                    and pair['reference'].shape==pair['candidate'].shape,'Local reconstruction tensor dimensions mismatch')
            error,failure=reconstruction_metric(pair['reference'],pair['candidate'])
            require(row['relative_l2']==error and row['numerical_error']==failure
                    and row['passed']==(failure is None and error<=.01),'Local reconstruction gate mismatch');gate &= row['passed']
            expected_order.append(('group_reconstruction',key,index))
        for task in tasks:
            name=task['id'];prompt,ids=inputs[name]
            row=next(r for r in group if r['kind']=='layout_correctness' and r['document']==name)
            candidate=load_file(checked_file(root,row['logits']['file'],row['logits']['sha256']))['logits']
            require(candidate.shape==dense[name].shape,'Layout logits shape mismatch')
            measured=quality_metrics(dense[name],candidate,ids);finite=measured is not None
            metrics=measured if finite else {}
            require(row['numerical_error']==(None if finite else 'nonfinite_logits')
                    and (all(row[k]==v for k,v in metrics.items()) if finite else not {'predicted_tokens','dense_nll','relative_perplexity','candidate_nll','mean_kl_dense_to_candidate','top1_agreement','logit_relative_l2','logit_max_abs_error'}.intersection(row)),'Layout metric mismatch')
            generated_tokens(row['token_ids'],prompt,vocabulary_size)
            gen_valid=generation_valid(root,row['generation'],prompt.shape[1]+31,vocabulary_size)
            equal=row['token_ids']==dense_rows[name]['token_ids'];passed=finite and numerical_pass(metrics) and equal and gen_valid
            require(row['generated_equal']==equal and row['passed']==passed,'Layout gate mismatch');gate &= passed
            expected_order.append(('layout_correctness',key,name))
        conditions=[];masked_generation_valid=True;masked_teacher_valid=True
        for mode in ('static','hindsight') if gate else ():
            selected=[]
            for task in tasks:
                name=task['id'];prompt,ids=inputs[name]
                row=next(r for r in group if r['kind']=='masked_task' and r['mode']==mode and r['document']==name)
                candidate=load_file(checked_file(root,row['logits']['file'],row['logits']['sha256']))['logits']
                masked_teacher_valid &= bool(torch.isfinite(candidate).all())
                require(row['full_metrics']==quality_metrics(dense[name],candidate,ids)
                    and row['answer_metrics']==quality_metrics(dense[name][:,prompt.shape[1]-1:],candidate[:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:])
                    and row['numerical_error']==(None if torch.isfinite(candidate).all() else 'nonfinite_logits'),'Masked quality metric mismatch')
                tokens=check_generation(row,prompt,tokenizer,task['answer'],vocabulary_size)
                masked_generation_valid &= generation_valid(root,row['generation'],prompt.shape[1]+31,vocabulary_size)
                require(row['generated_token_agreement']==float(np.equal(tokens,generated[name]).mean()),'Generation agreement mismatch')
                for receipt,steps,prefix in ((row['trace'],ids.shape[1],''),(row['generation_trace'],prompt.shape[1]+31,'generation_')):
                    count=checked_masks(root,receipt,steps,layers,neurons//8,mode)
                    full,chosen=ffn_volume(key,layers,hidden,neurons,steps,count)
                    require(row[prefix+'full_ffn_bytes']==full and row[prefix+'selected_ffn_bytes']==chosen,'FFN byte accounting mismatch')
                selected.append(row);expected_order.append(('masked_task',key,mode,name))
            value={'model':key,'mode':mode,'tasks':10,'exact_targets':sum(r['exact_target_match'] for r in selected),
                'dense_exact_targets':sum(r['exact_target_match'] for r in dense_rows.values()),
                'answer_metrics':quality_aggregate([r['answer_metrics'] for r in selected]),'full_metrics':quality_aggregate([r['full_metrics'] for r in selected]),
                'selected_ffn_fraction':sum(r['selected_ffn_bytes'] for r in selected)/sum(r['full_ffn_bytes'] for r in selected)}
            require(next(r for r in group if r['kind']=='aggregate' and r['mode']==mode)=={'kind':'aggregate',**value},'Balanced aggregate mismatch')
            conditions.append(value)
            for domain in dict.fromkeys(task['domain'] for task in tasks):
                names={r['id'] for r in tasks if r['domain']==domain};part=[r for r in selected if r['document'] in names]
                domains.append({'model':key,'mode':mode,'domain':domain,'tasks':len(part),
                    'exact_targets':sum(r['exact_target_match'] for r in part),'dense_exact_targets':sum(dense_rows[n]['exact_target_match'] for n in names),
                    'answer_metrics':quality_aggregate([r['answer_metrics'] for r in part])})
        expected_order.extend(('aggregate',key,mode) for mode in ('static','hindsight') if gate)
        status=('completed' if masked_generation_valid else 'masked_generation_failed') if gate else 'numerical_gate_failed'
        if gate and not masked_teacher_valid:status='masked_numerical_failed'
        models.append({'model':key,'status':status,'aggregates':conditions,
            'dense_exact_targets':sum(r['exact_target_match'] for r in dense_rows.values()),'tasks':10})
    def row_key(row):
        kind=row['kind'];key=row['model']
        if kind=='model_manifest':return kind,key
        if kind=='group_reconstruction':return kind,key,row['layer']
        if kind=='aggregate':return kind,key,row['mode']
        if kind=='masked_task':return kind,key,row['mode'],row['document']
        return kind,key,row['document']
    require([row_key(r) for r in rows]==expected_order,'Balanced ledger order/grid mismatch')
    require(summary['models']==models,'Balanced model summary mismatch')
    return {'status':'validated','models':models,'domains':domains,'runtime_nomination':None,
        'row_counts':dict(Counter(r['kind'] for r in rows)),
        'input_hashes':{name:digest(root/name) for name in ('config.json','manifest.json','corpus.jsonl','results.jsonl','summary.json')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'balanced_analysis.py')
    try:write_json(output/'summary.json',analyze(args.run))
    except (Exception,KeyboardInterrupt) as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':main()

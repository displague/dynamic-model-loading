"""Recompute causal development summaries from immutable ledgers and applied masks."""

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch
from safetensors.torch import load, load_file

from .cache import dense_static_frontier, group_arrays, replay
from .experiment import digest, load_corpus, write_json
from .metrics import aggregate, compare_logits


def require(ok, message):
    if not ok:raise ValueError(message)


def checked_trace(root, row, steps, layers, groups, keep):
    relative=Path(row['trace_file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid trace path')
    path=root/relative
    require(digest(path)==row['trace_sha256'],'Trace digest mismatch')
    with np.load(path,allow_pickle=False) as file:
        require(set(file.files)=={'bits','shape','audits'},'Trace fields mismatch')
        shape=file['shape'].tolist();bits=file['bits'];audit=file['audits']
    require(shape==[steps,layers,groups] and bits.dtype==np.uint8
            and bits.shape==(steps,layers,math.ceil(groups/8)), 'Trace shape mismatch')
    unpacked=np.unpackbits(bits,axis=-1,bitorder='little')
    require(not unpacked[:,:,groups:].any(),'Nonzero trace padding')
    mask=unpacked[:,:,:groups].astype(bool)
    require(np.all(mask.sum(-1)==math.ceil(keep*groups)), 'Mask retention mismatch')
    require(audit.shape==(steps,layers,3) and np.isfinite(audit).all()
            and (audit>=0).all() and (audit[:,:,1:]<=1+1e-6).all(), 'Invalid independent audits')
    return mask,audit


def metrics_valid(row, reference, token_ids, reference_side='dense_nll'):
    expected=compare_logits(reference,reference,token_ids)
    require(row['predicted_tokens']==expected['predicted_tokens'] and row[reference_side]==expected['dense_nll'],
            'Reference NLL/denominator mismatch')
    for key in expected:
        require(isinstance(row.get(key),(int,float)) and math.isfinite(row[key]) and row[key]>=0,
                'Invalid quality metric: '+key)
    require(row['top1_agreement']<=1 and row['relative_perplexity']>0 and
            math.isclose(math.log(row['relative_perplexity']),row['candidate_nll']-row['dense_nll'],abs_tol=1e-12,rel_tol=1e-10),
            'Quality metric consistency failure')


def analyze(root, parent):
    root,parent=Path(root),Path(parent)
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    cfg,manifest,started,summary=(read(name) for name in ('config.json','manifest.json','started.json','summary.json'))
    require(summary['status'] in ('causal_control_completed','cpu_fixture_completed') and summary['correctness_passed'] is True,
            'Incomplete causal run')
    require(not (root/'failure.json').exists(), 'Failure receipt present')
    require(cfg['modes']==['static','recency','ema','learned','hindsight'] and
            (cfg['group_width'],cfg['keep'],cfg['budget_bytes'],cfg['ridge'],cfg['features'],cfg['alpha'])==(8,.9,2**31,.01,64,.1),
            'Changed causal operating point')
    require((cfg['screen_relative_ppl_max'],cfg['screen_warm_savings_min'],cfg['screen_cost_ratio_max'])==(1.01,.1,.1),
            'Changed screen')
    require(digest(root/'config.json')==manifest['config_sha256'] and digest(root/'protocol.md')==manifest['protocol_sha256'],
            'Config/protocol hash mismatch')
    require(set(started)=={'started_utc','environment','config_sha256','protocol_sha256','sources','parent_hashes'}
            and all(manifest[k]==v for k,v in started.items()), 'Startup mismatch')
    require({'causal.py','causal_study.py','relu.py','experiment.py','adapters.py','ffn.py','metrics.py','cache.py','packing.py','trace.py'}
            <= set(manifest['sources']), 'Missing source dependencies')
    for name,expected in manifest['sources'].items():require(digest(root/'source'/name)==expected,'Source mismatch: '+name)
    require(manifest['parent_hashes']==cfg['parent_hashes'],'Parent inventory mismatch')
    for name,expected in cfg['parent_hashes'].items():require(digest(parent/name)==expected,'Parent mismatch: '+name)
    parent_manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['checkpoint_files']=={k:v['sha256'] for k,v in parent_manifest['checkpoint']['files'].items()},
            'Checkpoint catalog differs from pinned parent')
    for name in ('corpus.jsonl','token-ids.json','layouts.json.gz','calibration-importance.safetensors.gz'):
        require(digest(root/name)==cfg['parent_hashes'][name],'Copied parent mismatch')
    env=manifest['environment']; device=torch.device(env['device']).type
    require(env['torch']==cfg['expected_torch'] and env['cpu_threads']==cfg['cpu_threads'] and
            env['tf32_matmul'] is False and env['tf32_cudnn'] is False and manifest['attention_implementation']=='sdpa',
            'Execution policy mismatch')
    require((device=='cuda')==(summary['status']=='causal_control_completed'), 'CPU fixture cannot claim CUDA completion')
    torch.set_num_threads(cfg['cpu_threads'])
    corpus=load_corpus(root/'corpus.jsonl');ids=read('token-ids.json')
    cal=[r['id'] for r in corpus if r['split']=='calibration'];docs=[r['id'] for r in corpus if r['split']=='diagnostic']
    require([len(cal),len(docs)]==cfg['document_counts'] and set(ids)==set(cal+docs),'Document split mismatch')
    tokens={name:torch.tensor(value) for name,value in ids.items()}
    boundary=len(ids[docs[0]][0]);tokens['transition']=torch.cat((tokens[docs[0]],tokens[docs[1]]),1)
    episodes=docs+['transition'];layers=cfg['layers'];groups=math.ceil(cfg['neurons']/cfg['group_width'])
    layouts=json.loads(gzip.decompress((root/'layouts.json.gz').read_bytes()))
    means=load(gzip.decompress((root/'calibration-importance.safetensors.gz').read_bytes()))
    means=torch.stack([means[str(i)] for i in range(layers)]).numpy()
    sizes,importance=group_arrays(means,np.asarray(layouts['popularity']),cfg['group_width'],3*cfg['hidden']*4)
    require(manifest['groupable_ffn_bytes']==int(sizes.sum()) and manifest['diagnostic_backup_bytes']==int(sizes.sum()),
            'Model/backup byte accounting mismatch')
    rows=[json.loads(line) for line in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    timing_count=2*(cfg['timing_warmups']+cfg['timing_repetitions']) if device=='cuda' else 0
    counts={'dense_reference':len(episodes),'prefill_incremental_drift':len(docs),'dense_generation':cfg['generation_documents'],
            'group_reconstruction':layers,'layout_correctness':len(episodes),'calibration_fit':1,
            'selector_storage':5,'selector_document':5*len(episodes),'transition_after_boundary':5,
            'selector_generation':5*cfg['generation_documents'],'selector_aggregate':5}
    if timing_count:counts['selector_timing']=4*timing_count
    require(Counter(r['kind'] for r in rows)==counts,'Incomplete raw grid')
    by_kind={kind:[r for r in rows if r['kind']==kind] for kind in counts}
    expected_order=[]
    for name in episodes:
        expected_order+=['dense_reference']+(['prefill_incremental_drift'] if name!='transition' else [])
    expected_order+=['dense_generation']*cfg['generation_documents']+['group_reconstruction']*layers+['layout_correctness']*len(episodes)+['calibration_fit']
    for mode in cfg['modes']:
        expected_order+=['selector_storage']+['selector_document']*len(episodes)+['transition_after_boundary']
        if mode!='hindsight':expected_order+=['selector_timing']*timing_count
        expected_order+=['selector_generation']*cfg['generation_documents']+['selector_aggregate']
    require([r['kind'] for r in rows]==expected_order,'Gate/calibration/probe ordering mismatch')
    references={}
    require([r['document'] for r in by_kind['dense_reference']]==episodes,'Reference document order')
    for index,row in enumerate(by_kind['dense_reference']):
        path=root/'reference_logits'/f'{index:03d}.safetensors'
        require(digest(path)==row['logits_sha256'] and row['input_tokens']==tokens[row['document']].shape[1], 'Dense reference mismatch')
        references[row['document']]=path
    reference=lambda name:load_file(references[name])['logits']
    require([r['document'] for r in by_kind['prefill_incremental_drift']]==docs,'Prefill drift document grid mismatch')
    for row in by_kind['prefill_incremental_drift']:
        metrics_valid(row,reference(row['document']),tokens[row['document']],reference_side='candidate_nll')
    require([r['layer'] for r in by_kind['group_reconstruction']]==list(range(layers)) and
            all(r['passed'] is True and 0<=r['relative_l2']<=.01 for r in by_kind['group_reconstruction']), 'Local gate failure')
    require([r['document'] for r in by_kind['layout_correctness']]==episodes,'Layout gate grid mismatch')
    for row in by_kind['layout_correctness']:
        metrics_valid(row,reference(row['document']),tokens[row['document']])
        require(row['passed'] is True and row['logit_relative_l2']<=.01 and row['mean_kl_dense_to_candidate']<=.001,'Full gate failure')
    fit=by_kind['calibration_fit'][0]
    require(fit['documents']==cal and fit['observations']==[sum(tokens[d].shape[1] for d in cal)]*layers,'Calibration label split mismatch')
    for name,key in (('calibration-statistics.safetensors','statistics_sha256'),('learned.safetensors','learned_sha256')):
        require(digest(root/name)==fit[key],'Fitted artifact digest mismatch')
    learned=load_file(root/'learned.safetensors')
    dense_generations=by_kind['dense_generation']
    require([r['document'] for r in dense_generations]==docs[:cfg['generation_documents']], 'Dense generation grid')
    for row in dense_generations:
        require(row['token_ids'][:cfg['prompt_tokens']]==ids[row['document']][0][:cfg['prompt_tokens']] and
                len(row['token_ids'])==cfg['prompt_tokens']+cfg['generation_tokens'],'Dense generation length/prefix')
    baseline=dense_static_frontier(means,layouts,[8,32,128],[cfg['budget_bytes']],3*cfg['hidden']*4,[tokens[d].shape[1] for d in docs])
    baseline=baseline[(cfg['budget_bytes'],'warm')]
    require([(r['mode'],r['document']) for r in by_kind['selector_document']]==[(m,d) for m in cfg['modes'] for d in episodes],
            'Selector document grid')
    recomputed=[]
    for mode in cfg['modes']:
        storage=next(r for r in by_kind['selector_storage'] if r['mode']==mode)
        tensors=[]
        for i in range(layers) if mode!='hindsight' else []:
            shapes={'prior':([groups],'torch.float32',4)}
            if mode=='static':shapes['fixed']=([groups],'torch.bool',1)
            if mode in ('recency','ema'):shapes.update(norms=([cfg['neurons']],'torch.float32',4),state=([groups],'torch.float32',4))
            if mode=='learned':
                for k in ('projection','mean','scale','weight','intercept'):
                    t=learned[f'{i}.{k}'];shapes['learned_'+k]=(list(t.shape),str(t.dtype),t.element_size())
            tensors.append({k:{'shape':s,'dtype':dtype,'bytes':math.prod(s)*size} for k,(s,dtype,size) in shapes.items()})
        charged=sum(v['bytes'] for layer in tensors for v in layer.values())
        require(storage['tensors']==tensors and storage['persistent_bytes']==charged,'Selector storage mismatch')
        selected=[r for r in by_kind['selector_document'] if r['mode']==mode];measurements=[];all_audits=[];warm=0
        for row in selected:
            name=row['document'];value=reference(name)
            metrics_valid(row,value,tokens[name])
            mask,audit=checked_trace(root,row,tokens[name].shape[1],layers,groups,cfg['keep'])
            expected=replay(mask,sizes,importance,cfg['budget_bytes']-charged,'static_equal_layer')
            require(row['replay']==expected and row['selector_bytes']==charged,'Charged cache replay mismatch')
            if name!='transition':
                measurements.append(row);all_audits.append(audit)
                warm+=next(r['total_bytes'] for r in expected if r['state']=='warm')
            else:
                post=next(r for r in by_kind['transition_after_boundary'] if r['mode']==mode)
                require(post['boundary']==boundary,'Transition boundary mismatch')
                metrics_valid(post,value[:,boundary-1:],tokens[name][:,boundary-1:])
                require(post['replay']==replay(mask[boundary:],sizes,importance,cfg['budget_bytes']-charged,'static_equal_layer'),
                        'Post-transition replay mismatch')
        cost=medians=None
        if mode!='hindsight' and device=='cuda':
            times=[r for r in by_kind['selector_timing'] if r['mode']==mode]
            require([(r['workload'],r['repetition']) for r in times]==
                    [(w,i) for w in ('dense','selector') for i in range(cfg['timing_warmups']+cfg['timing_repetitions'])], 'Timing grid mismatch')
            require(all(r['warmup']==(r['repetition']<cfg['timing_warmups']) and
                        all(math.isfinite(r[k]) and r[k]>0 for k in ('cuda_ms','wall_ms')) for r in times), 'Invalid timing values')
            medians={w:{k:statistics.median(r[k] for r in times if r['workload']==w and not r['warmup'])
                        for k in ('cuda_ms','wall_ms')} for w in ('dense','selector')}
            cost={k:medians['selector'][k]/medians['dense'][k] for k in ('cuda_ms','wall_ms')}
        gens=[r for r in by_kind['selector_generation'] if r['mode']==mode]
        require([r['document'] for r in gens]==docs[:cfg['generation_documents']],'Generated condition grid')
        for index,row in enumerate(gens):
            require(row['token_ids'][:cfg['prompt_tokens']]==ids[row['document']][0][:cfg['prompt_tokens']] and
                    len(row['token_ids'])==cfg['prompt_tokens']+cfg['generation_tokens'],'Generated length/prefix')
            checked_trace(root,row,len(row['token_ids'])-1,layers,groups,cfg['keep'])
            file=root/'generation'/f'{mode}-{index:03d}.json'
            require(digest(file)==row['artifact_sha256'],'Generated artifact hash mismatch')
            require(json.loads(file.read_text(encoding='utf-8'))=={k:v for k,v in row.items() if k not in ('kind','artifact_sha256')},
                    'Generated artifact differs from ledger')
            match=np.equal(row['token_ids'][cfg['prompt_tokens']:],dense_generations[index]['token_ids'][cfg['prompt_tokens']:]).astype(np.float32).mean()
            require(float(match)==row['generated_token_agreement_with_dense'],'Generated agreement mismatch')
        quality=aggregate(measurements);audits=np.concatenate([a.reshape(-1,3) for a in all_audits],0)
        saving=1-warm/baseline['total_bytes'] if baseline['total_bytes'] else None
        result={'mode':mode,**quality,'selector_bytes':charged,'warm_bytes':warm,'warm_saving':saving,'dense_baseline':baseline,
                'cost_ratios':cost,'timing_medians':medians,'audit_mean':audits.mean(0).tolist(),
                'audit_p95':np.quantile(audits,.95,axis=0).tolist(),
                'audit_local_error_above_point1_fraction':float((audits[:,0]>.1).mean()),
                'passes_screen':mode!='hindsight' and cost is not None and saving is not None
                    and quality['relative_perplexity']<=1.01 and saving>=.1 and max(cost.values())<=.1}
        require(next(r for r in by_kind['selector_aggregate'] if r['mode']==mode)=={'kind':'selector_aggregate',**result},
                'Aggregate mismatch')
        recomputed.append(result)
    require(recomputed==summary['selectors'],'Summary mismatch')
    passing=sorted((r for r in recomputed if r['passes_screen']),key=lambda r:(r['warm_bytes'],r['mean_kl_dense_to_candidate'],r['mode']))
    require(summary['nominated_selector']==(passing[0]['mode'] if passing else None),'Nomination rule mismatch')
    return {'status':'validated','input_hashes':{name:digest(root/name) for name in ('manifest.json','config.json','summary.json','results.jsonl')},
            'row_counts':counts,'selectors':recomputed,'nominated_selector':summary['nominated_selector'],
            'limitations':'Receipt, applied-mask replay and aggregate validation; no new pretrained model scoring or physical transfer claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','parent','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'causal_analysis.py')
    try:write_json(output/'summary.json',analyze(args.run,args.parent))
    except (Exception,KeyboardInterrupt) as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':main()

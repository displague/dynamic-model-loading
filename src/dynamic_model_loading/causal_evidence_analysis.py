"""Verify frozen inputs, fitted predictors, permitted state, masks and quality receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from . import causal_evidence as ce
from .balanced_control import quality_metrics
from .causal_study import numerical_pass
from .causal_evidence_cost import action_cost,rounded_cost,batch_counts,TIMING_WORKLOADS
from .causal_evidence_study import load_inputs,model_bytes,policy_summary,read
from .dense_interface import aligned_metrics,prepare,score,decoding
from .experiment import digest,write_json
from .provenance import frozen_environment,owning_repository,verify_snapshot


def same(actual,expected,label):
    if actual!=expected:raise ValueError('Receipt mismatch: '+label)


def close(actual,expected,label):
    # CPU/GPU matrix reductions can differ. This checks numerical inference;
    # exact ranking is independently checked against retained GPU score vectors.
    if actual.shape!=expected.shape or not torch.isfinite(actual).all() or not torch.allclose(actual,expected,rtol=2e-4,atol=2e-4):
        raise ValueError('Tensor replay mismatch: '+label)


def generation_alignment(reference,own,aligned,generated,reference_ids):
    if generated==reference_ids and not torch.equal(own,aligned):
        raise ValueError('Contradictory own and aligned logits on identical generated prefixes')
    return aligned_metrics(reference,aligned)


def verify_trace(data,parent,models,mode,extra,visits):
    expected_shapes={'x':(visits,28,1536),'input_projection':(visits,28,8),
        'observed':(visits,28,1120),'initial':(visits,28,1120),'additions':(visits,28,1120),
        'initial_scores':(visits,28,1120),'acquisition_scores':(visits,28,1120),
        'first_norm':(visits,28),'audit':(visits,28,2),'fallback':(visits,28),
        'detection':(visits,28,1),'base_features':(visits,28,33),'repair_features':(visits,28,43)}
    same(set(data),set(expected_shapes),'trace fields')
    for key,shape in expected_shapes.items():
        same(tuple(data[key].shape),shape,key)
        if not torch.isfinite(data[key]).all():raise ValueError('Nonfinite trace')
    for layer in range(28):
        prior=torch.tensor(parent['importance'][layer],dtype=torch.float32)
        hot=torch.tensor(parent['hot'][layer])
        c=ce.Controller(prior,hot,ce.projections(1536,1120,1729+layer,'cpu'),
            None if models is None else models[layer],mode,extra)
        for step in range(visits):
            initial=data['initial'][step,layer];added=data['additions'][step,layer];final=initial|added
            observed=data['observed'][step,layer];gpu_scores=data['initial_scores'][step,layer]
            if initial.dtype!=torch.bool or added.dtype!=torch.bool or bool((initial&added).any()):raise ValueError('Overlapping acquisitions')
            if bool((observed[~final]!=0).any()) or bool((observed<0).any()):raise ValueError('Omitted observation leaked into state')
            if not bool(initial[hot].all()):raise ValueError('Resident core missing')
            close(data['input_projection'][step,layer],data['x'][step,layer]@c.projection['input_projection'],'input projection')
            c.begin_projected(data['input_projection'][step,layer])
            close(data['base_features'][step,layer],c.base,'base state features')
            close(gpu_scores,c.initial_scores,'initial predictor')
            count=1008+(extra if mode in ('one_shot','input_only') else 0)
            expected=hot|ce.rank(gpu_scores,~hot,count-int(hot.sum()))
            same(initial.tolist(),expected.tolist(),'initial exact ranking')
            # GPU decisions establish executed state; do not substitute CPU ties.
            c.first=initial.clone()
            permitted=torch.where(initial,observed,0.)
            features=c.evidence(permitted,data['first_norm'][step,layer])
            close(data['repair_features'][step,layer],features,'partial evidence')
            c.acquire(permitted,data['first_norm'][step,layer])
            close(data['detection'][step,layer],c.detection,'detector')
            fallback=mode=='fallback' and float(data['detection'][step,layer,0])>.05
            same(bool(data['fallback'][step,layer]),fallback,'fallback rule')
            if mode in ('initial','one_shot','input_only'):
                expected_added=torch.zeros_like(initial)
            elif mode=='full' or fallback:
                expected_added=~initial
            else:
                # The retained GPU detector decides the branch. Recompute scores
                # independently: an accepted CPU/GPU detector difference across
                # 0.05 must not substitute the CPU branch's zero score buffer.
                predicted=c.prior_log if mode=='predetermined' or models is None else ce.predict(features,c.models['repair'])
                close(data['acquisition_scores'][step,layer],predicted,'acquisition predictor')
                expected_added=ce.rank(data['acquisition_scores'][step,layer],~initial,extra)
            same(added.tolist(),expected_added.tolist(),'addition exact ranking')
            c.observe(observed,final)


def analyze(run_path,cost_path,output_path):
    torch.set_num_threads(4)
    run,cost=Path(run_path),Path(cost_path);root=owning_repository(__file__)
    cfg=read(run/'config.json');manifest=read(run/'manifest.json')
    verify_snapshot(run,manifest,'configs/causal-evidence.json',cfg['protocol'],__file__)
    frozen_environment(manifest['environment']);parent=load_inputs(root,cfg)
    same(read(run/'inputs.json'),{'calibration':parent['calibration'],'development':parent['development'],
        'tokens':{k:v.tolist() for k,v in parent['ids'].items()},'tasks':parent['tasks'],
        'hot':parent['hot'].tolist(),'dense_baseline':parent['baseline']},'recorded input inventory')
    rows=[json.loads(line) for line in (run/'results.jsonl').read_text().splitlines()]
    same(Counter(r['kind'] for r in rows),dict(native_wiki=4,native_task=20,calibration=8,fit=2,
        completion_control=4,task_completion_control=20,wiki=40,task=60),'full row grid')
    for kind,names in [('native_wiki',parent['development']),('completion_control',parent['development']),
        ('native_task',[t['id'] for t in parent['tasks']]),('task_completion_control',[t['id'] for t in parent['tasks']])]:
        same(sorted(r['document'] for r in rows if r['kind']==kind),sorted(names),kind+' identities')
    def tensors(receipt):
        path=run/receipt['file']
        same(digest(path),receipt['sha256'],'tensor checksum')
        return load_file(path)
    for kind,names,conditions in [
        ('wiki',parent['development'],ce.CONDITIONS),
        ('task',[t['id'] for t in parent['tasks']],(('one_shot',28),('predetermined',28),('partial',28)))]:
        selected=[r for r in rows if r['kind']==kind]
        same({(r['document'],r['mode'],r['extra']) for r in selected},
             {(name,mode,extra) for name in names for mode,extra in conditions},kind+' identities')
    checkpoint=Path(snapshot_download(cfg['model']['model'],revision=cfg['model']['revision'],local_files_only=True))
    catalog={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
    same(catalog,read(run/'model.json')['checkpoint_files'],'checkpoint receipt')
    same(catalog,read(root/cfg['model']['parent']/'manifest.json')['checkpoint_files'],'pinned checkpoint')
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    config=decoding(checkpoint);same(config.to_dict(),read(run/'model.json')['generation_config'],'decoding')
    eos=config.eos_token_id if isinstance(config.eos_token_id,list) else [config.eos_token_id]
    previous=None
    for pass_index in (1,2):
        selected=[r for r in rows if r['kind']=='calibration' and r['pass_index']==pass_index]
        same([r['document'] for r in selected],parent['calibration'],'calibration document order')
        collected=[]
        for row in selected:
            data=tensors(row['examples']);trace=tensors(row['trace'])
            verify_trace(trace,parent,previous,'predetermined' if pass_index==1 else 'partial',28,128)
            close(data['base'],trace['base_features'],'calibration features')
            close(data['repair'],trace['repair_features'],'calibration partial features')
            close(data['error'],trace['audit'][...,:1],'detector labels')
            close(data['x'],trace['x'],'calibration input')
            collected.append(data)
        combined={key:torch.cat([v[key] for v in collected]) for key in collected[0]}
        fitted=ce.fit_models(combined);row=next(r for r in rows if r['kind']=='fit' and r['pass_index']==pass_index)
        stored=tensors(row['tensors'])
        for key,value in ce.flatten_models(fitted).items():close(stored[key],value,'fitted '+key)
        same(row['storage'],model_bytes(fitted,parent),'charged model storage')
        if row['storage']['total']>cfg['controller_reservation_bytes']:raise ValueError('Undercharged reservation')
        previous=ce.unflatten_models(stored,28)
    models=previous;native={}
    tasks={t['id']:t for t in parent['tasks']}
    for row in rows:
        kind=row['kind']
        if kind not in ('native_wiki','native_task','completion_control','task_completion_control','wiki','task'):continue
        name=row['document'];value=tensors(row['tensors']);is_task=name in tasks
        if is_task:
            task=tasks[name];prompt=prepare(tokenizer,task,'chat')
            answer=tokenizer(task['answer'],add_special_tokens=False,return_tensors='pt')['input_ids']
            ids=torch.cat((prompt,answer),1)
        else:ids=parent['ids'][name]
        if value['logits'].shape!=(1,ids.shape[1],152064):raise ValueError('Logit dimensions mismatch')
        if kind.startswith('native_'):
            native[name]=row
            if not torch.isfinite(value['logits']).all():raise ValueError('Invalid native reference')
            same(row['finite'],True,'native finite status')
        else:
            ref=tensors(native[name]['tensors'])['logits'];quality=quality_metrics(ref,value['logits'],ids)
            same(quality,row['quality'],'quality')
            mode,extra=('full',112) if 'control' in kind else (row['mode'],row['extra'])
            trace=tensors(row['trace']);verify_trace(trace,parent,models,mode,extra,ids.shape[1])
            if kind in ('wiki','task'):same(row['traffic'],ce.traffic(trace,parent['hot']),'teacher traffic')
            if kind in ('wiki','task'):same(row['finite'],quality is not None and (not is_task or all(row['generation']['calls'])),'finite status')
            if 'control' in kind:
                passed=quality is not None and numerical_pass(quality)
                if is_task:
                    ref_gen=tensors(native[name]['tensors'])['generation_logits']
                    gen_quality=generation_alignment(ref_gen,value['generation_logits'],value['aligned_generation_logits'],
                        row['token_ids'],native[name]['token_ids'])
                    if row['token_ids']==native[name]['token_ids']:
                        same(row['aligned_generation_trace'],row['generation_trace'],'identical-prefix trace receipt')
                        same(row['aligned_generation'],row['generation'],'identical-prefix finite receipt')
                    same(row['generation_quality'],gen_quality,'completion generation logits')
                    passed=(passed and row['token_ids']==native[name]['token_ids'] and all(row['generation']['calls'])
                        and all(row['aligned_generation']['calls']) and gen_quality is not None
                        and gen_quality['relative_l2']<=.01 and gen_quality['mean_kl']<=.001)
                same(row['passed'],passed,'full-completion gate')
                if not passed:raise ValueError('Full-completion fidelity gate failed')
            if kind=='task':same(row['answer_quality'],quality_metrics(ref[:,prompt.shape[1]-1:],value['logits'][:,prompt.shape[1]-1:],ids[:,prompt.shape[1]-1:]),'answer quality')
        if is_task:
            generated=row['token_ids'];gen=value['generation_logits']
            if gen.shape!=(len(generated),152064) or gen.dtype!=torch.float32:raise ValueError('Generated-logit shape mismatch')
            if not 1<=len(generated)<=64 or any(t in eos for t in generated[:-1]) or (len(generated)<64 and generated[-1] not in eos):raise ValueError('Invalid EOS stop')
            same(generated,gen.argmax(-1).tolist(),'greedy generation')
            generation=row['generation'];same(len(generation['calls']),prompt.shape[1]+len(generated)-1,'generation calls')
            if not all(generation['calls']) or generation['failures'] or not torch.isfinite(gen).all():raise ValueError('Nonfinite generation')
            if kind!='task_completion_control':
                text=tokenizer.decode(generated,skip_special_tokens=True);same(row['text'],text,'text')
                for key,expected in score(text,task).items():same(row[key],expected,'score '+key)
                same(row['domain'],task['domain'],'task domain')
                if kind=='native_task':
                    same(row['prompt_ids'],prompt[0].tolist(),'native prompt')
                    same(row['teacher_ids'],ids[0].tolist(),'native teacher tokens')
            else:same(generated,native[name]['token_ids'],'full completion generation')
            if not kind.startswith('native_'):
                trace=tensors(row['generation_trace']);verify_trace(trace,parent,models,mode,extra,prompt.shape[1]+len(generated)-1)
                if kind=='task':same(row['generation_traffic'],ce.traffic(trace,parent['hot']),'generation traffic')
                else:
                    visits=prompt.shape[1]+len(native[name]['token_ids'])-1
                    same(len(row['aligned_generation']['calls']),visits,'aligned generation calls')
                    if row['aligned_generation_trace']!=row['generation_trace']:
                        verify_trace(tensors(row['aligned_generation_trace']),parent,models,mode,extra,visits)
    conditions=policy_summary(rows,parent['baseline']['total_bytes'])
    same(read(run/'summary.json')['conditions'],conditions,'quality summary')
    cost_manifest=read(cost/'manifest.json')
    same(cost_manifest['source_commit'],manifest['source_commit'],'timing source')
    same(cost_manifest['results_sha256'],digest(run/'results.jsonl'),'timed result identity');frozen_environment(cost_manifest['environment'])
    samples=[json.loads(s) for s in (cost/'results.jsonl').read_text().splitlines()]
    same(len(samples),len(ce.CONDITIONS)*len(TIMING_WORKLOADS)*13,'timing rows')
    same({(r['mode'],r['extra'],r['workload'],r['repetition']) for r in samples},
        {(m,e,w,i) for m,e in ce.CONDITIONS for w in TIMING_WORKLOADS for i in range(13)},'timing grid')
    for r in samples:
        if r['warmup']!=(r['repetition']<3) or r['visits']!=896 or not all(np.isfinite(r[k]) and r[k]>0 for k in ('wall_ms','cuda_ms')):raise ValueError('Invalid timing sample')
    timing=[]
    for mode,extra in ce.CONDITIONS:
        for workload in TIMING_WORKLOADS:
            group=[r for r in samples if (r['mode'],r['extra'],r['workload'])==(mode,extra,workload) and not r['warmup']]
            timing.append(dict(mode=mode,extra=extra,workload=workload,
                **{key:statistics.median(r[key] for r in group) for key in ('wall_ms','cuda_ms')},visits=896))
    same(timing,read(cost/'summary.json')['aggregates'],'timing aggregates')
    from .hardware_cost import synthetic_inputs,input_receipt
    pool,order,x=synthetic_inputs()
    same(input_receipt(pool,order,x),read(cost/'batch-inputs.json'),'exact-batch input recipe')
    del pool,order,x
    batch_rows=[json.loads(s) for s in (cost/'batch-results.jsonl').read_text().splitlines()]
    counts=batch_counts(parent,cfg);same(read(cost/'batch-summary.json')['groups'],counts,'exact sizes')
    same(Counter(r['kind'] for r in batch_rows),{'integrity':2*len(counts),'timing':26*len(counts)},'exact-batch grid')
    hardware={'aggregates':[]}
    for count in counts:
        for workload in ('resident_ffn','gather_transfer_pack_ffn'):
            checks=[r for r in batch_rows if (r['kind'],r['groups'],r['workload'])==('integrity',count,workload)]
            if len(checks)!=1 or not checks[0]['passed'] or not 0<=checks[0]['relative_l2']<=.01:raise ValueError('Primitive integrity failure')
            group=[r for r in batch_rows if (r['kind'],r['groups'],r['workload'])==('timing',count,workload)]
            same(sorted(r['repetition'] for r in group),list(range(13)),'batch repetition grid')
            if any(r['warmup']!=(r['repetition']<3) or not all(np.isfinite(r[k]) and r[k]>0 for k in ('wall_ms','cuda_ms')) for r in group):raise ValueError('Invalid batch timing')
            hardware['aggregates'].append(dict(groups=count,workload=workload,medians={key:statistics.median(r[key] for r in group if not r['warmup']) for key in ('wall_ms','cuda_ms')}))
    same(hardware['aggregates'],read(cost/'batch-summary.json')['aggregates'],'batch medians')
    # Dense static equal-layer core at the full allowance, including its incoming
    # slot. All groups are equal-sized; its warm bytes match the dense frontier.
    from .cache import cache_shape,hot_mask
    dense_hot=hot_mask(parent['importance'],parent['sizes'],cache_shape(parent['sizes'],cfg['budget_bytes']-cfg['acquisition_workspace_bytes'])['retained_slots'],'static_equal_layer')
    dense_ms=sum(rounded_cost(int(v),'resident_ffn',hardware)+rounded_cost(1120-int(v),'gather_transfer_pack_ffn',hardware) for v in dense_hot.sum(-1))*512
    dense_add_per_visit=min(r['wall_ms']/896 for r in timing if r['workload']=='output_additions' and r['mode'] in ('initial','one_shot','input_only'))
    dense_ms+=512*28*dense_add_per_visit
    for item in conditions:
        mode,extra=item['mode'],item['extra'];selected=[r for r in rows if r['kind']=='wiki' and (r['mode'],r['extra'])==(mode,extra)]
        timing_row=next(r for r in timing if (r['mode'],r['extra'],r['workload'])==(mode,extra,'controller'))
        additions_row=next(r for r in timing if (r['mode'],r['extra'],r['workload'])==(mode,extra,'output_additions'))
        costs=[action_cost(tensors(r['trace']),parent['hot'],hardware,timing_row['wall_ms']/896,additions_row['wall_ms']/896) for r in selected]
        item['serialized_cost_ms']={key:sum(r[key] for r in costs) for key in costs[0]}
        item['serialized_dense_ms']=dense_ms;item['serialized_headroom_ms']=dense_ms-item['serialized_cost_ms']['total_ms']
        item['gate_c_plausibility']=item['traffic_pass'] and item['serialized_headroom_ms']>0
    behavior=[]
    dense_success={r['document'] for r in rows if r['kind']=='native_task' and r['success']}
    for mode in ('one_shot','predetermined','partial'):
        selected=[r for r in rows if r['kind']=='task' and r['mode']==mode]
        lost=[r['document'] for r in selected if r['document'] in dense_success and not r['success']]
        behavior.append(dict(mode=mode,successes=sum(r['success'] for r in selected),dense_successes=len(dense_success),
            lost_dense_successes=lost,new_format_failures=[r['document'] for r in selected if r['document'] in dense_success and not r['format_compliant']]))
    ablations=[]
    for extra in (28,56):
        scores={m:next(c for c in conditions if c['mode']==m and c['extra']==extra) for m in ('one_shot','predetermined','partial')}
        aggregate_better=all(scores['partial']['quality']['relative_perplexity']<scores[m]['quality']['relative_perplexity'] for m in ('one_shot','predetermined'))
        individual_no_worse=all(next(r for r in rows if r['kind']=='wiki' and (r['document'],r['mode'],r['extra'])==(name,'partial',extra))['quality']['relative_perplexity']<=
            next(r for r in rows if r['kind']=='wiki' and (r['document'],r['mode'],r['extra'])==(name,m,extra))['quality']['relative_perplexity'] for name in parent['development'] for m in ('one_shot','predetermined'))
        ablations.append(dict(extra=extra,aggregate_better_than_both=aggregate_better,individual_no_worse=individual_no_worse,
            evidence_quality_advantage=aggregate_better and individual_no_worse))
    result=dict(status='verified',rows=len(rows),source_commit=manifest['source_commit'],conditions=conditions,behavior=behavior,
                ablations=ablations,timing=timing,runtime_nomination=None,
                exact_batch_costs=hardware['aggregates'],
                limitation='Serialized synthetic primitive sensitivity is not exposed runtime latency; broader held-out gates remain open.')
    output=Path(output_path);output.mkdir(parents=True,exist_ok=False);write_json(output/'summary.json',result)
    write_json(output/'verification.json',{'source_commit':manifest['source_commit'],'results_sha256':digest(run/'results.jsonl'),
        'timing_results_sha256':digest(cost/'results.jsonl'),'status':'verified'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','cost','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.run,args.cost,args.output),indent=2))


if __name__=='__main__':main()

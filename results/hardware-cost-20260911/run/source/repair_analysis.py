"""Validate complete repair receipts, reproduce rankings, and rebuild the frontier."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors.torch import load_file

from . import repair
from .cache import dense_static_frontier, replay
from .causal_analysis import metrics_valid, require
from .causal_study import numerical_pass
from .experiment import digest, write_json
from .metrics import aggregate
from .repair_study import load_parent, read, repair_hot, validate_config
from .provenance import frozen_environment, verify_snapshot


def safe_path(root, relative):
    path = Path(relative)
    require(not path.anchor and '..' not in path.parts, 'Invalid artifact path')
    return root/path


def unpack(file, key, shape):
    bits = file[key]
    require(bits.dtype==np.uint8 and bits.shape==(*shape[:-1],math.ceil(shape[-1]/8)), 'Invalid packed bits')
    value = np.unpackbits(bits,axis=-1,bitorder='little')
    require(not value[...,shape[-1]:].any(), 'Nonzero mask padding')
    return value[...,:shape[-1]].astype(bool)


def validate_repair_trace(path, initial, hot, schedule, limit):
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)=={'initial_bits','repair_bits','final_bits','shape','omitted_scores','audits'}, 'Repair trace fields mismatch')
        shape = tuple(data['shape'].tolist())
        require(shape==initial.shape, 'Repair trace shape mismatch')
        frozen,added,final = (unpack(data,key,shape) for key in ('initial_bits','repair_bits','final_bits'))
        scores,audits = data['omitted_scores'],data['audits']
    require(np.array_equal(frozen,initial) and not (added & initial).any() and np.array_equal(final,initial|added), 'Invalid additive partition')
    omitted_count = int((~initial[0,0]).sum())
    require(scores.shape==(*shape[:2],omitted_count) and scores.dtype==np.float32, 'Omitted-score shape/dtype mismatch')
    require(audits.shape==(*shape[:2],5) and np.isfinite(audits).all() and (audits>=0).all(), 'Invalid local repair audits')
    for token in range(shape[0]):
        for layer in range(shape[1]):
            expected = repair.expected_additions(initial[token,layer],hot[layer],scores[token,layer],schedule,limit)
            require(np.array_equal(expected,added[token,layer]), 'Privileged ranking mismatch')
    return added,final,audits


def analyze(root, parent_path, archive_path):
    root = Path(root)
    cfg = read(root/'config.json'); validate_config(cfg)
    parent = load_parent(parent_path,archive_path,cfg)
    started,manifest,summary = (read(root/name) for name in ('started.json','manifest.json','summary.json'))
    require(summary['status']=='repair_diagnostic_completed' and summary['correctness_passed'] is True and summary['runtime_nomination'] is None,
            'Incomplete diagnostic or invalid runtime nomination')
    require(not (root/'failure.json').exists(), 'Failure receipt present')
    require(all(manifest.get(k)==v for k,v in started.items()), 'Startup manifest mismatch')
    require(digest(root/'config.json')==started['config_sha256'] and digest(root/'protocol.md')==started['protocol_sha256'], 'Protocol/config mismatch')
    require(manifest['parent_index_sha256']==cfg['parent_index_sha256'], 'Parent identity mismatch')
    require(manifest['checkpoint_files']==parent['manifest']['checkpoint_files'], 'Checkpoint catalog mismatch')
    source_files = {p.relative_to(root/'source').as_posix() for p in (root/'source').rglob('*.py')}
    require(source_files==set(started['sources']) and {'repair.py','repair_study.py','causal.py','causal_study.py','cache.py','relu.py'}<=source_files,
            'Source inventory mismatch')
    for name,value in started['sources'].items():
        require(digest(safe_path(root/'source',name))==value, 'Source hash mismatch')
    verify_snapshot(root,started,'configs/refinement-feasibility.json',cfg['protocol'],__file__)
    env = manifest['environment']
    frozen_environment(env)
    require(torch.device(env['device']).type=='cuda' and env['torch']==cfg['expected_torch'] and env['cpu_threads']==cfg['cpu_threads']
        and env['tf32_matmul'] is False and env['tf32_cudnn'] is False and manifest['attention_implementation']=='sdpa', 'Execution policy mismatch')
    require(manifest['model_parameter_bytes']==parent['manifest']['model_parameter_bytes']
        and manifest['restoration_bytes']==int(parent['sizes'].sum())
        and manifest['frozen_mask_bytes_per_episode']==cfg['tokens']*cfg['layers']*cfg['groups']
        and manifest['repair_metadata_bytes']==cfg['repair_metadata_bytes']==cfg['layers']*(cfg['neurons']*4+cfg['groups']), 'Tensor accounting mismatch')
    require(read(root/'token-ids.json')=={k:v.tolist() for k,v in parent['tokens'].items()}, 'Sample IDs mismatch')
    torch.set_num_threads(cfg['cpu_threads'])
    rows = [json.loads(line) for line in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_order = [('dense_reference',name) for name in parent['docs']]
    expected_order += [('group_reconstruction',layer) for layer in range(cfg['layers'])]
    expected_order += [('layout_correctness',name) for name in parent['docs']]
    for mode in repair.MODES:
        for name in parent['docs']:
            expected_order += [(kind,mode,name) for kind in ('selector_reproduction','zero_repair_correctness','full_repair_correctness')]
    condition_keys = []
    for mode in repair.MODES:
        for schedule in repair.SCHEDULES:
            for limit in repair.LIMITS:
                condition = f'{mode}-{schedule}-{limit}'; condition_keys.append(condition)
                expected_order += [('repair_document',condition,name) for name in parent['docs']]
        for keep in repair.ONE_SHOT:
            condition = f'{mode}-one_shot-{keep:g}'; condition_keys.append(condition)
            expected_order += [('one_shot_document',condition,name) for name in parent['docs']]
    expected_order += [('aggregate',name) for name in condition_keys]
    def key(row):
        kind = row['kind']
        if kind=='group_reconstruction':return kind,row['layer']
        if kind in ('selector_reproduction','zero_repair_correctness','full_repair_correctness'):return kind,row['mode'],row['document']
        if kind in ('repair_document','one_shot_document'):return kind,row['condition'],row['document']
        if kind=='aggregate':return kind,row['condition']
        return kind,row.get('document')
    require([key(row) for row in rows]==expected_order, 'Incomplete or reordered diagnostic grid')
    native = {}
    for index,name in enumerate(parent['docs']):
        row = next(r for r in rows if r['kind']=='dense_reference' and r['document']==name)
        require(row['file']==f'references/dense-{index}.safetensors', 'Native reference path mismatch')
        path = safe_path(root,row['file']); require(digest(path)==row['sha256'], 'Native reference hash mismatch')
        native[name] = load_file(path)['logits']
        old = load_file(Path(parent_path)/'reference_logits'/f'{index:03d}.safetensors')['logits'][:,:cfg['tokens']]
        from .metrics import compare_logits
        require(all(row[k]==v for k,v in compare_logits(old,native[name],parent['tokens'][name]).items()), 'Fresh reference mismatch')
    for row in rows:
        kind = row['kind']
        if kind=='selector_reproduction':
            mode,name=row['mode'],row['document']; index=parent['docs'].index(name)
            require(row['trace_file']==f'traces/control-{mode}-{index}.npz' and row['reference_file']==f'references/{mode}-{index}.safetensors', 'Selector control paths mismatch')
            require(digest(root/row['trace_file'])==row['trace_sha256'] and digest(root/row['reference_file'])==row['reference_sha256'], 'Selector control hash mismatch')
            with np.load(root/row['trace_file'],allow_pickle=False) as data:
                require(set(data.files)=={'bits','shape'} and data['shape'].tolist()==[cfg['tokens'],cfg['layers'],cfg['groups']], 'Selector control shape mismatch')
                mask=unpack(data,'bits',tuple(data['shape'].tolist()))
            require(row['masks_equal'] is True and row['passed'] is True and np.array_equal(mask,parent['masks'][mode,name])
                and row['selector_bytes']==parent['storage'][mode], 'Selector reproduction mismatch')
        elif kind=='group_reconstruction':
            require(row['passed'] is True and math.isfinite(row['relative_l2']) and 0<=row['relative_l2']<=.01, 'Local gate failed')
        elif kind in ('dense_reference','layout_correctness','zero_repair_correctness','full_repair_correctness'):
            name = row['document']; reference = native[name]
            if kind in ('zero_repair_correctness','full_repair_correctness'):
                index = parent['docs'].index(name)
                relative = f"references/{row['mode']}-{index}.safetensors" if kind=='zero_repair_correctness' else f'references/dense-{index}.safetensors'
                require(row['reference_file']==relative and digest(root/relative)==row['reference_sha256'], 'Control reference mismatch')
                reference = load_file(root/relative)['logits']
                require(row['initial_mask_sha256']==hashlib.sha256(parent['masks'][row['mode'],name].tobytes()).hexdigest(), 'Control initial mask mismatch')
            if kind!='dense_reference':metrics_valid(row,reference,parent['tokens'][name])
            require(row['passed'] is True and numerical_pass(row), 'Numerical gate failed')
    documents = []
    for row in rows:
        if row['kind'] not in ('repair_document','one_shot_document'):continue
        mode,name = row['mode'],row['document']; metrics_valid(row,native[name],parent['tokens'][name])
        path = safe_path(root,row['trace_file']); require(digest(path)==row['trace_sha256'], 'Trace hash mismatch')
        if row['kind']=='repair_document':
            schedule,limit = row['schedule'],row['limit']
            require(schedule in repair.SCHEDULES and limit in repair.LIMITS and row['condition']==f'{mode}-{schedule}-{limit}', 'Repair condition mismatch')
            initial = parent['masks'][mode,name]; hot,charged,budget = repair_hot(parent,mode,cfg)
            added,final,audits = validate_repair_trace(path,initial,hot,schedule,limit)
            require(row['initial_replay']==replay(initial,parent['sizes'],parent['importance'],budget,'static_equal_layer'), 'Initial replay mismatch')
            require(all(row[k]==v for k,v in repair.trace_metrics(initial,added,hot,parent['sizes']).items()), 'Acquisition accounting mismatch')
            require(row['audit_mean']==audits.mean((0,1)).tolist() and row['audit_p95']==np.quantile(audits,.95,axis=(0,1)).tolist(), 'Audit summary mismatch')
        else:
            keep = row['keep']; require(keep in repair.ONE_SHOT and row['schedule']=='one_shot' and row['condition']==f'{mode}-one_shot-{keep:g}', 'One-shot condition mismatch')
            with np.load(path,allow_pickle=False) as data:
                require(set(data.files)=={'bits','shape'} and data['shape'].tolist()==[cfg['tokens'],cfg['layers'],cfg['groups']], 'One-shot trace shape mismatch')
                final = unpack(data,'bits',tuple(data['shape'].tolist()))
            require((final.sum(-1)==math.ceil(keep*cfg['groups'])).all(), 'One-shot retention mismatch')
            charged = parent['storage'][mode]; budget = cfg['budget_bytes']-charged
        require(row['selector_bytes']==charged and row['replay']==replay(final,parent['sizes'],parent['importance'],budget,'static_equal_layer'), 'Final charged replay mismatch')
        documents.append(row)
    baseline = dense_static_frontier(parent['means'],parent['layouts'],[8,32,128],[cfg['budget_bytes']],3*cfg['hidden']*4,
        [cfg['tokens']]*cfg['documents'])[(cfg['budget_bytes'],'warm')]
    results = []
    for condition in condition_keys:
        selected = [r for r in documents if r['condition']==condition]
        warm = sum(next(v['total_bytes'] for v in r['replay'] if v['state']=='warm') for r in selected)
        cold = sum(next(v['total_bytes'] for v in r['replay'] if v['state']=='cold') for r in selected)
        quality = aggregate(selected); saving = 1-warm/baseline['total_bytes']
        value = {'condition':condition,**quality,'warm_bytes':warm,'cold_bytes':cold,'dense_baseline':baseline,
            'warm_saving':saving,'diagnostic_quality_traffic_pass':quality['relative_perplexity']<=cfg['ppl_limit'] and saving>=cfg['warm_saving_min'],
            'runtime_eligible':False}
        require(next(r for r in rows if r['kind']=='aggregate' and r['condition']==condition)=={'kind':'aggregate',**value}, 'Aggregate mismatch')
        results.append(value)
    require(summary['conditions']==results, 'Summary differs from ledger')
    return {'status':'validated','row_counts':dict(Counter(r['kind'] for r in rows)),'conditions':results,
        'input_hashes':{name:digest(root/name) for name in ('config.json','manifest.json','results.jsonl','summary.json')},
        'runtime_nomination':None,'scope':'Receipt/ranking/replay validation of a privileged fixed-mask diagnostic; no causal policy or runtime claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','parent','archive','output'):parser.add_argument('--'+name,required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'repair_analysis.py')
    try:write_json(output/'summary.json',analyze(args.run,args.parent,args.archive))
    except (Exception,KeyboardInterrupt) as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)}); raise


if __name__ == '__main__':main()

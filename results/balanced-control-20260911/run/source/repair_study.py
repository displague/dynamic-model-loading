"""Bounded pretrained repair diagnostic with immutable v0.7 starting masks."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load, load_file, save_file
from transformers import AutoModelForCausalLM

from .adapters import extract_ffns
from . import causal, repair
from .cache import cache_shape, dense_static_frontier, group_arrays, hot_mask, replay
from .causal_study import incremental, numerical_pass, packed
from .experiment import digest, environment, load_corpus, write_json
from .ffn import dimensions, grouped_forward
from .metrics import aggregate, compare_logits, relative_l2
from .provenance import committed_inputs, frozen_environment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_parent(parent, archive, cfg):
    parent, archive = Path(parent), Path(archive)
    if digest(archive/'archive.json') != cfg['parent_index_sha256']:
        raise ValueError('Parent archive index mismatch')
    index = read(archive/'archive.json')
    for name, receipt in index['files'].items():
        if name.startswith('run/'):
            path = parent / name[4:]
            if path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
                raise ValueError('Parent file mismatch: ' + name)
    corpus = load_corpus(parent/'corpus.jsonl')
    docs = [r['id'] for r in corpus if r['split'] == 'diagnostic'][:cfg['documents']]
    ids = read(parent/'token-ids.json')
    tokens = {name: torch.tensor(ids[name], dtype=torch.long)[:, :cfg['tokens']] for name in docs}
    if len(docs) != cfg['documents'] or any(value.shape != (1,cfg['tokens']) for value in tokens.values()):
        raise ValueError('Parent sample mismatch')
    rows = [json.loads(line) for line in (parent/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    payloads = {p['file']: p for a in index['assets'] for p in a['payloads']}
    required = ['learned.safetensors']
    for row in rows:
        if row['kind'] == 'selector_document' and row['mode'] in repair.MODES and row['document'] in docs:
            required.append(row['trace_file'])
    required += [f'reference_logits/{i:03d}.safetensors' for i in range(len(docs))]
    for name in required:
        receipt, path = payloads[name], parent/name
        if path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
            raise ValueError('Parent payload mismatch: '+name)
    masks = {}
    for mode in repair.MODES:
        for name in docs:
            row = next(r for r in rows if r['kind']=='selector_document' and r['mode']==mode and r['document']==name)
            with np.load(parent/row['trace_file'], allow_pickle=False) as trace:
                shape = tuple(trace['shape'].tolist())
                bits = np.unpackbits(trace['bits'], axis=-1, count=shape[-1], bitorder='little').astype(bool)
            value = bits[:cfg['tokens']]
            if value.shape != (cfg['tokens'], cfg['layers'], cfg['groups']) or not (value.sum(-1)==cfg['initial_groups']).all():
                raise ValueError('Invalid starting masks')
            masks[mode,name] = value
    sizes_and_importance = load(gzip.decompress((parent/'calibration-importance.safetensors.gz').read_bytes()))
    means = torch.stack([sizes_and_importance[str(i)] for i in range(cfg['layers'])]).numpy()
    layouts = json.loads(gzip.decompress((parent/'layouts.json.gz').read_bytes()))
    sizes, importance = group_arrays(means,np.asarray(layouts['popularity']),cfg['width'],3*cfg['hidden']*4)
    original = read(parent/'summary.json')
    storage = {r['mode']: r['selector_bytes'] for r in original['selectors']}
    learned = load_file(parent/'learned.safetensors')
    return dict(docs=docs,tokens=tokens,masks=masks,means=means,layouts=layouts,sizes=sizes,
                importance=importance,storage=storage,learned=learned,manifest=read(parent/'manifest.json'))


def validate_config(cfg):
    expected = {'model':'Qwen/Qwen2.5-1.5B-Instruct','revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        'expected_torch':'2.10.0+cu130','documents':2,'tokens':128,'layers':28,'hidden':1536,
        'neurons':8960,'groups':1120,'initial_groups':1008,'width':8,'budget_bytes':2**31,
        'repair_metadata_bytes':1034880,'cpu_threads':4,'seed':1729,
        'limits':list(repair.LIMITS),'schedules':list(repair.SCHEDULES),'modes':list(repair.MODES),
        'one_shot':list(repair.ONE_SHOT),'ppl_limit':1.01,'warm_saving_min':.1,
        'protocol':'docs/refinement-feasibility-protocol.md'}
    if any(cfg.get(k) != v for k,v in expected.items()):
        raise ValueError('Changed prospective repair configuration')


def selector_list(mlps, parent, mode, keep, width):
    selectors = []
    for layer, mlp in enumerate(mlps):
        learned = {k.split('.',1)[1]:v for k,v in parent['learned'].items() if k.startswith(str(layer)+'.')}
        selectors.append(causal.Selector(mode,torch.tensor(parent['importance'][layer],dtype=torch.float32,device=mlp.down_proj.weight.device),
            mlp.down_proj.weight.detach().float().norm(dim=0),width,learned=learned,keep=keep))
    return selectors


def run_causal(model, mlps, parent, mode, keep, ids, width):
    selectors = selector_list(mlps,parent,mode,keep,width)
    observers = [causal.AppliedProbe(mlp,selector,width,keep,audit=False) for mlp,selector in zip(mlps,selectors,strict=True)]
    with causal.probes(mlps,observers):
        logits = incremental(model,ids)
    mask = np.stack([observer.take()[0].numpy() for observer in observers],axis=1)
    return logits,mask,sum(s.bytes() for s in selectors)


def run_repair(model, mlps, ids, initial, hot, schedule, limit, width):
    device = ids.device
    observers = [repair.RepairProbe(mlp,torch.tensor(initial[:,i],device=device),
        torch.tensor(hot[i],device=device),schedule,limit,width) for i,mlp in enumerate(mlps)]
    with repair.repair_probes(mlps,observers):
        logits = incremental(model,ids)
    payloads = [observer.take() for observer in observers]
    repairs,scores,audits = (np.stack([value[i] for value in payloads],axis=1) for i in range(3))
    return logits,repairs,scores,audits


def repair_hot(parent, mode, cfg):
    charged = parent['storage'][mode] + cfg['repair_metadata_bytes']
    budget = cfg['budget_bytes']-charged
    slots = cache_shape(parent['sizes'],budget)['retained_slots']
    return hot_mask(parent['importance'],parent['sizes'],slots,'static_equal_layer'), charged, budget


def save_selector_control(output, mode, index, name, logits, masks, initial, storage, expected_storage):
    path = output/'references'/f'{mode}-{index}.safetensors'
    save_file({'logits':logits},path)
    trace = output/'traces'/f'control-{mode}-{index}.npz'
    np.savez_compressed(trace,bits=np.packbits(masks,axis=-1,bitorder='little'),shape=np.asarray(masks.shape))
    return {'mode':mode,'document':name,'reference_file':path.relative_to(output).as_posix(),
        'reference_sha256':digest(path),'trace_file':trace.relative_to(output).as_posix(),'trace_sha256':digest(trace),
        'selector_bytes':storage,'masks_equal':bool(np.array_equal(masks,initial)),
        'passed':bool(np.array_equal(masks,initial) and storage==expected_storage)}


@torch.inference_mode()
def run(config_path, parent_path, archive_path, output_path):
    output, config_path = Path(output_path), Path(config_path)
    output.mkdir(parents=True,exist_ok=False)
    stream = (output/'results.jsonl').open('x',encoding='utf-8')
    def record(kind, **values):
        stream.write(json.dumps({'kind':kind,**values},allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    try:
        cfg = read(config_path); validate_config(cfg)
        if str(torch.__version__) != cfg['expected_torch'] or not torch.cuda.is_available():
            raise ValueError('Pinned CUDA environment required; no fallback')
        source_receipt, protocol = committed_inputs(__file__,config_path,cfg['protocol'])
        if 'configs/refinement-feasibility.json' not in source_receipt['committed_files']:
            raise ValueError('Use the committed refinement configuration path')
        torch.set_num_threads(cfg['cpu_threads']); torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        shutil.copyfile(config_path,output/'config.json')
        shutil.copyfile(protocol,output/'protocol.md')
        shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
        parent = load_parent(parent_path,archive_path,cfg)
        write_json(output/'token-ids.json',{k:v.tolist() for k,v in parent['tokens'].items()})
        env = environment(torch.device('cuda'))
        started = {'started_utc':datetime.now(timezone.utc).isoformat(),**source_receipt,
            'config_sha256':digest(config_path),'protocol_sha256':digest(output/'protocol.md'),
            'sources':{p.relative_to(output/'source').as_posix():digest(p) for p in (output/'source').rglob('*.py')},
            'environment':env,'parent_index_sha256':cfg['parent_index_sha256']}
        write_json(output/'started.json',started)
        frozen_environment(env)
        snapshot = Path(snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True))
        checkpoint = {p.name:digest(p) for p in snapshot.iterdir() if p.is_file()}
        if checkpoint != parent['manifest']['checkpoint_files']:
            raise ValueError('Checkpoint differs from pinned parent')
        print('Loading pinned Qwen for bounded repair controls',flush=True)
        model = AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).to('cuda').eval()
        mlps = extract_ffns(model)
        if [dimensions(m) for m in mlps] != [(cfg['hidden'],cfg['neurons'])]*cfg['layers']:
            raise ValueError('Model shapes differ from protocol')
        manifest = {**started,'checkpoint_files':checkpoint,'attention_implementation':model.config._attn_implementation,
            'model_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'restoration_bytes':int(parent['sizes'].sum()),
            'frozen_mask_bytes_per_episode':cfg['tokens']*cfg['layers']*cfg['groups'],
            'repair_metadata_bytes':cfg['repair_metadata_bytes']}
        write_json(output/'manifest.json',manifest)
        for directory in ('references','traces'):(output/directory).mkdir()
        native = {}; gate = True
        for index,name in enumerate(parent['docs']):
            ids = parent['tokens'][name].cuda(); value = incremental(model,ids)
            path = output/'references'/f'dense-{index}.safetensors'; save_file({'logits':value},path)
            old = load_file(Path(parent_path)/'reference_logits'/f'{index:03d}.safetensors')['logits'][:,:cfg['tokens']]
            metrics = compare_logits(old,value,ids.cpu()); passed = numerical_pass(metrics); gate &= passed
            record('dense_reference',document=name,file=path.relative_to(output).as_posix(),sha256=digest(path),passed=passed,**metrics)
            native[name] = value
        with packed(mlps,parent['layouts']['popularity']):
            captured = []; handles = []
            try:
                for mlp in mlps:
                    handles.append(mlp.register_forward_pre_hook(lambda module,args:captured.append(args[0].reshape(-1,cfg['hidden'])[:2].clone())))
                cal_id = next(r['id'] for r in load_corpus(Path(parent_path)/'corpus.jsonl') if r['split']=='calibration')
                cal = torch.tensor(read(Path(parent_path)/'token-ids.json')[cal_id],device='cuda')
                model(input_ids=cal,use_cache=False)
            finally:
                for handle in handles:handle.remove()
            for layer,(mlp,x) in enumerate(zip(mlps,captured,strict=True)):
                error = relative_l2(mlp(x),grouped_forward(mlp,x,cfg['width'])); passed = error<=.01; gate &= passed
                record('group_reconstruction',layer=layer,relative_l2=error,passed=passed)
            del captured
            for name in parent['docs']:
                metrics = compare_logits(native[name],incremental(model,parent['tokens'][name].cuda()),parent['tokens'][name])
                passed = numerical_pass(metrics); gate &= passed
                record('layout_correctness',document=name,passed=passed,**metrics)
            for mode in repair.MODES:
                hot,charged,budget = repair_hot(parent,mode,cfg)
                for index,name in enumerate(parent['docs']):
                    ids = parent['tokens'][name].cuda(); initial = parent['masks'][mode,name]
                    original,masks,storage = run_causal(model,mlps,parent,mode,.9,ids,cfg['width'])
                    control = save_selector_control(output,mode,index,name,original,masks,initial,storage,parent['storage'][mode])
                    record('selector_reproduction',**control)
                    if not control['passed']:
                        raise ValueError('Fresh initial selector differs from frozen parent')
                    path = output/'references'/f'{mode}-{index}.safetensors'
                    for limit,kind,reference in ((0,'zero_repair_correctness',original),(112,'full_repair_correctness',native[name])):
                        repaired,_,_,_ = run_repair(model,mlps,ids,initial,hot,'hindsight',limit,cfg['width'])
                        metrics = compare_logits(reference,repaired,ids.cpu()); passed = numerical_pass(metrics); gate &= passed
                        reference_path = path if limit == 0 else output/'references'/f'dense-{index}.safetensors'
                        record(kind,mode=mode,document=name,passed=passed,reference_file=reference_path.relative_to(output).as_posix(),
                            reference_sha256=digest(reference_path),initial_mask_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),**metrics)
            if not gate:raise ValueError('Numerical gate failed; repair curves blocked')
            document_rows = []
            for mode in repair.MODES:
                hot,charged,budget = repair_hot(parent,mode,cfg)
                for schedule in repair.SCHEDULES:
                    for limit in repair.LIMITS:
                        condition = f'{mode}-{schedule}-{limit}'
                        for index,name in enumerate(parent['docs']):
                            ids = parent['tokens'][name].cuda(); initial = parent['masks'][mode,name]
                            logits,added,scores,audits = run_repair(model,mlps,ids,initial,hot,schedule,limit,cfg['width'])
                            path = output/'traces'/f'{condition}-{index}.npz'; repair.save_receipt(path,initial,added,scores,audits)
                            final = initial|added
                            value = {'condition':condition,'mode':mode,'schedule':schedule,'limit':limit,'document':name,
                                'trace_file':path.relative_to(output).as_posix(),'trace_sha256':digest(path),
                                'selector_bytes':charged,'initial_replay':replay(initial,parent['sizes'],parent['importance'],budget,'static_equal_layer'),
                                'replay':replay(final,parent['sizes'],parent['importance'],budget,'static_equal_layer'),
                                **repair.trace_metrics(initial,added,hot,parent['sizes']),
                                'audit_mean':audits.mean((0,1)).tolist(),'audit_p95':np.quantile(audits,.95,axis=(0,1)).tolist(),
                                **compare_logits(native[name],logits,ids.cpu())}
                            record('repair_document',**value); document_rows.append(value)
                        print('Completed',condition,flush=True)
                for keep in repair.ONE_SHOT:
                    condition = f'{mode}-one_shot-{keep:g}'
                    for index,name in enumerate(parent['docs']):
                        logits,mask,storage = run_causal(model,mlps,parent,mode,keep,parent['tokens'][name].cuda(),cfg['width'])
                        path = output/'traces'/f'{condition}-{index}.npz'
                        np.savez_compressed(path,bits=np.packbits(mask,axis=-1,bitorder='little'),shape=np.asarray(mask.shape))
                        value = {'condition':condition,'mode':mode,'schedule':'one_shot','keep':keep,'document':name,
                            'trace_file':path.relative_to(output).as_posix(),'trace_sha256':digest(path),'selector_bytes':storage,
                            'replay':replay(mask,parent['sizes'],parent['importance'],cfg['budget_bytes']-storage,'static_equal_layer'),
                            **compare_logits(native[name],logits,parent['tokens'][name])}
                        record('one_shot_document',**value); document_rows.append(value)
                    print('Completed',condition,flush=True)
            baseline = dense_static_frontier(parent['means'],parent['layouts'],[8,32,128],[cfg['budget_bytes']],
                3*cfg['hidden']*4,[cfg['tokens']]*cfg['documents'])[(cfg['budget_bytes'],'warm')]
            results = []
            for condition in dict.fromkeys(r['condition'] for r in document_rows):
                selected = [r for r in document_rows if r['condition']==condition]
                warm = sum(next(x['total_bytes'] for x in r['replay'] if x['state']=='warm') for r in selected)
                cold = sum(next(x['total_bytes'] for x in r['replay'] if x['state']=='cold') for r in selected)
                quality = aggregate(selected); saving = 1-warm/baseline['total_bytes']
                value = {'condition':condition,**quality,'warm_bytes':warm,'cold_bytes':cold,'dense_baseline':baseline,
                    'warm_saving':saving,'diagnostic_quality_traffic_pass':quality['relative_perplexity']<=cfg['ppl_limit'] and saving>=cfg['warm_saving_min'],
                    'runtime_eligible':False}
                record('aggregate',**value); results.append(value)
        summary = {'status':'repair_diagnostic_completed','correctness_passed':True,'conditions':results,
            'runtime_nomination':None,'finished_utc':datetime.now(timezone.utc).isoformat()}
        write_json(output/'summary.json',summary)
        return summary
    except (Exception,KeyboardInterrupt) as error:
        record('error',error_type=type(error).__name__,message=str(error))
        write_json(output/'failure.json',{'error_type':type(error).__name__,'message':str(error)})
        raise
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','parent','archive','output'):parser.add_argument('--'+name,required=True)
    args = parser.parse_args(); print(run(args.config,args.parent,args.archive,args.output)['status'])


if __name__ == '__main__':main()

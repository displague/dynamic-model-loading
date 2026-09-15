"""CPU-only independent decision-field ledger and full-logit replay."""
import itertools
import json
from pathlib import Path

import numpy as np

from .decision_field import FROZEN, branches, validate_config
from .fault_screen import require_supervisor

SLAB = 10321920


def need(ok, message):
    if not ok:
        raise ValueError(message)


def frame_metrics(vectors):
    names = {name for name,_ in branches()} | {'target'}
    need(set(vectors)==names, 'Logit branch inventory')
    values = {k:np.asarray(v,dtype=np.float64) for k,v in vectors.items()}
    shape = values['base'].shape
    need(len(shape)==1 and shape[0]>1 and all(v.shape==shape and np.isfinite(v).all() for v in values.values()), 'Invalid logits')
    ids = {k:int(v.argmax()) for k,v in values.items()}
    base = values['base']
    pairs = []
    for i,j in itertools.combinations(FROZEN['sites'],2):
        name = f'{i}+{j}'
        predicted = values[str(i)]+values[str(j)]-base
        actual = values[name]
        interaction = float(np.linalg.norm(actual-predicted))
        change = float(np.linalg.norm(actual-base))
        pairs.append(dict(pair=name, argmax_match=int(predicted.argmax())==ids[name],
            logit_relative_l2=interaction/max(float(np.linalg.norm(actual)),1e-12),
            interaction_norm=interaction, joint_change_norm=change,
            interaction_ratio=interaction/max(change,1e-12)))
    available = [name for name,sites in branches() if 0<len(sites)<=2]
    return dict(argmax=ids, opportunity=ids['base']!=ids['all8'] and any(ids[k]==ids['all8'] for k in available),
        base_target_match=ids['base']==ids['target'], all8_target_match=ids['all8']==ids['target'],
        base_all8_match=ids['base']==ids['all8'], pairs=pairs)


def summarize_frames(frames):
    need(bool(frames), 'No frames')
    pairs = [p for f in frames for p in f['pairs']]
    ratio = sum(p['interaction_norm'] for p in pairs)/max(sum(p['joint_change_norm'] for p in pairs),1e-12)
    identity = sum(p['argmax_match'] for p in pairs)/len(pairs)
    return dict(positions=len(frames), opportunities=sum(f['opportunity'] for f in frames),
        base_all8_matches=sum(f['base_all8_match'] for f in frames),
        base_target_matches=sum(f['base_target_match'] for f in frames),
        all8_target_matches=sum(f['all8_target_match'] for f in frames),
        pairs=len(pairs), additive_argmax_fraction=identity, aggregate_interaction_ratio=ratio,
        mean_pair_logit_relative_l2=float(np.mean([p['logit_relative_l2'] for p in pairs])))


def validate_branch_rows(rows,docs,tokens):
    expected = [(doc,split,pos,name,sites) for doc,split in docs for pos in range(4) for name,sites in branches()]
    need(len(rows)==len(expected), 'Missing/extra branch rows')
    for r,(doc,split,pos,name,sites) in zip(rows,expected,strict=True):
        need((r['document'],r['split'],r['position'],r['branch'],r['sites'])==(doc,split,pos,name,sites),'Branch order drift')
        need(r['consumed_token']==tokens[doc][4+pos] and r['base_length']==4+pos,'Token trajectory drift')
        need(r['end_length']==4+pos+int(name=='base'), 'Crop length mismatch')
        need(r['prefix_before']==r['prefix_after'] and len(r['prefix_before'])==64, 'KV fingerprint drift')
        need(r['kv_fingerprint_d2h_bytes']==57344*(4+pos),'KV fingerprint accounting')
        need(r['draft_kv_bytes']==57344*r['end_length'] and r['target_kv_bytes']==57344*(5+pos),'KV allocation accounting')
        need(r['draft_kv_storage_bytes']==57344*(5+pos) and r['target_kv_storage_bytes']==57344*(5+pos),'Actual KV storage accounting')
        need(r['h2d_bytes']==2*len(sites)*SLAB, 'Physical branch traffic mismatch')
        need(r['correction_scalar_d2h_bytes']==8*len(sites), 'Scalar readback accounting')
        need(np.isfinite(r['wall_seconds']) and r['wall_seconds']>0, 'Invalid branch time')
    for offset in range(0,len(rows),12):
        need(len({r['prefix_before'] for r in rows[offset:offset+12]})==1,'Branches did not share history')


def validate_page_rows(rows,docs):
    expected = []
    for layer in range(28):
        expected.extend(('mechanics',None,None,None,layer+1,[layer,s]) for s in (0,1))
    for doc,_ in docs:
        for token in range(1,5):
            expected.extend(('prefill',doc,None,None,token,[layer,s]) for layer in range(28) for s in (0,1))
        for pos in range(4):
            for bi,(name,sites) in enumerate(branches()):
                expected.extend(('intervention',doc,pos,name,5+pos*12+bi,[layer,s]) for layer in sites for s in (0,1))
    need(len(rows)==len(expected), 'Missing/extra physical page records')
    for r,exp in zip(rows,expected,strict=True):
        need((r['phase'],r['document'],r['position'],r['branch'],r['token'],r['key'])==exp,'Page order/key mismatch')
        need(r['kind']=='load' and r['slot']==1 and r['bytes']==SLAB,'Expected cold physical bypass load')
    return dict(loads=len(rows),h2d_bytes=len(rows)*SLAB,hits=0)


def validate_parent_reference(run, inventory):
    from .experiment import digest
    need(digest(Path(run)/'parent-mechanics.safetensors')==inventory['mechanics.safetensors'],
         'Numerical reference differs from frozen representation parent')


def validate_phases(phases,docs,known):
    need([r['phase'] for r in phases]==['model_loading','construction_and_equality','mechanics']+
         ['prefill','frame','frame','frame','frame']*len(docs),'Phase inventory drift')
    construction = phases[1]
    need(construction['equal_layers']==[True]*28 and construction['construction_h2d_bytes']==650280960 and
         construction['snapshot_d2h_bytes']==650280960,'Construction accounting')
    mechanics = phases[2]
    need(mechanics['correction_scalar_d2h_bytes']==28*8 and mechanics['input_h2d_bytes']==28*1536*4 and
         mechanics['output_d2h_bytes']==28*1536*4,'Mechanical transfer accounting')
    need(len(mechanics['relative_l2'])==28 and all(np.isfinite(v) and 0<=v<=.01 for v in mechanics['relative_l2']),
         'Numerical phase failure')
    for di,(doc,split) in enumerate(docs):
        prefill = phases[3+5*di]
        need(prefill['document']==doc and prefill['kv_bytes']==2*4*57344 and
             prefill['correction_scalar_d2h_bytes']==4*28*8,'Prefill accounting')
        for pos in range(4):
            frame = phases[4+5*di+pos]
            need((frame['document'],frame['split'],frame['position'])==(doc,split,pos),'Frame phase identity')
            need(frame['initial_fingerprint_d2h_bytes']==57344*(4+pos) and
                 frame['target_logit_d2h_bytes']==151936*4 and
                 np.isfinite(frame['target_seconds']) and 0<frame['target_seconds']<=frame['wall_seconds'],
                 'Frame readback/time accounting')
    need(all(r['extra_cuda_peak_bytes']==r['cuda']['peak_allocated_bytes']-known and
        0<=r['extra_cuda_peak_bytes']<=1024*2**20 and np.isfinite(r['wall_seconds']) and r['wall_seconds']>0
        for r in phases),'Phase memory/time failure')


def analyze(run):
    from safetensors.numpy import load_file
    from .experiment import digest, load_corpus
    from .fault_pager_analysis import read_rows
    from .provenance import verify_snapshot, frozen_environment
    run = Path(run)
    require_supervisor(run)
    need((run/'completion.json').is_file() and not (run/'failure.json').exists(),'Incomplete worker')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/decision-field.json',FROZEN['protocol'],__file__)
    frozen_environment(manifest['environment'])
    env = manifest['environment']
    need(env['device']=='cuda:0' and env['cpu_threads']==4 and not env['tf32_matmul'] and not env['tf32_cudnn'],'Environment drift')
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    inventory = json.loads((run/'files.json').read_text(encoding='utf-8'))
    need(set(inventory)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p.name!='files.json'},'Raw inventory drift')
    for name,expected in inventory.items():
        need(digest(run/name)==expected,'Raw digest drift: '+name)
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256'),('representation-files.json','representation_files_sha256')]:
        need(digest(run/name)==cfg[key],'Frozen artifact drift')
    validate_parent_reference(run,json.loads((run/'representation-files.json').read_text(encoding='utf-8')))
    need(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','Checkpoint inventory drift')
    need(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text(encoding='utf-8'))['checkpoint']['files'],'Checkpoint drift')
    corpus = load_corpus(run/'corpus.jsonl')
    docs = [(r['id'],'fit') for r in corpus if r['split']=='calibration'][:6]
    docs += [(r['id'],'diagnostic') for r in corpus if r['split']=='diagnostic'][:2]
    tokens = json.loads((run/'token-ids.json').read_text(encoding='utf-8'))
    rows = list(read_rows(run/'branches.jsonl'))
    validate_branch_rows(rows,docs,tokens)
    traffic = validate_page_rows(list(read_rows(run/'pages.jsonl.gz')),docs)
    phases = list(read_rows(run/'phases.jsonl'))
    account = json.loads((run/'allocation.json').read_text(encoding='utf-8'))
    expected = dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        draft_host_parameters_bytes=4624220160,charged_baseline_bytes=7725494272,shared_bytes=0,
        resident_bytes=650280960,workspace_bytes=254607360,cache_pool_bytes=2*SLAB,
        staging_bytes=SLAB,host_increments_bytes=578027520)
    need(all(account.get(k)==v for k,v in expected.items()),'Allocation drift')
    validate_phases(phases,docs,account['charged_baseline_bytes'])
    original,actual = load_file(run/'parent-mechanics.safetensors'),load_file(run/'mechanics.safetensors')
    need(set(actual)=={f'q8.{i}' for i in range(28)},'Mechanics inventory drift')
    errors = []
    for key,v in actual.items():
        need(v.shape==original[key].shape==(1,1,1536) and v.dtype==original[key].dtype==np.float32,
             'Mechanical tensor shape/dtype drift')
        need(np.isfinite(v).all() and np.isfinite(original[key]).all(),'Nonfinite mechanics')
        errors.append(float(np.linalg.norm(v.astype('float64')-original[key]))/max(float(np.linalg.norm(original[key].astype('float64'))),1e-12))
    need(max(errors)<=.01,'Numerical failure')
    frames = []
    for di,(doc,split) in enumerate(docs):
        for pos in range(4):
            vectors = load_file(run/f'frame-{di:02}-{pos}'/'logits.safetensors')
            need(all(v.shape==(151936,) and v.dtype==np.float32 for v in vectors.values()),'Logit shape/dtype drift')
            need(all(r['logit_d2h_bytes']==151936*4 for r in rows if r['document']==doc and r['position']==pos),'Logit readback accounting')
            frames.append(dict(document=doc,split=split,position=pos,**frame_metrics(vectors)))
    stats = {split:summarize_frames([f for f in frames if f['split']==split]) for split in ('fit','diagnostic')}
    diagnostic = stats['diagnostic']
    gates = dict(Hdecision=diagnostic['opportunities']>=2,Hjoint=diagnostic['additive_argmax_fraction']>=.9 and diagnostic['aggregate_interaction_ratio']<=.25)
    samples = list(read_rows(run/'resources.jsonl'))
    need(bool(samples) and all(r['gpu_used']<=15000*2**20 and r['host_available']>=2048*2**20 for r in samples),'Resource cap failure')
    resources = dict(samples=len(samples),peak_gpu_used=max(r['gpu_used'] for r in samples),
        min_host_available=min(r['host_available'] for r in samples),peak_rss=max(r['process']['rss'] for r in samples),passed=True,error=None)
    need(resources==json.loads((run/'final-resources.json').read_text(encoding='utf-8')),'Resource receipt mismatch')
    return dict(source_commit=manifest['source_commit'],statistics=stats,gates=gates,
        decision='short_estimation_hypothesis_only' if gates['Hdecision'] else 'stop_this_action_frontier',
        additive_model_supported=gates['Hjoint'],frames=frames,traffic=traffic,allocation=account,resources=resources,
        max_ffn_relative_l2=max(errors),peak_extra_cuda_bytes=max(r['extra_cuda_peak_bytes'] for r in phases),
        branch_wall_seconds=sum(r['wall_seconds'] for r in rows),
        note='Privileged teacher-forced development interventions; no causal policy, generated accepts, capacity or speedup claim.')

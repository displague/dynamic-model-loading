"""Independent, model-free replay of the progressive-acquisition short screen."""
from collections import Counter
import json
import math
from pathlib import Path

from .debt_analysis import demand,read_rows,audit_timing,audit_cuda

SLAB_BYTES=10321920


def extrapolation_diagnostic(mech,layers=28):
    from .metrics import relative_l2
    rows=[]
    for i in range(layers):
        ref,a,b,c=(mech[f'{m}.{i}'].double() for m in ('fp32','q4','q6','q8'))
        d1,d2=b-a,c-b
        n1,n2=float(d1.norm()),float(d2.norm())
        rows.append(dict(layer=i,q6_error=relative_l2(ref,b),
            richardson_error=relative_l2(ref,b+d1/3),
            increment_ratio=n2/n1 if n1 else None,
            increment_cosine=float((d1*d2).sum())/(n1*n2) if n1 and n2 else None))
    mean6=sum(r['q6_error'] for r in rows)/layers
    mean_r=sum(r['richardson_error'] for r in rows)/layers
    nonworse=sum(r['richardson_error']<=r['q6_error'] for r in rows)
    return dict(rows=rows,mean_q6_error=mean6,mean_richardson_error=mean_r,
        nonworse_layers=nonworse,passed=mean_r<=.9*mean6 and nonworse>=math.ceil(.75*layers))


def audit_policy(policy,pages,mode,consumed,*,layers=28,slots=8,slab_bytes=SLAB_BYTES,threshold=.02):
    """Replay admissions from prior observations, not runtime-reported scores."""
    demand(len(policy)==consumed*layers,'missing layer observations')
    scores,entries={},{}
    physical=iter(pages)
    stats=dict(h2d_bytes=0,hits=0,loads=0,evictions=0)
    d2h=0
    precisions=Counter()
    calibration=[]
    for token in range(1,consumed+1):
        ranked=sorted((k for k,v in scores.items() if v>0),key=lambda k:(-scores[k],k))
        admitted={(i,0) for i in range(min(slots,layers))} if mode=='static' else set(ranked[:slots]) if mode=='retained' else set()
        for key in sorted(set(entries)-admitted):
            demand(next(physical,None)==dict(kind='evict',token=token,key=list(key)),'incorrect eviction')
            entries.pop(key)
            stats['evictions']+=1
        scores={k:v*.5 for k,v in scores.items()}
        for layer in range(layers):
            row=policy[(token-1)*layers+layer]
            demand(row['token']==token and row['layer']==layer and row['mode']==mode,'layer ordering drift')
            values=row['values']
            demand(isinstance(values,list) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),'bad correction observation')
            expected=0 if mode=='q4' else 1 if mode=='q6' else 2 if mode=='q8' else (1+(values[0]/4>threshold) if values else -1)
            demand(row['stages']==len(values)==expected and row['d2h_bytes']==4*expected,'precision or readback drift')
            d2h+=row['d2h_bytes']
            precisions[4+2*expected]+=1
            if expected==2:
                calibration.append(dict(predicted=values[0]/4,observed=values[1]))
            for stage,value in enumerate(values):
                key=(layer,stage)
                if key in entries:
                    kind,slot='hit',entries[key]
                    stats['hits']+=1
                else:
                    kind='load'
                    slot=next(i for i in range(slots) if i not in entries.values()) if key in admitted else slots
                    if slot!=slots:
                        entries[key]=slot
                    stats['loads']+=1
                    stats['h2d_bytes']+=slab_bytes
                demand(next(physical,None)==dict(kind=kind,token=token,key=list(key),slot=slot,
                    bytes=slab_bytes if kind=='load' else 0),'physical increment event differs')
                scores[key]=scores.get(key,0)+.5*value
    demand(next(physical,None) is None,'extra physical event')
    return dict(cache=stats,d2h_bytes=d2h,precisions=dict(precisions),
                proxy_observations=calibration,final_resident_slabs=len(entries))


def decision(c):
    h1=c['q6']['acceptance']>=.25 and c['q6']['acceptance']>=c['q4']['acceptance']+.10
    h2=c['adaptive']['acceptance']>=c['q6']['acceptance']-.10 and c['adaptive']['h2d_per_consumed']<=.9*c['q8']['h2d_per_consumed']
    h3=c['retained']['h2d_bytes']<=.9*c['adaptive']['h2d_bytes']
    return dict(hypotheses=dict(H1_representation=h1,H2_acquisition=h2,H3_persistence=h3),
                retained_5pct_faster=c['retained']['wall_seconds']<=.95*c['adaptive']['wall_seconds'],
                decision='advance_to_new_protocol' if h1 and (h2 or h3) else 'stop',full_suite_launched=False)


def analyze(run):
    import torch
    from safetensors.torch import load_file
    from .experiment import digest,load_corpus
    from .metrics import relative_l2
    from .provenance import verify_snapshot,frozen_environment
    from .fault_screen import require_supervisor
    from .fault_pager_analysis import audit_rounds
    from .refinement_screen import FROZEN,validate_config

    run=Path(run)
    torch.set_num_threads(4)
    require_supervisor(run)
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    files=json.loads((run/'files.json').read_text(encoding='utf-8'))
    demand(set(files)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}-{'files.json'},'raw inventory drift')
    for name,value in files.items():
        demand(digest(run/name)==value,'raw digest differs: '+name)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/refinement-screen.json',cfg['protocol'],__file__)
    frozen_environment(manifest['environment'])
    demand(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','checkpoint parent drift')
    demand(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text(encoding='utf-8'))['checkpoint']['files'],'checkpoint receipt differs')
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        demand(digest(run/name)==cfg[key],'frozen input differs')
    a=json.loads((run/'allocation.json').read_text(encoding='utf-8'))
    expected=dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        draft_host_parameters_bytes=4624220160,shared_bytes=0,charged_baseline_bytes=7725494272,
        resident_bytes=650280960,workspace_bytes=254607360,cache_pool_bytes=9*SLAB_BYTES,
        staging_bytes=SLAB_BYTES,host_increments_bytes=56*SLAB_BYTES,
        construction_h2d_bytes=650280960,construction_d2h_bytes=0,snapshot_d2h_bytes=650280960)
    for key,value in expected.items():
        demand(type(a[key]) is int and a[key]==value,'invalid allocation: '+key)
    static=a['resident_bytes']+a['workspace_bytes']+a['cache_pool_bytes']
    baseline=a['baseline_cuda']['allocated_bytes']
    demand(baseline>=a['charged_baseline_bytes'],'baseline undercharged')
    for label in ('baseline_cuda','construction_cuda'):
        audit_cuda(a[label])
    demand(a['construction_cuda']['allocated_bytes']>=baseline+static,'resident storage undercharged')
    demand(a['construction_cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']<=1024*2**20,'construction allowance exceeded')
    for field in ('construction_wall_seconds','snapshot_wall_seconds'):
        demand(math.isfinite(a[field]) and a[field]>0,'invalid setup timing')
    demand(set(a['constructed_sha256'])=={f'layer-{i:02}.safetensors' for i in range(28)},'constructed inventory differs')
    sizes=Counter()
    mech=load_file(run/'mechanics.safetensors')
    demand(set(mech)=={f'{m}.{i}' for m in ('input','fp32','independent8','q4','q6','q8') for i in range(28)},'mechanics inventory differs')
    demand(all(t.shape==(1,1,1536) and torch.isfinite(t).all() for t in mech.values()),'mechanics shape/finite check')
    artifact_errors=[]
    for name,sha in a['constructed_sha256'].items():
        demand(digest(run/'constructed'/name)==sha,'constructed hash differs')
        t=load_file(run/'constructed'/name)
        demand(set(t)=={f'{j}.{n}' for j in range(3) for n in ('base','minimum','scale')}|{'increment.0','increment.1'},'constructed tensors differ')
        for key,value in t.items():
            shape=(3,8960,384) if key.startswith('increment') else (8960,768) if key.endswith('base') else (8960,12)
            dtype=torch.uint8 if key.startswith('increment') or key.endswith('base') else torch.float32
            demand(value.shape==shape and value.dtype==dtype and torch.isfinite(value).all(),'constructed tensor metadata/finite check')
            sizes['host' if key.startswith('increment') else 'resident']+=value.numel()*value.element_size()
        # Independent CPU interpretation of the archived bits: do not call the
        # runtime encoder, unpacker, controller or patched model.
        layer=int(name[6:8])
        weights=[]
        for j in range(3):
            packed=t[f'{j}.base']
            code=torch.stack([packed&15,packed>>4],dim=-1).reshape(8960,1536).float()*16+7.5
            weights.append(code)
        for stage,mode in enumerate(('q4','q6','q8')):
            if stage:
                for j in range(3):
                    packed=t[f'increment.{stage-1}'][j]
                    extra=torch.stack([(packed>>(2*k))&3 for k in range(4)],dim=-1).reshape(8960,1536).float()
                    weights[j].add_(extra*(4 if stage==1 else 1)-(6 if stage==1 else 1.5))
                del extra
            material=[(w.reshape(8960,12,128)*t[f'{j}.scale'][...,None]+t[f'{j}.minimum'][...,None]).reshape(8960,1536)
                      for j,w in enumerate(weights)]
            x=mech[f'input.{layer}'].reshape(1,1536)
            expected=((torch.nn.functional.silu(x@material[0].T)*(x@material[1].T))@material[2]).reshape(1,1,1536)
            error=relative_l2(expected,mech[f'{mode}.{layer}'])
            demand(error<=.01,'archived bit planes do not reconstruct local output')
            artifact_errors.append(error)
            del material,expected
        del weights
        del t
    demand(sizes==dict(host=a['host_increments_bytes'],resident=a['resident_bytes']),'constructed byte accounting differs')
    errs=[relative_l2(mech[f'independent8.{i}'],mech[f'q8.{i}']) for i in range(28)]
    numerical=json.loads((run/'mechanics.json').read_text(encoding='utf-8'))
    demand(numerical['passed'] and max(errs)<=.01 and numerical['relative_l2']==errs and numerical['h2d_bytes']==28*3*SLAB_BYTES,'independent reconstruction differs')
    audit_cuda(numerical['cuda'])
    demand(numerical['cuda']['allocated_bytes']>=baseline+static and
           numerical['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']<=1024*2**20,'mechanics CUDA allowance failed')
    local={m:[relative_l2(mech[f'fp32.{i}'],mech[f'{m}.{i}']) for i in range(28)] for m in ('q4','q6','q8')}
    extrapolation=extrapolation_diagnostic(mech)
    corpus=load_corpus(run/'corpus.jsonl')
    ds=[r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    cal=next(r['id'] for r in corpus if r['split']=='calibration')
    rows=list(read_rows(run/'episodes.jsonl'))
    order=[('target',d,d==cal) for d in [cal,*ds]]
    order +=[(m,cal,True) for m in cfg['conditions']]
    order +=[(m,d,False) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
    demand([(r['mode'],r['document'],r['warmup']) for r in rows]==order,'episode matrix/order differs')
    audits={}
    traces={}
    refs={r['document']:r for r in rows[:3]}
    for index,row in enumerate(rows):
        audit_timing(row)
        if index:
            demand(row['charged_started_monotonic']>=rows[index-1]['charged_finished_monotonic'],'overlapping episodes')
        prefix=2 if row['warmup'] else 4
        if row['mode']=='target':
            demand(row['target_kv_peak_bytes']==57344*(prefix+len(row['ids'])-1) and row['draft_kv_peak_bytes']==0,'target KV drift')
            audit_cuda(row['cuda'])
            extra=row['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']
            demand(row['cuda']['allocated_bytes']>=baseline+static and extra==row['extra_cuda_peak_bytes']<=1024*2**20
                   and extra>=baseline-a['charged_baseline_bytes']+static+row['target_kv_peak_bytes'],'target-reference memory accounting')
            continue
        folder=run/f'episode-{index-3:02}'
        demand(json.loads((folder/'episode.json').read_text(encoding='utf-8'))==row,'episode receipt drift')
        ref=refs[row['document']]
        demand(row['reference_match'] and row['ids']==ref['ids'] and row['stop_reason']==ref['stop_reason'],'reference mismatch')
        count=audit_rounds(folder/'rounds.jsonl',row,prefix_tokens=prefix,generation_tokens=1 if row['warmup'] else 4)//28
        demand(count==row['consumed_draft_tokens'],'consumption counter drift')
        policy=list(read_rows(folder/'policy.jsonl.gz'))
        pages=list(read_rows(folder/'pages.jsonl.gz'))
        audit=audit_policy(policy,pages,row['mode'],count)
        demand(audit['cache']==row['cache'],'physical counters differ')
        audits[row['episode']]=audit
        rr=list(read_rows(folder/'rounds.jsonl'))
        expected_kv=57344*max([prefix]+[r['base']+4 for r in rr])
        demand(row['target_kv_peak_bytes']==row['draft_kv_peak_bytes']==expected_kv,'joint KV drift')
        audit_cuda(row['cuda'])
        demand(row['cuda']['allocated_bytes']>=baseline+static,'runtime static storage undercharged')
        extra=row['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']
        demand(extra==row['extra_cuda_peak_bytes']<=1024*2**20 and
               extra>=baseline-a['charged_baseline_bytes']+static+2*expected_kv,'runtime peak allowance/charging error')
        demand(type(row['host_controller_metadata_bytes']) is int and row['host_controller_metadata_bytes']>0,'missing controller metadata')
        traces[(row['document'],row['mode'])]=(rr,[{k:v for k,v in p.items() if k!='mode'} for p in policy])
    for d in [cal,*ds]:
        demand(traces[(d,'adaptive')]==traces[(d,'retained')],'cache affected precision/verification behavior')
        demand(traces[(d,'adaptive')]==traces[(d,'static')],'static cache affected numerical behavior')
    samples=list(read_rows(run/'resources.jsonl'))
    demand(samples and all(s['gpu_used']<=15000*2**20 and s['host_available']>=2048*2**20 for s in samples),'physical resource limit')
    resource=dict(samples=len(samples),peak_gpu_used=max(s['gpu_used'] for s in samples),
        min_host_available=min(s['host_available'] for s in samples),peak_rss=max(s['process']['rss'] for s in samples),passed=True,error=None)
    demand(json.loads((run/'completion.json').read_text(encoding='utf-8'))['resources']==resource,'resource summary differs')
    conditions={}
    for mode in ['target',*FROZEN['conditions']]:
        group=[r for r in rows if r['mode']==mode and not r['warmup']]
        wall=sum(r['wall_seconds'] for r in group)
        consumed=sum(r.get('consumed_draft_tokens',0) for r in group)
        attempted=sum(r['attempted'] for r in group)
        accepted=sum(r['accepted'] for r in group)
        h2d=sum(r.get('cache',{}).get('h2d_bytes',0) for r in group)
        precision=Counter()
        proxy=[]
        for r in group:
            if r.get('episode') in audits:
                precision.update(audits[r['episode']]['precisions'])
                proxy.extend(audits[r['episode']]['proxy_observations'])
        conditions[mode]=dict(tokens=sum(len(r['ids']) for r in group),wall_seconds=wall,
            target_seconds=sum(r['target_seconds'] for r in group),prefill_seconds=sum(r['prefill_seconds'] for r in group),
            decode_seconds=sum(r['decode_seconds'] for r in group),attempted=attempted,accepted=accepted,
            acceptance=accepted/attempted if attempted else None,consumed_draft_tokens=consumed,
            h2d_bytes=h2d,h2d_per_consumed=h2d/consumed if consumed else None,
            hits=sum(r.get('cache',{}).get('hits',0) for r in group),precisions=dict(precision),
            proxy_count=len(proxy),proxy_underestimates=sum(p['observed']>p['predicted'] for p in proxy),
            d2h_bytes=sum(audits.get(r.get('episode'),{}).get('d2h_bytes',0) for r in group))
    return dict(source_commit=manifest['source_commit'],conditions=conditions,allocation=a,resources=resource,
        retained_vs_static_h2d_ratio=conditions['retained']['h2d_bytes']/conditions['static']['h2d_bytes'],
        max_extra_cuda_bytes=max(r.get('extra_cuda_peak_bytes',0) for r in rows),
        warmup_wall_seconds=sum(r['wall_seconds'] for r in rows if r['warmup']),
        all_candidate_ids_match=True,retained_behavior_identical=True,
        numerical_max_relative_l2=max(errs),local_ffn_relative_l2={m:dict(mean=sum(v)/len(v),maximum=max(v)) for m,v in local.items()},
        archived_bitplane_max_relative_l2=max(artifact_errors),
        H4_extrapolation=extrapolation,
        **decision(conditions))

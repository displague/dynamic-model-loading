"""Independent CPU replay: GP linear algebra, physical acquisition and mixed KV."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .debt_analysis import demand,read_rows,audit_timing,audit_cuda
from .refinement_analysis import SLAB_BYTES


def fit_replay(x,z,blocks):
    """Do not call the runtime GP implementation; use NumPy dense solves."""
    x,z=np.asarray(x,np.float64),np.asarray(z,np.float64)
    mean,scale=x.mean(0),np.maximum(x.std(0),1e-6)
    u=(x-mean)/scale
    geometry=np.linalg.cholesky(.8*u.T@u/len(u)+.2*np.eye(x.shape[1]))
    train=np.linalg.solve(geometry,u.T).T
    distance=((train[:,None]-train[None,:])**2).sum(-1)
    off=np.sort(distance[~np.eye(len(x),dtype=bool)])
    length2=max(float(off[(len(off)-1)//2]),1e-6)  # torch's lower median
    zm,zs=float(z.mean()),max(float(z.std()),.1)
    matrix=np.exp(-.5*distance/length2)+(.1+1e-8)*np.eye(len(x))
    alpha=np.linalg.solve(matrix,(z-zm)/zs)
    result=dict(xmean=mean,xscale=scale,geometry=geometry,train=train,length2=np.asarray(length2),
                zmean=np.asarray(zm),zscale=np.asarray(zs),factor=np.linalg.cholesky(matrix),alpha=alpha)
    def predict(v):
        t=np.linalg.solve(geometry,(np.asarray(v)-mean)/scale)
        k=np.exp(-.5*((train-t)**2).sum(-1)/length2)
        mu=zm+zs*(k@alpha)
        sd=zs*math.sqrt(max(1.1-k@np.linalg.solve(matrix,k),1e-12))
        return float(mu),float(sd)
    scores=[]
    constants=[]
    for bx,bz in blocks:
        predictions=[predict(v) for v in bx]
        scores.append(max((v-mu)/sd for v,(mu,sd) in zip(bz,predictions,strict=True)))
        constants.append(max((v-zm)/zs for v in bz))
    demand(len(blocks)==9,'expected nine document calibration blocks')
    result['q']=np.asarray([max(0.,max(scores)),max(0.,max(constants))])
    def bounds(v):
        mu,sd=predict(v)
        return dict(mean=mu,sd=sd,upper=float(mu+result['q'][0]*sd),constant=float(zm+result['q'][1]*zs))
    return result,bounds


def close(a,b,why):
    demand(np.allclose(a,b,rtol=1e-7,atol=1e-8) and np.isfinite(a).all() and np.isfinite(b).all(),why)


def audit_policy(policy,pages,mode,consumed,prefix,predictors=None,*,layers=28,slots=8,slab_bytes=SLAB_BYTES):
    demand(len(policy)==consumed*layers,'missing layer observations')
    events=iter(pages)
    stats=dict(h2d_bytes=0,hits=0,loads=0,evictions=0)
    entries={}
    precision=Counter()
    d2h=0
    for token in range(1,consumed+1):
        for layer in range(layers):
            r=policy[(token-1)*layers+layer]
            pre=token<=prefix
            demand((r['token'],r['layer'],r['mode'],r['phase'])==(token,layer,mode,'prefill' if pre else 'decode'),'phase/layer drift')
            feature_needed=not pre and mode in ('shadow','p8mean','p8gp','p8constant')
            estimate=None
            if feature_needed:
                demand(len(r['feature'])==18 and np.isfinite(r['feature']).all(),'invalid causal feature')
                if predictors is not None:
                    estimate=predictors[layer](r['feature'])
                    demand(set(r['estimate'])==set(estimate),'missing prediction')
                    for k,v in estimate.items():
                        close(v,r['estimate'][k],'GP prediction differs')
                else:
                    demand(r['estimate'] is None,'unfrozen training prediction')
            else:
                demand(r['feature'] is None and r['estimate'] is None,'unexpected controller work')
            if pre:
                promoted=mode!='q6'
            elif mode in ('shadow','p8d8'):
                promoted=True
            elif mode in ('q6','p8d6'):
                promoted=False
            else:
                demand(estimate is not None,'missing predictor')
                promoted=estimate[{'p8mean':'mean','p8gp':'upper','p8constant':'constant'}[mode]]>math.log(.06)
            demand(r['stages']==1+promoted,'acquisition decision differs')
            observed=feature_needed and promoted
            demand((type(r['observed_log']) in (int,float) and math.isfinite(r['observed_log'])) if observed else r['observed_log'] is None,'counterfactual label drift')
            copies=(72 if feature_needed else 0)+(4 if observed else 0)
            demand(r['d2h_bytes']==copies,'controller readback differs')
            d2h+=copies
            precision[(r['phase'],4+2*r['stages'])]+=1
            for stage in range(1+promoted):
                key=(layer,stage)
                if key in entries:
                    kind,slot='hit',entries[key]
                    stats['hits']+=1
                else:
                    kind='load'
                    slot=layer if stage==0 and layer<min(slots,layers) else slots
                    if slot!=slots:
                        entries[key]=slot
                    stats['loads']+=1
                    stats['h2d_bytes']+=slab_bytes
                demand(next(events,None)==dict(kind=kind,token=token,key=list(key),slot=slot,bytes=slab_bytes if kind=='load' else 0),'physical event differs')
    demand(next(events,None) is None,'extra physical event')
    return dict(cache=stats,d2h_bytes=d2h,precisions={f'{phase}_{bits}':n for (phase,bits),n in precision.items()})


def audit_kv(rows,prefix,consumed,rounds=None,*,bytes_per_token=57344):
    expected=[('step',prefix,prefix)]
    token=prefix
    if rounds is None:
        expected +=[('step',t,t) for t in range(prefix+1,consumed+1)]
    else:
        for r in rounds:
            for p in range(1,5):
                token+=1
                expected.append(('step',token,r['base']+p))
            if r['fallback'] is not None:
                expected.append(('crop',token,r['base']+r['accepted']))
                if r['stop_reason'] not in ('length','eos'):
                    token+=1
                    expected.append(('step',token,r['base']+r['accepted']+1))
        demand(token==consumed,'KV consumption drift')
    demand([(r['kind'],r['token'],r['length']) for r in rows]==expected,'KV event sequence differs')
    digest=rows[0]['sha256']
    demand(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'bad prefix fingerprint')
    demand(all(r['sha256']==digest and r['readback_bytes']==prefix*bytes_per_token for r in rows),'prefix mutation/readback charge')
    return sum(r['readback_bytes'] for r in rows)


def statistical_metrics(rows,predictors,means):
    metrics={k:dict(absolute_errors=[],blocks={},safe=0,false_safe=0) for k in ('mean','upper','constant')}
    constant_errors=[]
    for d,r in rows:
        prediction=predictors[r['layer']](r['feature'])
        z=r['observed_log']
        constant_errors.append(abs(z-means[r['layer']]))
        for k in metrics:
            m=metrics[k]
            m['absolute_errors'].append(abs(z-prediction[k]))
            key=(d,r['layer'])
            m['blocks'][key]=m['blocks'].get(key,True) and z<=prediction[k]
            if prediction[k]<=math.log(.06):
                m['safe']+=1
                m['false_safe']+=z>math.log(.06)
    return dict(rows=len(rows),constant_mean_mae=sum(constant_errors)/len(rows),
        predictions={k:dict(mae=sum(m['absolute_errors'])/len(rows),blocks=len(m['blocks']),
            covered_blocks=sum(m['blocks'].values()),block_coverage=sum(m['blocks'].values())/len(m['blocks']),
            safe=m['safe'],false_safe=m['false_safe']) for k,m in metrics.items()})


def decision(c,s):
    p=s['predictions']
    hcal=p['mean']['mae']<=.9*s['constant_mean_mae'] and p['upper']['block_coverage']>=.9
    hunc=p['upper']['false_safe']<p['mean']['false_safe'] and p['upper']['safe']>0
    hpre=c['p8d6']['acceptance']>=c['q6']['acceptance']+.1 and c['p8d6']['h2d_per_committed']<=.9*c['p8d8']['h2d_per_committed']
    hpolicy=(c['p8gp']['acceptance']>=c['p8d8']['acceptance']-.1 and
        c['p8gp']['h2d_per_committed']<=.9*c['p8d8']['h2d_per_committed'] and
        c['p8gp']['h2d_per_committed']<=.95*c['p8constant']['h2d_per_committed'] and
        c['p8gp']['acceptance']>=c['p8constant']['acceptance']-.1)
    return dict(hypotheses=dict(Hcal=hcal,Hunc=hunc,Hprefill=hpre,Hpolicy=hpolicy),
        decision='advance_to_new_protocol' if hpre or (hcal and hunc and hpolicy) else 'stop',
        full_suite_launched=False,native_admission_evaluated=False)


def analyze(run):
    import torch
    from safetensors.torch import load_file
    from .experiment import digest,load_corpus
    from .metrics import relative_l2
    from .provenance import verify_snapshot,frozen_environment
    from .fault_screen import require_supervisor
    from .fault_pager_analysis import audit_rounds
    from .bayes_screen import FROZEN,validate_config
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
    verify_snapshot(run,manifest,'configs/bayes-screen.json',cfg['protocol'],__file__)
    frozen_environment(manifest['environment'])
    demand(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','checkpoint parent drift')
    demand(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text(encoding='utf-8'))['checkpoint']['files'],'checkpoint differs')
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256'),('representation-files.json','representation_files_sha256')]:
        demand(digest(run/name)==cfg[key],'frozen input differs')
    parent_files=json.loads((run/'representation-files.json').read_text(encoding='utf-8'))
    demand(digest(run/'parent-mechanics.safetensors')==parent_files['mechanics.safetensors'],'parent mechanics drift')
    a=json.loads((run/'allocation.json').read_text(encoding='utf-8'))
    expected=dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        draft_host_parameters_bytes=4624220160,shared_bytes=0,charged_baseline_bytes=7725494272,
        resident_bytes=650280960,workspace_bytes=254607360,cache_pool_bytes=9*SLAB_BYTES,
        staging_bytes=SLAB_BYTES,host_increments_bytes=56*SLAB_BYTES,projection_cuda_bytes=1536*16*4,
        construction_h2d_bytes=650280960+1536*16*4)
    for k,v in expected.items():
        demand(type(a[k]) is int and a[k]==v,'allocation differs: '+k)
    for i in range(28):
        name=f'layer-{i:02}.safetensors'
        demand(digest(run/'constructed'/name)==a['constructed_sha256'][name],'constructed digest drift')
        # Stronger than tensor equality: unchanged serialization matches parent.
        demand(digest(run/'constructed'/name)==parent_files['constructed/'+name],'parent bitplane bytes drift')
    mech,parent=load_file(run/'mechanics.safetensors'),load_file(run/'parent-mechanics.safetensors')
    demand(set(mech)=={f'q8.{i}' for i in range(28)},'mechanics inventory differs')
    errors=[relative_l2(parent[k],v) for k,v in mech.items()]
    demand(max(errors)<=.01,'eight-bit numerical control fails')
    static=a['resident_bytes']+a['workspace_bytes']+a['cache_pool_bytes']+a['projection_cuda_bytes']
    baseline=a['baseline_cuda']['allocated_bytes']
    audit_cuda(a['baseline_cuda'])
    demand(baseline>=a['charged_baseline_bytes'] and a['baseline_cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']<=1024*2**20,'baseline/model-loading allowance')
    phases=list(read_rows(run/'phases.jsonl'))
    demand([p['phase'] for p in phases]==['construction','snapshot','mechanics']+['shadow']*17+['fit']+['shadow']*2,'phase inventory differs')
    for p in phases:
        audit_cuda(p['cuda'])
        demand(math.isfinite(p['wall_seconds']) and p['wall_seconds']>0,'invalid phase time')
        demand(p['cuda']['allocated_bytes']>=baseline+static and
            p['extra_cuda_peak_bytes']==p['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']<=1024*2**20,'phase CUDA charge')
    close(phases[2]['relative_l2'],[relative_l2(parent[f'q8.{i}'],mech[f'q8.{i}']) for i in range(28)],'mechanics receipt differs')
    corpus=load_corpus(run/'corpus.jsonl')
    cal=[r['id'] for r in corpus if r['split']=='calibration'][:17]
    ds=[r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    fitted=load_file(run/'gp.safetensors')
    fit=json.loads((run/'fit.json').read_text(encoding='utf-8'))
    demand(fit['fit_documents']==cal[:8] and fit['calibration_documents']==cal[8:] and fit['gp_sha256']==digest(run/'gp.safetensors'),'fit split/artifact drift')
    projection=(torch.randint(0,2,(1536,16),generator=torch.Generator().manual_seed(20260915)).float()*2-1)/4
    demand(torch.equal(fitted['projection'],projection),'projection drift')
    observations={}
    for i,p in enumerate([p for p in phases if p['phase']=='shadow']):
        d=([*cal,*ds])[i]
        demand((p['document'],p['folder'],p['split'])==(d,f'shadow-{i:02}','fit' if i<8 else 'calibration' if i<17 else 'diagnostic'),'shadow split order drift')
        observations[d]=list(read_rows(run/p['folder']/'policy.jsonl.gz'))
    def data(documents,layer):
        rows=[r for d in documents for r in observations[d] if r['layer']==layer and r['phase']=='decode']
        return [r['feature'] for r in rows],[r['observed_log'] for r in rows]
    models,predictors=[],[]
    expected_keys={'projection'}
    for i in range(28):
        model,predict=fit_replay(*data(cal[:8],i),[data([d],i) for d in cal[8:]])
        for k,v in model.items():
            key=f'{i}.{k}'
            expected_keys.add(key)
            close(v,fitted[key].numpy(),'fitted tensor differs: '+key)
        models.append(model); predictors.append(predict)
    demand(set(fitted)==expected_keys,'GP tensor inventory differs')
    demand(fit['host_model_tensor_bytes']==sum(t.numel()*t.element_size() for k,t in fitted.items() if k!='projection') and fit['host_model_metadata_bytes']>0 and fit['observation_metadata_bytes']>0 and fit['projection_d2h_bytes']==1536*16*4,'fit memory/copy charge')
    shadow_stats=[]
    for i,p in enumerate([p for p in phases if p['phase']=='shadow']):
        folder=run/p['folder']
        report=audit_policy(observations[p['document']],list(read_rows(folder/'pages.jsonl.gz')),'shadow',12,8,predictors if i>=17 else None)
        demand(report['cache']==p['cache'] and p['consumed_draft_tokens']==12 and p['kv_bytes']==12*57344,'shadow counters differ')
        report['kv_readback_bytes']=audit_kv(list(read_rows(folder/'kv.jsonl.gz')),8,12)
        shadow_stats.append(report)
    diagnostic=[(d,r) for d in ds for r in observations[d] if r['phase']=='decode']
    statistical=statistical_metrics(diagnostic,predictors,[float(m['zmean']) for m in models])
    rows=list(read_rows(run/'episodes.jsonl'))
    order=[('target',d,d==cal[0]) for d in [cal[0],*ds]]+[(m,cal[0],True) for m in cfg['conditions']]
    order +=[(m,d,False) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
    demand([(r['mode'],r['document'],r['warmup']) for r in rows]==order,'episode order differs')
    refs={r['document']:r for r in rows[:3]}
    audits={}
    for i,r in enumerate(rows):
        audit_timing(r); audit_cuda(r['cuda'])
        if i:
            demand(r['charged_started_monotonic']>=rows[i-1]['charged_finished_monotonic'],'overlap')
        prefix=2 if r['warmup'] else 8
        demand(r['extra_cuda_peak_bytes']==r['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']<=1024*2**20,'episode CUDA allowance')
        demand(r['cuda']['allocated_bytes']>=baseline+static,'undercharged resident memory')
        if r['mode']=='target':
            demand(r['target_kv_peak_bytes']==57344*(prefix+len(r['ids'])-1) and r['draft_kv_peak_bytes']==0,'scalar reference KV drift')
        else:
            folder=run/f'episode-{i-3:02}'
            demand(json.loads((folder/'episode.json').read_text(encoding='utf-8'))==r,'episode receipt drift')
            ref=refs[r['document']]
            demand(r['reference_match'] and r['ids']==ref['ids'] and r['stop_reason']==ref['stop_reason'],'reference mismatch')
            consumed=audit_rounds(folder/'rounds.jsonl',r,prefix_tokens=prefix,generation_tokens=1 if r['warmup'] else 4)//28
            demand(consumed==r['consumed_draft_tokens'],'consumption counter differs')
            rr=list(read_rows(folder/'rounds.jsonl'))
            report=audit_policy(list(read_rows(folder/'policy.jsonl.gz')),list(read_rows(folder/'pages.jsonl.gz')),r['mode'],consumed,prefix,predictors)
            report['kv_readback_bytes']=audit_kv(list(read_rows(folder/'kv.jsonl.gz')),prefix,consumed,rr)
            demand(report['cache']==r['cache'],'episode physical counters differ')
            expected_kv=57344*max([prefix]+[v['base']+4 for v in rr])
            demand(r['target_kv_peak_bytes']==r['draft_kv_peak_bytes']==expected_kv,'joint KV differs')
            demand(r['host_controller_metadata_bytes']>0 and 0<=r['controller_seconds']<=r['wall_seconds'],'controller cost differs')
            audits[r['episode']]=report
        demand(r['extra_cuda_peak_bytes']>=baseline-a['charged_baseline_bytes']+static+r['target_kv_peak_bytes']+r['draft_kv_peak_bytes'],'peak memory lower bound')
    samples=list(read_rows(run/'resources.jsonl'))
    demand(samples and all(s['gpu_used']<=15000*2**20 and s['host_available']>=2048*2**20 for s in samples),'physical resource limit')
    resource=dict(samples=len(samples),peak_gpu_used=max(s['gpu_used'] for s in samples),min_host_available=min(s['host_available'] for s in samples),peak_rss=max(s['process']['rss'] for s in samples),passed=True,error=None)
    demand(json.loads((run/'completion.json').read_text(encoding='utf-8'))['resources']==resource,'resource summary differs')
    conditions={}
    for mode in ['target',*FROZEN['conditions']]:
        group=[r for r in rows if r['mode']==mode and not r['warmup']]
        tokens=sum(len(r['ids']) for r in group)
        attempted=sum(r['attempted'] for r in group)
        accepted=sum(r['accepted'] for r in group)
        h2d=sum(r.get('cache',{}).get('h2d_bytes',0) for r in group)
        consumed=sum(r.get('consumed_draft_tokens',0) for r in group)
        precision=Counter()
        for r in group:
            precision.update(audits.get(r.get('episode'),{}).get('precisions',{}))
        conditions[mode]=dict(tokens=tokens,accepted=accepted,attempted=attempted,acceptance=accepted/attempted if attempted else None,
            consumed_draft_tokens=consumed,h2d_bytes=h2d,h2d_per_committed=h2d/tokens,h2d_per_consumed=h2d/consumed if consumed else None,
            precisions=dict(precision),**{k:sum(r.get(k,0) for r in group) for k in ('wall_seconds','prefill_seconds','decode_seconds','target_seconds','controller_seconds')},
            d2h_bytes=sum(audits.get(r.get('episode'),{}).get('d2h_bytes',0) for r in group),
            kv_readback_bytes=sum(audits.get(r.get('episode'),{}).get('kv_readback_bytes',0) for r in group))
    return dict(source_commit=manifest['source_commit'],conditions=conditions,statistical=statistical,
        allocation=a,fit=fit,resources=resource,numerical_max_relative_l2=max(errors),
        all_candidate_ids_match=True,prefix_kv_preserved=True,
        max_extra_cuda_bytes=max([a['baseline_cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']]+[p['extra_cuda_peak_bytes'] for p in [*phases,*rows]]),
        shadow_h2d_bytes=sum(p['cache']['h2d_bytes'] for p in shadow_stats),
        shadow_wall_seconds=sum(p['wall_seconds'] for p in phases if p['phase']=='shadow'),
        warmup_wall_seconds=sum(r['wall_seconds'] for r in rows if r['warmup']),**decision(conditions,statistical))

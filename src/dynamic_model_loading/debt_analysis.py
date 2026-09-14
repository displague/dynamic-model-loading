"""Model-free independent replay of correction acquisition and short-screen gates."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path

PAGE_BYTES=4718592


def demand(condition,message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as stream:
        for line in stream:
            yield json.loads(line)


def audit_timing(row):
    names=('charged_started_monotonic','started_monotonic','finished_monotonic',
           'cleanup_started_monotonic','charged_finished_monotonic')
    times=[row[k] for k in names]
    demand(all(math.isfinite(t) for t in times) and times==sorted(times),'invalid charged interval')
    for k in ('wall_seconds','generation_wall_seconds','prefill_seconds','decode_seconds',
              'target_seconds','cleanup_seconds','wrapper_seconds'):
        demand(math.isfinite(row[k]) and row[k]>=0,'invalid timing component: '+k)
    demand(row['wall_seconds']==times[-1]-times[0] and row['wall_seconds']>0,'charged wall time differs')
    demand(row['cleanup_seconds']==times[-1]-times[-2],'cleanup interval differs')
    generation=row['generation_wall_seconds']
    demand(math.isclose(generation,row['prefill_seconds']+row['decode_seconds'],rel_tol=1e-12,abs_tol=1e-9),
           'generation timing components differ')
    demand(0<generation<=times[2]-times[1] and row['target_seconds']<=generation,'impossible generation timing')
    demand(math.isclose(row['wall_seconds'],generation+row['cleanup_seconds']+row['wrapper_seconds'],
                        rel_tol=1e-12,abs_tol=1e-9),'charged timing components differ')


def audit_cuda(receipt):
    demand(all(type(v) is int and v>=0 for v in receipt.values()),'invalid allocator bytes')
    a,r,pa,pr=(receipt[k] for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'))
    demand(a<=r<=pr and a<=pa<=pr,'inconsistent allocator receipt')


def audit_allocation(account):
    baseline=account['non_ffn_baseline_allocated']
    static=account['target_parameters_bytes']+account['draft_cuda_parameters_bytes']
    demand(account['charged_baseline_bytes']==static<=baseline,'invalid charged parameter baseline')
    audit_cuda(account['baseline_cuda'])
    demand(account['baseline_cuda']['allocated_bytes']==baseline,'baseline allocator drift')
    audit_cuda(account['construction_cuda'])
    minimum=account['resident_bytes']+account['workspace_bytes']
    demand(baseline-static+minimum<=768*2**20,'baseline overhead exhausted inference allowance')
    demand(account['construction_cuda']['allocated_bytes']>=baseline+minimum,'constructed allocation undercharged')
    demand(math.isfinite(account['construction_wall_seconds']) and account['construction_wall_seconds']>0,
           'invalid construction timing')


def audit_episode_memory(row,account,rounds):
    prefix=2 if row['warmup'] else 8
    peak_positions=max([prefix]+[r['base']+4 for r in rounds])
    expected_kv=57344*peak_positions
    demand(row['target_kv_peak_bytes']==row['draft_kv_peak_bytes']==expected_kv,'invalid KV accounting')
    demand(row['staging_bytes']==PAGE_BYTES,'staging omitted')
    audit_cuda(row['cuda'])
    minimum=account['resident_bytes']+account['workspace_bytes']
    baseline=account['non_ffn_baseline_allocated']
    demand(row['cuda']['allocated_bytes']>=baseline+minimum,'resident allocation undercharged')
    extra=row['cuda']['peak_allocated_bytes']-account['charged_baseline_bytes']
    page_peak=PAGE_BYTES if row['mode']!='base' and row['cache'].get('load',0) else 0
    demand(row['extra_cuda_peak_bytes']==extra and
           baseline-account['charged_baseline_bytes']+minimum+max(2*expected_kv,page_peak)
           <=extra<=768*2**20,'extra CUDA budget failed')


def policy_choices(row,*,pages=35,rank=16,maximum=4,cuda=True):
    mode,sketches,choices=row['mode'],row['sketches'],row['choices']
    demand(mode in ('dense_stream','base','fixed','debt','complete'),'unknown policy mode')
    chosen=[v['page'] for v in choices]
    demand(all(type(p) is int and 0<=p<pages for p in chosen) and len(set(chosen))==len(chosen),'invalid/repeated page choice')
    expected_d2h=0
    if mode in ('fixed','debt'):
        demand(len(sketches)==pages and all(len(v)==rank for v in sketches),'sketch dimensions differ')
        demand(all(math.isfinite(a) for v in sketches for a in v),'nonfinite sketch')
        norms=[math.fsum(a*a for a in v) for v in sketches]
        debt=[math.fsum(v[j] for v in sketches) for j in range(rank)]
        used=set()
        fixed=sorted(range(pages),key=lambda p:(-norms[p],p))[:maximum]
        demand(len(choices)<=maximum,'too many corrections')
        for i,choice in enumerate(choices):
            demand(len(choice['observed'])==rank and all(math.isfinite(a) for a in choice['observed']),'missing observed correction')
            if mode=='fixed':
                demand(choice['page']==fixed[i] and choice['predicted_gain'] is None,'fixed ranking changed')
            else:
                scores={p:2*math.fsum(debt[j]*sketches[p][j] for j in range(rank))-norms[p]
                        for p in range(pages) if p not in used}
                best=max(scores,key=lambda p:(scores[p],-p))
                demand(scores[best]>0 and choice['page']==best,'incorrect feedback-directed acquisition')
                demand(choice['predicted_gain']==scores[best],'predicted gain differs')
                debt=[a-b for a,b in zip(debt,choice['observed'],strict=True)]
            used.add(choice['page'])
        if mode=='fixed':
            demand(chosen==fixed and row['stop_gain'] is None,'fixed policy stopped early')
        elif len(choices)<maximum:
            remaining=[2*math.fsum(debt[j]*sketches[p][j] for j in range(rank))-norms[p]
                       for p in range(pages) if p not in used]
            stop=max(remaining) if remaining else 0.0
            demand(stop<=0 and row['stop_gain']==stop,'unjustified early stopping')
        else:
            demand(row['stop_gain'] is None,'unexpected stop score at page cap')
        expected_d2h=4*(pages*rank+len(choices)*rank) if cuda else 0
    else:
        demand(not sketches and row['stop_gain'] is None,'unexpected sketch work')
        demand(chosen==([] if mode=='base' else list(range(pages))),'incorrect full/base page set')
        demand(all(not c['observed'] and c['predicted_gain'] is None for c in choices),'unexpected acquisition metadata')
    demand(row['d2h_bytes']==expected_d2h,'controller D2H byte drift')
    return chosen


def audit_acquisition(policy_path,page_path,expected_layers):
    rows=list(read_rows(policy_path))
    demand([(r['mode'],r['layer']) for r in rows]==expected_layers,'layer execution order differs')
    requested=[]
    d2h=0
    for row in rows:
        requested.extend((row['layer'],p) for p in policy_choices(row))
        d2h+=row['d2h_bytes']
    stats=Counter()
    active=None
    observed=[]
    for event in read_rows(page_path):
        key=(event['key']['layer'],event['key']['page'])
        if event['outcome']=='load':
            demand(active is None and event['request']=='demand','overlapping or speculative acquisition')
            demand(event['bytes']==event['h2d_bytes']==event['used_bytes']==PAGE_BYTES,'wrong physical payload')
            demand(event['evicted']==[],'unexpected cache eviction')
            demand(math.isfinite(event['wall_ms']) and math.isfinite(event['cuda_ms']) and
                   event['wall_ms']>=0 and event['cuda_ms']>=0,'invalid copy timing')
            active=key
            observed.append(key)
            stats['h2d_bytes']+=PAGE_BYTES
            stats['demand_bytes']+=PAGE_BYTES
            stats['transfer_wall_ms']+=event['wall_ms']
            stats['transfer_cuda_ms']+=event['cuda_ms']
            stats['peak_payload_bytes']=PAGE_BYTES
        else:
            demand(event['outcome']=='release' and active==key and event['request']=='eager_release' and
                   event['bytes']==PAGE_BYTES and event['used_bytes']==0,'invalid immediate page release')
            active=None
        stats[event['outcome']]+=1
    demand(active is None and observed==requested,'policy/physical acquisition mismatch')
    return dict(stats),dict(layers=len(rows),pages=len(requested),d2h_bytes=d2h,
                           zero_fetch_layers=sum(not r['choices'] for r in rows))


def decision(conditions):
    d,b,f,s=(conditions[k] for k in ('debt','base','fixed','dense_stream'))
    demand(all(r['attempted']>0 and r['tokens']>0 and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0
               for r in (d,b,f,s)),'missing economics denominator')
    checks=dict(acceptance_at_least_half=2*d['accepted']>=d['attempted'],
        acceptance_gain_ten_points=10*(d['accepted']*b['attempted']-b['accepted']*d['attempted'])>=d['attempted']*b['attempted'],
        wall_five_percent_below_base=100*d['wall_seconds']*b['tokens']<=95*b['wall_seconds']*d['tokens'],
        wall_five_percent_below_fixed=100*d['wall_seconds']*f['tokens']<=95*f['wall_seconds']*d['tokens'],
        h2d_ten_percent_below_stream=10*d['h2d_bytes']*s['tokens']<=9*s['h2d_bytes']*d['tokens'],
        h2d_no_higher_than_fixed=d['h2d_bytes']*f['tokens']<=f['h2d_bytes']*d['tokens'])
    return dict(decision='eligible_for_expanded_protocol' if all(checks.values()) else 'stop',
                checks=checks,full_suite_launched=False,native_admission_evaluated=False)


def audit_constructed(run,account):
    import torch
    from safetensors.torch import load_file
    from .experiment import digest
    folder=run/'constructed'
    demand(set(account['constructed_sha256'])=={'projection.safetensors'}|{f'layer-{i:02}.safetensors' for i in range(28)},'constructed inventory differs')
    for name,sha in account['constructed_sha256'].items():
        demand(digest(folder/name)==sha,'constructed artifact differs')
    generator=torch.Generator(device='cpu').manual_seed(20260914)
    expected=(torch.randint(0,2,(1536,16),generator=generator).float()*2-1)/4
    projection=load_file(folder/'projection.safetensors')
    demand(set(projection)=={'projection'} and torch.equal(projection['projection'],expected),'output projection changed')
    index=load_file(run/'index.safetensors')
    resident=expected.numel()*4
    for i in range(28):
        data=load_file(folder/f'layer-{i:02}.safetensors')
        shapes=dict(basis=(1536,16),gate_error=(8960,16),up_error=(8960,16),
                    down_full=(8960,16),down_base=(8960,16))
        for name in ('gate','up','down'):
            shapes.update({name+'.codes':(8960,384),name+'.minimum':(8960,12),name+'.scale':(8960,12)})
        demand(set(data)==set(shapes),'constructed tensor inventory differs')
        for key,shape in shapes.items():
            t=data[key]
            demand(tuple(t.shape)==shape and t.dtype==(torch.uint8 if key.endswith('.codes') else torch.float32),'constructed shape/type drift')
            demand(torch.isfinite(t).all().item(),'nonfinite constructed tensor')
            if key.endswith('.scale'):
                demand(torch.all(t>=0).item(),'negative scale')
            resident+=t.numel()*t.element_size()
        basis=data['basis']
        demand(torch.allclose(basis.T@basis,torch.eye(16),atol=1e-5,rtol=1e-5),'nonorthonormal input basis')
        c=index[f'centroids.{i}']
        demand(torch.linalg.vector_norm(c-(c@basis)@basis.T)/torch.linalg.vector_norm(c)<1e-5,'basis changed calibration span')
    demand(account['resident_bytes']==resident==428343296,'resident representation byte drift')
    demand(account['workspace_bytes']==3*8960*1536*4+8960*384*5,'workspace byte drift')
    demand(account['construction_h2d_bytes']==4624220160+28*1536*16*4+1536*16*4,'construction H2D drift')
    demand(account['construction_d2h_bytes']==resident,'construction D2H drift')
    for k,v in dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
                    draft_host_parameters_bytes=4624220160,shared_bytes=0,catalogue_aliases_host=True).items():
        demand(account[k]==v,'model allocation drift: '+k)
    audit_allocation(account)


def analyze(run):
    import torch
    from safetensors.torch import load_file
    from .debt_screen import validate_config,FROZEN,CONFIG
    from .experiment import digest,load_corpus
    from .fault_pager_analysis import audit_rounds
    from .fault_screen import require_supervisor,full_logit_metrics
    from .metrics import relative_l2
    from .provenance import verify_snapshot,frozen_environment
    torch.set_num_threads(4)
    run=Path(run)
    require_supervisor(run)
    demand((run/'completion.json').is_file() and not (run/'failure.json').exists(),'incomplete debt screen')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/'+CONFIG.name,FROZEN['protocol'],__file__)
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    frozen_environment(manifest['environment'])
    env=manifest['environment']
    demand(env['device']=='cuda:0' and env['cpu_threads']==4 and not env['tf32_matmul'] and
           not env['tf32_cudnn'] and 'gpu' in env,'execution environment drift')
    demand(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','checkpoint parent drift')
    demand(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text(encoding='utf-8'))['checkpoint']['files'],'checkpoint inventory drift')
    files=json.loads((run/'files.json').read_text(encoding='utf-8'))
    demand(set(files)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p!=run/'files.json'},'file inventory differs')
    for name,sha in files.items():
        demand(digest(run/name)==sha,'raw hash differs: '+name)
    for name,key in [('index.safetensors','index_sha256'),('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        demand(digest(run/name)==cfg[key],'parent input changed')
    account=json.loads((run/'allocation.json').read_text(encoding='utf-8'))
    audit_constructed(run,account)
    corpus=load_corpus(run/'corpus.jsonl')
    ds=[r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    cal=next(r['id'] for r in corpus if r['split']=='calibration')
    rows=list(read_rows(run/'episodes.jsonl'))
    expected=[('target',cal,True),('target',ds[0],False),('target',ds[1],False)]
    expected += [(m,cal,True) for m in cfg['conditions']]
    expected += [(m,d,False) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
    demand([(r['mode'],r['document'],r['warmup']) for r in rows]==expected,'episode order/matrix drift')
    for i,row in enumerate(rows):
        audit_timing(row)
        if i:
            demand(row['charged_started_monotonic']>=rows[i-1]['charged_finished_monotonic'],'overlapping episodes')
        if row['mode']=='target':
            prefix=2 if row['warmup'] else 8
            demand(row['target_kv_peak_bytes']==57344*(prefix+len(row['ids'])-1) and
                   row['draft_kv_peak_bytes']==0,'invalid scalar-reference KV accounting')
    refs={r['document']:r for r in rows[:3]}
    counters={}
    for number,row in enumerate(rows[3:]):
        folder=run/f'episode-{number:02}'
        demand(row==json.loads((folder/'episode.json').read_text(encoding='utf-8')),'episode receipt drift')
        ref=refs[row['document']]
        demand(row['reference_match'] and row['ids']==ref['ids'] and row['stop_reason']==ref['stop_reason'],'reference mismatch')
        layers=audit_rounds(folder/'rounds.jsonl',row,prefix_tokens=2 if row['warmup'] else 8,
                            generation_tokens=4 if row['warmup'] else 8)
        stats,counts=audit_acquisition(folder/'policy.jsonl.gz',folder/'pages.jsonl.gz',
                                      [(row['mode'],i%28) for i in range(layers)])
        for key in set(stats)|set(row['cache']):
            demand(math.isclose(stats.get(key,0),row['cache'].get(key,0),rel_tol=1e-9,abs_tol=1e-9),'physical counter drift: '+key)
        audit_episode_memory(row,account,list(read_rows(folder/'rounds.jsonl')))
        counters[row['episode']]=counts
    expected_mechanics=[(m,i) for m in ('complete','base','fixed','debt') for i in range(28) for _ in range(2)]
    expected_mechanics += [('complete',i) for i in range(28) for _ in range(2)]
    audit_acquisition(run/'mechanics-policy.jsonl.gz',run/'mechanics-pages.jsonl.gz',expected_mechanics)
    tensors=load_file(run/'mechanics.safetensors')
    keys={f'{m}.{i}' for m in ('input','reference','complete','base','fixed','debt') for i in range(28)}|{'reference.logits','complete.logits'}
    demand(set(tensors)==keys,'numerical inventory differs')
    demand(all(tensors[k].shape==(1,2,1536) for k in keys if not k.endswith('logits')),'numerical FFN shape differs')
    errors={m:[relative_l2(tensors[f'reference.{i}'][:,j],tensors[f'{m}.{i}'][:,j]) for i in range(28) for j in range(2)]
            for m in ('complete','base','fixed','debt')}
    metrics=full_logit_metrics(tensors['reference.logits'],tensors['complete.logits'])
    passed=max(errors['complete'])<=.01 and metrics['logit_relative_l2']<=.01 and metrics['mean_kl_dense_to_candidate']<=.001
    demand(passed,'all-page reconstruction failed')
    demand(json.loads((run/'mechanics.json').read_text(encoding='utf-8'))==dict(passed=passed,ffn_relative_l2=errors,**metrics),'numerical receipt drift')
    samples=list(read_rows(run/'resources.jsonl'))
    demand(samples and all(r['gpu_used']<=15000*2**20 and r['host_available']>=2048*2**20 for r in samples),'joint resources failed')
    resource=dict(samples=len(samples),peak_gpu_used=max(r['gpu_used'] for r in samples),
        min_host_available=min(r['host_available'] for r in samples),peak_rss=max(r['process']['rss'] for r in samples),passed=True,error=None)
    demand(json.loads((run/'completion.json').read_text(encoding='utf-8'))['resources']==resource,'final resource receipt differs')
    conditions={}
    for mode in ['target',*FROZEN['conditions']]:
        group=[r for r in rows if r['mode']==mode and not r['warmup']]
        wall=sum(r['wall_seconds'] for r in group)
        count=sum(len(r['ids']) for r in group)
        attempted=sum(r['attempted'] for r in group)
        accepted=sum(r['accepted'] for r in group)
        h2d=sum(r.get('cache',{}).get('h2d_bytes',0) for r in group)
        conditions[mode]=dict(episodes=len(group),tokens=count,wall_seconds=wall,
            committed_tokens_per_second=count/wall,attempted=attempted,accepted=accepted,
            acceptance=accepted/attempted if attempted else None,h2d_bytes=h2d,
            h2d_per_committed=h2d/count,target_seconds=sum(r['target_seconds'] for r in group),
            controller_d2h_bytes=sum(counters.get(r.get('episode'),{}).get('d2h_bytes',0) for r in group),
            acquired_pages=sum(counters.get(r.get('episode'),{}).get('pages',0) for r in group),
            zero_fetch_layers=sum(counters.get(r.get('episode'),{}).get('zero_fetch_layers',0) for r in group))
    return dict(source_commit=manifest['source_commit'],conditions=conditions,allocation=account,resources=resource,
        max_extra_cuda_bytes=max(r['extra_cuda_peak_bytes'] for r in rows[3:]),
        warmup_wall_seconds=sum(r['wall_seconds'] for r in rows if r['warmup']),
        local_ffn_relative_l2={m:dict(mean=math.fsum(v)/len(v),maximum=max(v)) for m,v in errors.items()},
        numerical_logits=metrics,all_candidate_ids_match=True,
        note='Bounded acquisition-policy hypothesis test; reused diagnostics, not target-scale or native admission.',
        **decision(conditions))

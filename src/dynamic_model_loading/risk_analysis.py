"""Independent CPU replay of physical policy, KV, verification and byte ledgers."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from .debt_analysis import demand,read_rows,audit_cuda
from .fault_pager_analysis import audit_rounds
from .decision_acquisition import acquire

SLAB=10321920
PAGE=294912

def validate_execution(env):
    demand(env.get('device')=='cuda:0' and type(env.get('cpu_threads')) is int and env['cpu_threads']==4 and
        env.get('tf32_matmul') is False and env.get('tf32_cudnn') is False,'Frozen CUDA/thread/TF32 settings changed')

def validate_generation_timing(row,phase):
    fields=('started_monotonic','finished_monotonic','wall_seconds','prefill_seconds','decode_seconds','target_seconds')
    demand(all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) for k in fields),'Invalid generator timing fields')
    demand(phase['started_monotonic']<=row['started_monotonic']<=row['finished_monotonic']<=phase['finished_monotonic'] and
        0<row['wall_seconds']<=row['finished_monotonic']-row['started_monotonic'] and
        0<=row['target_seconds']<=row['wall_seconds'] and min(row['prefill_seconds'],row['decode_seconds'])>=0 and
        math.isclose(row['wall_seconds'],row['prefill_seconds']+row['decode_seconds'],abs_tol=1e-9),'Generator timing not reconciled/contained')

def generator_copies(rounds,output_count,*,target_only=False):
    operations={}; h2d=d2h=0
    def charge(direction,size,name):
        nonlocal h2d,d2h
        operations[name]=operations.get(name,0)+size
        if direction=='h2d': h2d+=size
        else: d2h+=size
    if target_only:
        charge('d2h',8*output_count,'target_argmax')
        if output_count>1: charge('h2d',8*(output_count-1),'target_next_token')
    else:
        for row in rounds:
            full=row['accepted']+int(row['fallback'] is not None)
            for direction,size,name in [('d2h',32,'draft_argmax'),('h2d',32,'draft_next_token'),
                ('h2d',32,'target_proposal_batch'),('h2d',32,'verifier_proposals'),('d2h',64,'verifier_inputs'),
                ('h2d',8*full,'verifier_committed_tensor'),('d2h',8*full,'verifier_committed_readback'),('d2h',32,'ledger_target_predictions')]:
                charge(direction,size,name)
            if row['fallback'] is not None and row['stop_reason'] not in ('eos','length'):
                charge('h2d',8,'fallback_token')
    return dict(h2d_bytes=h2d,d2h_bytes=d2h,operations=operations)

def audit_policy(row,index):
    policy=row['policy']; mode=row['condition']
    if row['prefill']:
        demand(policy is None,'Prefill used approximate acquisition')
        return list(range(35)),0.
    pages=policy['pages']; obs=policy['observations']
    demand(len(pages)==len(obs)==(35 if mode=='all35' else 17) and len(set(pages))==len(pages),'Invalid purchased pages')
    demand(all(type(p) is int and 0<=p<35 for p in pages) and all(math.isfinite(v) for v in obs),'Invalid page observation')
    if mode=='risk':
        axis=np.asarray(policy['axis'],dtype=np.float64)
        demand(axis.shape==(1536,) and np.isfinite(axis).all(),'Invalid current direction')
        prior=index['vector_prior']@axis
        cov=index['vector_covariance']*float(np.sum(index['feature_variance']*axis**2))
        counter=iter(zip(pages,obs))
        def observe(page):
            expected,value=next(counter)
            demand(page==expected,'Policy queried an unpurchased/incorrect page')
            return value
        replay=acquire(prior,cov,policy['base_margin'],observe,'risk',budget=17)
        demand(next(counter,None) is None,'Unused purchased observations')
        demand({k:v.tolist() for k,v in replay.items()}==policy['trace'],'Risk trace does not replay')
    else:
        demand(pages==list(range(len(pages))) and policy['axis'] is None and policy['trace'] is None,'Control changed')
    predicted=(policy['base_margin']+sum(obs))*policy['base_rms']/policy['final_rms']
    demand(all(math.isfinite(policy[k]) for k in ('base_margin','base_pair','final_pair','predicted_pair','base_rms','final_rms','acquisition_seconds','observation_seconds','nonprobe_acquisition_seconds')) and
        min(policy['base_rms'],policy['final_rms'],policy['acquisition_seconds'])>0,'Nonfinite policy receipt')
    demand(0<policy['observation_seconds']<=policy['acquisition_seconds'] and policy['nonprobe_acquisition_seconds']>=0 and
        math.isclose(policy['acquisition_seconds'],policy['observation_seconds']+policy['nonprobe_acquisition_seconds'],abs_tol=1e-12),'Acquisition subclock mismatch')
    demand(len(policy['base_ids'])==2 and len(set(policy['base_ids']))==2 and all(type(i) is int and 0<=i<151936 for i in policy['base_ids']),'Invalid base token pair')
    error=max(abs(policy['base_pair']-policy['base_margin']),abs(policy['final_pair']-predicted))
    demand(abs(predicted-policy['predicted_pair'])<1e-12 and
        abs(policy['base_pair']-policy['base_margin'])<=1e-4+1e-5*abs(policy['base_pair']) and
        abs(policy['final_pair']-predicted)<=1e-4+1e-5*abs(policy['final_pair']),'Physical pair-readout mismatch')
    return pages,error

def audit_calls(rows,rounds,prefix,condition,index):
    journal={0:hashlib.sha256(b'').hexdigest()}; calls=[]; crops=[]; length=0; previous_storage=0
    copy_bytes=0; max_error=0.; expected_events=[]; acquisition_seconds=0.
    for row in rows:
        if row['kind']=='crop':
            demand(row['after_call']==len(calls) and row['sha256']==journal.get(row['length']),'Crop journal drift')
            length=row['length']; journal={n:v for n,v in journal.items() if n<=length}
            demand(row['draft_kv_logical_bytes']==length*57344 and row['draft_kv_storage_bytes']==previous_storage and
                row['kv_fingerprint_d2h_bytes']==length*57344,'Cropped storage/copy accounting')
            crops.append((len(calls),length)); copy_bytes+=row['kv_fingerprint_d2h_bytes']; continue
        demand(row['kind']=='call','Unknown runtime event'); calls.append(row); call=len(calls)
        demand(row['call']==call and row['condition']==condition and row['prefill']==(call<=4),'Call phase/order drift')
        demand(all(type(row[k]) is int and 0<=row[k]<151936 for k in ('consumed_id','next_id')),'Invalid token readout')
        demand(row['base_length']==length and row['end_length']==length+1 and row['prior_sha']==journal[length] and
            row['prior_sha']==row['prior_after_sha'] and row['post_sha']==row['post_after_sha'] and
            len(row['post_sha'])==64,'Actual KV history drift')
        demand(row['kv_fingerprint_d2h_bytes']==(4*length+2)*57344 and
            row['draft_kv_logical_bytes']==row['draft_kv_storage_bytes']==(length+1)*57344,'Appended storage/copy accounting')
        length+=1; previous_storage=row['draft_kv_storage_bytes']; journal[length]=row['post_sha']
        selected,error=audit_policy(row,index); max_error=max(max_error,error)
        count=0 if row['prefill'] else len(selected)
        demand(row['axis_d2h_bytes']==(6144 if condition=='risk' and count else 0) and
            row['observation_d2h_bytes']==4*count and row['readout_scalar_d2h_bytes']==(52 if count else 16) and
            row['inherited_correction_d2h_bytes']==27*(8 if row['prefill'] else 4),'Readback accounting drift')
        demand(math.isfinite(row['wall_seconds']) and row['wall_seconds']>0,'Invalid call time')
        if count:
            acquisition_seconds+=row['policy']['acquisition_seconds']
            demand(row['policy']['acquisition_seconds']<=row['wall_seconds'],'Acquisition not contained')
        copy_bytes+=sum(row[k] for k in ('kv_fingerprint_d2h_bytes','axis_d2h_bytes','observation_d2h_bytes',
                                         'readout_scalar_d2h_bytes','inherited_correction_d2h_bytes'))
        for layer in range(27):
            for stage in range(2 if row['prefill'] else 1): expected_events.append(('layer',call,row['prefill'],[layer,stage],SLAB))
        for page in selected:
            for stage in range(3): expected_events.append(('page',call,row['prefill'],[page,stage],PAGE))
    demand([r['consumed_id'] for r in calls[:4]]==prefix,'Prefill tokens changed')
    cursor=4; next_id=calls[3]['next_id']; expected_crops=[]
    for round_ in rounds:
        for proposed in round_['proposed']:
            demand(next_id==proposed and cursor<len(calls),'Proposal not tied to recorded full readout')
            demand(calls[cursor]['consumed_id']==proposed,'Proposed trajectory differs')
            next_id=calls[cursor]['next_id']; cursor+=1
        if round_['fallback'] is not None:
            expected_crops.append((cursor,round_['base']+round_['accepted']))
            if round_['stop_reason'] not in ('eos','length'):
                demand(cursor<len(calls) and calls[cursor]['consumed_id']==round_['fallback'],'Fallback not consumed')
                next_id=calls[cursor]['next_id']; cursor+=1
    demand(cursor==len(calls) and crops==expected_crops,'Missing/extra calls or crops')
    return dict(calls=len(calls),crops=len(crops),explicit_d2h_bytes=copy_bytes,
        max_pair_error=max_error,expected_events=expected_events,acquisition_seconds=acquisition_seconds,
        call_seconds=sum(r['wall_seconds'] for r in calls))

def screen_decision(conditions,target_seconds):
    risk,fixed,high=(conditions[k] for k in ('risk','fixed','all35'))
    eligible=all(c['accepted']>0 for c in (risk,fixed,high))
    acquisition=(eligible and risk['acceptance']>=.5 and risk['acceptance']>=fixed['acceptance'] and
        risk['h2d_per_accepted']<=.95*min(fixed['h2d_per_accepted'],high['h2d_per_accepted']))
    runtime=risk['charged_wall_seconds']<=min(fixed['charged_wall_seconds'],high['charged_wall_seconds'])
    return dict(gates=dict(Hfaithfulness=True,Hacquisition=bool(acquisition),Hruntime=bool(runtime)),
        decision='eligible_for_new_expanded_protocol' if acquisition and runtime else 'stop_this_physical_candidate',
        beats_resident_target_clock=risk['charged_wall_seconds']<=target_seconds,
        native_admission_evaluated=False,full_suite_launched=False)

def analyze(run):
    from safetensors.numpy import load_file
    from .experiment import digest,load_corpus
    from .provenance import verify_snapshot,frozen_environment
    from .fault_screen import require_supervisor
    from .risk_screen import FROZEN,validate_config
    from .output_sensors_analysis import integer_identity,relative
    from .decision_field_analysis import validate_parent_reference
    run=Path(run); supervisor=require_supervisor(run)
    cfg=json.loads((run/'config.json').read_text()); validate_config(cfg)
    manifest=json.loads((run/'manifest.json').read_text()); frozen_environment(manifest['environment'])
    validate_execution(manifest['environment'])
    verify_snapshot(run,manifest,'configs/risk-screen.json',cfg['protocol'],__file__)
    demand(json.loads((run/'completion.json').read_text())==dict(complete=True,cuda_execution=True) and not (run/'failure.json').exists(),'Incomplete worker')
    inventory=json.loads((run/'files.json').read_text())
    demand(set(inventory)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}-{'files.json'},'Raw inventory drift')
    for name,sha in inventory.items(): demand(digest(run/name)==sha,'Raw bytes differ: '+name)
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256'),('representation-files.json','representation_files_sha256'),('index-parent.safetensors','index_parent_sha256')]:
        demand(digest(run/name)==cfg[key],'Frozen dependency changed')
    demand(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','Checkpoint binding')
    demand(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text())['checkpoint']['files'],'Checkpoint receipt')
    validate_parent_reference(run,json.loads((run/'representation-files.json').read_text()))
    index_all=load_file(run/'index-parent.safetensors'); index={k:index_all[k] for k in ('vector_prior','vector_covariance','feature_variance')}
    demand([v.shape for v in index.values()]==[(35,1536),(35,35),(1536,)] and all(np.isfinite(v).all() for v in index.values()),'Index shape/value')
    planes=load_file(run/'final-planes.safetensors')
    demand(set(planes)=={f'{name}.{j}' for name in ('low','next','parent4') for j in range(3)},'Plane inventory')
    for j in range(3): integer_identity(planes[f'parent4.{j}'],planes[f'low.{j}'],planes[f'next.{j}'],cfg['final_base_sha256'][j])
    mech=load_file(run/'mechanics.safetensors'); parent=load_file(run/'parent-mechanics.safetensors')
    demand(set(mech)=={f'q8.{i}' for i in range(28)},'Numerical control inventory')
    errors=[relative(parent[f'q8.{i}'],mech[f'q8.{i}']) for i in range(28)]
    demand(max(errors)<=.01,'Numerical control failed')
    a=json.loads((run/'allocation.json').read_text())
    expected=dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,draft_host_parameters_bytes=4624220160,
        charged_baseline_bytes=7725494272,shared_bytes=0,resident_bytes=639959040,workspace_bytes=254607360,
        layer_pool_bytes=20643840,page_pool_bytes=589824,layer_staging_bytes=10321920,page_staging_bytes=294912,
        original_host_increment_bytes=578027520,final_host_base4_bytes=20643840,new_host_plane_bytes=10321920,
        new_device_base2_bytes=10321920,index_host_array_bytes=452168)
    demand(all(type(a.get(k)) is int and a[k]==v for k,v in expected.items()),'Allocation mismatch')
    phases=list(read_rows(run/'phases.jsonl')); demand(len(phases)==13,'Missing phase receipt')
    peak=0
    for phase in phases:
        audit_cuda(phase['cuda']); extra=phase['cuda']['peak_allocated_bytes']-a['charged_baseline_bytes']
        demand(phase['extra_cuda_peak_bytes']==extra and 0<=extra<=1024*2**20,'CUDA budget/accounting')
        demand(all(math.isfinite(phase[k]) for k in ('wall_seconds','started_monotonic','finished_monotonic')) and
            phase['wall_seconds']==phase['finished_monotonic']-phase['started_monotonic']>0,'Invalid phase time')
        peak=max(peak,extra)
    static=sum(a[k] for k in ('resident_bytes','workspace_bytes','layer_pool_bytes','page_pool_bytes'))
    demand(phases[1]['cuda']['allocated_bytes']>=a['charged_baseline_bytes']+static,'Resident CUDA undercharge')
    construction=phases[1]
    demand(phases[0]['phase']=='model_loading' and construction['phase']=='construction' and phases[2]['phase']=='mechanics' and
        construction['equal_layers']==[True]*28 and construction['construction_h2d_bytes']==660602880 and
        construction['construction_d2h_bytes']==20643840 and construction['equality_d2h_bytes']==629637120 and
        construction['plane_snapshot_d2h_bytes']==10321920,'Setup receipts differ')
    demand(sum(p['wall_seconds'] for p in phases)<=supervisor['worker_wall_seconds'],'Phase time exceeds worker')
    demand(all(p['finished_monotonic']<=q['started_monotonic'] for p,q in zip(phases[:-1],phases[1:])),'Overlapping phase charges')
    corpus=[r['id'] for r in load_corpus(run/'corpus.jsonl') if r['split']=='diagnostic']; docs=[corpus[i] for i in cfg['diagnostic_indices']]
    phase_order=[('model_loading',None,None),('construction',None,None),('mechanics',None,None)]
    for di,doc in enumerate(docs):
        phase_order.append(('target_reference',doc,None))
        phase_order.extend(('episode',None,f'episode-{di}-{condition}') for condition in cfg['conditions'][di])
        phase_order.append(('target_repeat',doc,None))
    demand([(p['phase'],p.get('document'),p.get('episode')) for p in phases]==phase_order,'Phase sequence/document binding drift')
    tokens=json.loads((run/'token-ids.json').read_text())
    all_events=list(read_rows(run/'pages.jsonl.gz')); expected_events=[]
    for layer in range(27):
        for stage in range(2): expected_events.append(dict(episode='mechanics',call=None,prefill=True,cache='layer',kind='load',token=layer+1,key=[layer,stage],slot=1,bytes=SLAB))
    for page in range(35):
        for stage in range(3): expected_events.append(dict(episode='mechanics',call=None,prefill=True,cache='page',kind='load',token=28,key=[page,stage],slot=1,bytes=PAGE))
    episodes=[]; target_seconds=0.; references=[]
    for di,doc in enumerate(docs):
        reference=json.loads((run/f'reference-{di}.json').read_text()); repeat=json.loads((run/f'reference-repeat-{di}.json').read_text())
        demand(reference['document']==repeat['document']==doc and reference['ids']==repeat['ids'] and 0<len(reference['ids'])<=8,'Unstable target reference')
        demand(reference['attempted']==repeat['attempted']==0 and all(math.isfinite(r['wall_seconds']) and r['wall_seconds']>0 for r in (reference,repeat)),'Target timing/reference')
        for row,name in [(reference,'target_reference'),(repeat,'target_repeat')]:
            phase=next(p for p in phases if p['phase']==name and p.get('document')==doc)
            validate_generation_timing(row,phase)
            demand(row['generator_copies']==generator_copies([],len(row['ids']),target_only=True),'Target generator copies differ')
            if name=='target_reference': demand(phase['prefix_h2d_bytes']==32,'Input prefix transfer charge')
        target_seconds+=min(reference['wall_seconds'],repeat['wall_seconds']); references.append(reference['ids'])
        for condition in cfg['conditions'][di]:
            name=f'episode-{di}-{condition}'; folder=run/name
            episode=json.loads((folder/'episode.json').read_text())
            demand(episode['document']==doc and episode['condition']==condition and episode['ids']==reference['ids'] and episode['output_matches_reference'] is True,'Committed target output drift')
            consumed=audit_rounds(folder/'rounds.jsonl',episode,prefix_tokens=4,generation_tokens=8)//28
            rounds=list(read_rows(folder/'rounds.jsonl')); rows=list(read_rows(folder/'calls.jsonl.gz'))
            demand(episode['generator_copies']==generator_copies(rounds,len(episode['ids'])),'Generator/verifier copy charge drift')
            audit=audit_calls(rows,rounds,tokens[doc][:4],condition,index)
            demand(audit['calls']==consumed==episode['draft_calls'],'Consumed-call counter drift')
            counts={cache:0 for cache in ('layer','page')}
            for cache,call,prefill,key,size in audit.pop('expected_events'):
                counts[cache]+=1
                expected_events.append(dict(episode=name,call=call,prefill=prefill,cache=cache,kind='load',token=call,key=key,slot=1,bytes=size))
            for cache,size in [('layer',SLAB),('page',PAGE)]:
                demand(episode[cache+'_cache']==dict(h2d_bytes=counts[cache]*size,hits=0,loads=counts[cache],evictions=0),'Physical cache accounting drift')
            demand(all(math.isfinite(episode[k]) and episode[k]>0 for k in ('wall_seconds','charged_wall_seconds','prefill_seconds','decode_seconds')) and
                episode['wall_seconds']<=episode['charged_wall_seconds'] and audit['call_seconds']<=episode['charged_wall_seconds'] and
                math.isclose(episode['wall_seconds'],episode['prefill_seconds']+episode['decode_seconds'],abs_tol=1e-9),'Episode timing drift')
            demand(episode['charged_started_monotonic']<=episode['started_monotonic']<=episode['finished_monotonic']<=episode['charged_finished_monotonic'] and
                episode['charged_wall_seconds']==episode['charged_finished_monotonic']-episode['charged_started_monotonic'],'Charged interval mismatch')
            phase=next(p for p in phases if p.get('episode')==name)
            validate_generation_timing(episode,phase)
            demand(episode['charged_started_monotonic']==phase['started_monotonic'] and
                episode['charged_finished_monotonic']<=phase['finished_monotonic'],'Episode not inside phase')
            max_positions=max([4]+[r['base']+4 for r in rounds])
            demand(episode['target_kv_peak_bytes']==episode['draft_kv_peak_bytes']==max_positions*57344,'KV peaks differ')
            demand(phase['extra_cuda_peak_bytes']>=static+2*max_positions*57344,'KV CUDA peak undercharge')
            episodes.append(dict(**episode,**audit))
    demand(all_events==expected_events,'Physical transfer events missing, reordered or uncharged')
    resource=json.loads((run/'final-resources.json').read_text()); samples=list(read_rows(run/'resources.jsonl'))
    demand(resource['passed'] is True and resource['error'] is None and resource['samples']==len(samples)>0 and
        resource['peak_gpu_used']==max(r['gpu_used'] for r in samples)<=15000*2**20 and
        resource['min_host_available']==min(r['host_available'] for r in samples)>=2048*2**20 and
        resource['peak_rss']==max(r['process']['rss'] for r in samples),'Resource receipt drift')
    conditions={}
    for condition in ('fixed','risk','all35'):
        group=[e for e in episodes if e['condition']==condition]
        accepted=sum(e['accepted'] for e in group); attempted=sum(e['attempted'] for e in group)
        h2d=sum(e['layer_cache']['h2d_bytes']+e['page_cache']['h2d_bytes'] for e in group)
        conditions[condition]=dict(accepted=accepted,attempted=attempted,acceptance=accepted/attempted,
            h2d_bytes=h2d,h2d_per_accepted=h2d/accepted if accepted else None,
            charged_wall_seconds=sum(e['charged_wall_seconds'] for e in group),
            prefill_seconds=sum(e['prefill_seconds'] for e in group),decode_seconds=sum(e['decode_seconds'] for e in group),
            acquisition_seconds=sum(e['acquisition_seconds'] for e in group),
            runtime_explicit_d2h_bytes=sum(e['explicit_d2h_bytes'] for e in group),
            generator_h2d_bytes=sum(e['generator_copies']['h2d_bytes'] for e in group),
            generator_d2h_bytes=sum(e['generator_copies']['d2h_bytes'] for e in group),
            explicit_d2h_bytes=sum(e['explicit_d2h_bytes']+e['generator_copies']['d2h_bytes'] for e in group),draft_calls=sum(e['calls'] for e in group))
    return dict(**screen_decision(conditions,target_seconds),conditions=conditions,episodes=episodes,reference_ids=references,
        target_reference_best_sum_seconds=target_seconds,maximum_pair_error=max(e['max_pair_error'] for e in episodes),
        all_committed_outputs_match=True,mechanics_max_relative_l2=max(errors),allocation=a,resource=resource,
        extra_cuda_peak_bytes=peak,worker_page_payload_h2d_bytes=sum(r['bytes'] for r in all_events),
        construction_h2d_bytes=construction['construction_h2d_bytes'],source_commit=manifest['source_commit'],
        new_cuda_inference=True,full_argmax_readouts_used=True,predictor_fit_reused=True,
        new_to_this_course_document_indices=cfg['diagnostic_indices'],capacity_frontier_tested=False)

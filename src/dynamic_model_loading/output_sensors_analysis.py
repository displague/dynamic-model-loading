"""Independent CPU replay of final-FFN sensor observations and physical ledgers."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .decision_field_analysis import need,validate_parent_reference
from .fault_screen import require_supervisor
from .output_sensors import FROZEN,validate_config

LAYER_BYTES=10321920
PAGE_BYTES=294912


def close(actual,expected):
    return bool(np.all(np.abs(np.asarray(actual)-np.asarray(expected))<=1e-4+1e-5*np.abs(expected)))


def relative(reference,candidate):
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid numerical arrays')
    return float(np.linalg.norm(a-b))/max(float(np.linalg.norm(a)),1e-12)


def integer_identity(parent,low,next_,expected):
    need(hashlib.sha256(parent.tobytes()).hexdigest()==expected,'Final base not bound to frozen parent')
    def expand(value,bits):
        return np.stack([(value>>(bits*k))&(2**bits-1) for k in range(8//bits)],-1).reshape(value.shape[0],-1)
    need(np.array_equal(4*expand(low,2)+expand(next_,2),expand(parent,4)),'Final-plane integer identity')


def metrics(data,row):
    names={'base','high','fixed','oracle','target','h','residual','base_y','canonical_y','deltas','gain',
           'basis','headrows','oracle_basis','oracle_row','oracle_utilities','fixed_y','oracle_y'}
    if row['replay'] is not None:
        names.add('replay')
    need(set(data)==names,'Sensor tensor inventory')
    logits={'base','high','fixed','oracle','target','replay'}
    for key,value in data.items():
        shape=(151936,) if key in logits else (35,1536) if key=='deltas' else (2,1536) if key=='headrows' else (35,) if key=='oracle_utilities' else (1536,)
        need(value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all(),'Sensor tensor shape/dtype/value: '+key)
    a={key:value.astype(np.float64) for key,value in data.items()}
    u=int(a['base'].argmax())
    alternate=a['base'].copy(); alternate[u]=-np.inf
    v=int(alternate.argmax())
    need(row['base_ids']==[u,v] and row['high_id']==int(a['high'].argmax()),'Decision identity drift')
    need(close(a['basis'],(a['headrows'][0]-a['headrows'][1])*a['gain']),'Basis/readout mismatch')
    need(close(a['oracle_basis'],(a['oracle_row']-a['headrows'][0])*a['gain']),'Oracle basis mismatch')
    h,deltas,eps=a['h'],a['deltas'],row['epsilon']
    need(eps==1e-6,'RMS epsilon drift')
    need(close(h,a['residual']+a['base_y']),'Final residual/base reconstruction')
    for name,y in [('base',a['base_y']),('high',a['canonical_y']),('fixed',a['fixed_y']),('oracle',a['oracle_y'])]:
        hidden=a['residual']+y
        direct=a['headrows']@(a['gain']*hidden/np.sqrt(np.mean(hidden*hidden)+eps))
        need(close(a[name][u]-a[name][v],direct[0]-direct[1]),'Full-logit pair/readout mismatch')
    rms=np.sqrt(np.mean(h*h)+eps)
    oracle_utilities=deltas@a['oracle_basis']/rms
    need(close(a['oracle_utilities'],oracle_utilities),'Oracle utility reconstruction')
    selected=sorted(range(35),key=lambda p:(-float(data['oracle_utilities'][p]),p))[:17]
    need(row['oracle_pages']==selected,'Privileged selection drift')
    grouped=relative(a['canonical_y'],a['base_y']+deltas.sum(0))
    need(grouped<=.01 and close(row['grouped_relative_l2'],grouped),'Grouped FFN reconstruction')
    fixed_error=relative(a['fixed_y'],a['base_y']+deltas[:17].sum(0))
    oracle_error=relative(a['oracle_y'],a['base_y']+deltas[selected].sum(0))
    need(max(fixed_error,oracle_error)<=.01,'Partial FFN reconstruction')
    need(len(row['observations'])==35,'Missing page observations')
    projection_pass=True
    projection_max=0.
    for page,obs in enumerate(row['observations']):
        need(obs['page']==page,'Observation order')
        corrected=h+deltas[page]
        scale=np.sqrt(np.mean(corrected*corrected)+eps)
        expected=float(a['basis']@corrected/scale)
        direct=a['headrows']@(a['gain']*corrected/scale)
        need(close(obs['predicted'],expected) and close(obs['actual'],direct[0]-direct[1]),'Readout replay mismatch')
        need(close(obs['utility'],float(a['basis']@deltas[page]/rms)),'Page utility mismatch')
        error=abs(obs['predicted']-obs['actual'])
        projection_max=max(projection_max,error)
        projection_pass &= close(obs['predicted'],obs['actual'])
    ids={key:int(a[key].argmax()) for key in ('base','high','fixed','oracle','target')}
    base_wrong=ids['base']!=ids['high']
    result=dict(document=row['document'],split=row['split'],position=row['position'],ids=ids,
        base_wrong=base_wrong,oracle_repair=base_wrong and ids['oracle']==ids['high'],
        fixed_repair=base_wrong and ids['fixed']==ids['high'],
        oracle_new_error=not base_wrong and ids['oracle']!=ids['high'],
        fixed_new_error=not base_wrong and ids['fixed']!=ids['high'],
        high_target_match=ids['high']==ids['target'],base_target_match=ids['base']==ids['target'],
        high_outside_base_pair=ids['high'] not in (u,v),projection_pass=bool(projection_pass),
        max_projection_error=projection_max,grouped_relative_l2=grouped,
        fixed_grouped_relative_l2=fixed_error,oracle_grouped_relative_l2=oracle_error)
    if row['replay'] is not None:
        replay=row['replay']
        error=relative(a['high'],a['replay'])
        need(error<=.01 and ids['high']==int(a['replay'].argmax()) and
             close(error,replay['relative_l2']) and replay['argmax_match'] is True,'Full-model replay mismatch')
    return result


def summarize(frames):
    return dict(positions=len(frames),**{key:sum(int(f[key]) for f in frames) for key in
        ('base_wrong','oracle_repair','fixed_repair','oracle_new_error','fixed_new_error',
         'high_target_match','base_target_match','high_outside_base_pair')},
        all_projection_checks_pass=all(f['projection_pass'] for f in frames),
        max_projection_error=max(f['max_projection_error'] for f in frames),
        max_grouped_relative_l2=max(f['grouped_relative_l2'] for f in frames))


def decision(stats,observation_mean,replay_mean):
    d=stats['diagnostic']
    gates=dict(Hobservation=all(v['all_projection_checks_pass'] for v in stats.values()) and observation_mean<replay_mean,
        Haction=d['base_wrong']>=4 and d['oracle_repair']>=2 and d['oracle_new_error']<=d['fixed_new_error'])
    return dict(gates=gates,decision='eligible_for_short_estimation' if all(gates.values()) else 'stop',
                full_suite_launched=False,native_admission_evaluated=False)


def validate_rows(rows,docs,tokens):
    need(len(rows)==16*len(docs),'Missing/extra frames')
    for index,(doc,split) in enumerate(docs):
        for pos in range(16):
            row=rows[16*index+pos]
            need((row['document'],row['split'],row['position'])==(doc,split,pos),'Frame trajectory/order')
            need(row['consumed_token']==tokens[doc][4+pos] and row['base_length']==4+pos and
                row['end_length']==5+pos and row['token_counter']==6+pos,'Token/KV length drift')
            need(row['prior_sha']==row['prior_after_sha'] and row['post_sha']==row['post_after_sha'] and
                all(len(row[k])==64 for k in ('prior_sha','post_sha')),'KV fingerprint drift')
            if pos:
                need(row['prior_sha']==rows[16*index+pos-1]['post_after_sha'],'KV history chain drift')
            need(row['kv_fingerprint_d2h_bytes']==2*(9+2*pos)*57344 and
                row['kv_logical_bytes']==row['kv_storage_bytes']==2*(5+pos)*57344,'KV copy/storage accounting')
            need(row['full_logit_d2h_bytes']==5*151936*4 and row['auxiliary_d2h_bytes']==12*1536*4+4 and
                row['delta_d2h_bytes']==35*1536*4 and row['correction_scalar_d2h_bytes']==27*4*(1+int(pos==0)),
                'Frame readback accounting')
            need(all(np.isfinite(row[k]) and row[k]>0 for k in ('base_seconds','target_seconds',
                'high_readout_seconds','oracle_selection_seconds','fixed_seconds','oracle_seconds')),'Frame timing missing')
            need(len(row['observations'])==35,'Missing observation')
            for page,obs in enumerate(row['observations']):
                need(obs['page']==page and obs['h2d_bytes']==3*PAGE_BYTES and obs['d2h_bytes']==1536*4+12 and
                    np.isfinite(obs['wall_seconds']) and obs['wall_seconds']>0,'Observation accounting')
            if pos==0:
                r=row['replay']
                need(r is not None and r['kv_sha']==row['post_sha'] and r['kv_d2h_bytes']==5*57344 and
                    r['logit_d2h_bytes']==151936*4 and np.isfinite(r['wall_seconds']) and r['wall_seconds']>0,'Replay accounting')
            else:
                need(row['replay'] is None,'Unexpected full replay')


def validate_pages(rows,docs,frames):
    expected=[]
    def layer(phase,doc,pos,action,token,count=27,stages=1):
        expected.extend((phase,doc,pos,action,'layer',token,[i,s],LAYER_BYTES) for i in range(count) for s in range(stages))
    def page(phase,doc,pos,action,token,selected):
        expected.extend((phase,doc,pos,action,'page',token,[i,s],PAGE_BYTES) for i in selected for s in range(3))
    for i in range(27):
        expected.extend(('mechanics',None,None,None,'layer',i+1,[i,s],LAYER_BYTES) for s in range(2))
    page('mechanics',None,None,None,28,range(35))
    for di,(doc,_) in enumerate(docs):
        for token in range(1,5):
            layer('prefill',doc,None,None,token,stages=2)
            page('prefill',doc,None,None,token,range(35))
        for pos in range(16):
            token=5 if pos==0 else 6+pos
            layer('decode',doc,pos,'base',token)
            page('decode',doc,pos,'observe',token,range(35))
            page('decode',doc,pos,'fixed',token,range(17))
            page('decode',doc,pos,'oracle',token,frames[16*di+pos]['oracle_pages'])
            if pos==0:
                layer('decode',doc,pos,'full_replay',6)
                page('decode',doc,pos,'full_replay',6,range(35))
    need(len(rows)==len(expected),'Page event count drift')
    for r,e in zip(rows,expected,strict=True):
        need((r['phase'],r['document'],r['position'],r['action'],r['cache'],r['token'],r['key'],r['bytes'])==e and
            r['kind']=='load' and r['slot']==1,'Physical page event drift')
    return {cache:dict(loads=sum(r['cache']==cache for r in rows),h2d_bytes=sum(r['bytes'] for r in rows if r['cache']==cache),hits=0)
            for cache in ('layer','page')}


def analyze(run):
    from safetensors.numpy import load_file
    from .experiment import digest,load_corpus
    from .fault_pager_analysis import read_rows
    from .provenance import verify_snapshot,frozen_environment
    run=Path(run)
    require_supervisor(run)
    need((run/'completion.json').is_file() and not (run/'failure.json').exists(),'Incomplete sensor worker')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/output-sensors.json',FROZEN['protocol'],__file__)
    frozen_environment(manifest['environment'])
    env=manifest['environment']
    need(env['device']=='cuda:0' and env['cpu_threads']==4 and not env['tf32_matmul'] and not env['tf32_cudnn'],'Environment drift')
    validate_config(json.loads((run/'config.json').read_text(encoding='utf-8')))
    inventory=json.loads((run/'files.json').read_text(encoding='utf-8'))
    need(set(inventory)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p!=run/'files.json'},'Raw inventory drift')
    for name,expected in inventory.items():
        need(digest(run/name)==expected,'Raw SHA drift: '+name)
    for name,key in [('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256'),('representation-files.json','representation_files_sha256')]:
        need(digest(run/name)==FROZEN[key],'Frozen artifact drift')
    validate_parent_reference(run,json.loads((run/'representation-files.json').read_text(encoding='utf-8')))
    need(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a','Checkpoint parent drift')
    need(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text())['checkpoint']['files'],'Checkpoint metadata drift')
    corpus=load_corpus(run/'corpus.jsonl')
    docs=[(r['id'],'fit') for r in corpus if r['split']=='calibration'][:6]
    docs += [(r['id'],'diagnostic') for r in corpus if r['split']=='diagnostic'][:2]
    tokens=json.loads((run/'token-ids.json').read_text())
    rows=list(read_rows(run/'frames.jsonl'))
    validate_rows(rows,docs,tokens)
    traffic=validate_pages(list(read_rows(run/'pages.jsonl.gz')),docs,rows)
    phases=list(read_rows(run/'phases.jsonl'))
    need([r['phase'] for r in phases]==['model_loading','construction','mechanics']+(['prefill']+['frame']*16)*8,'Phase inventory')
    construction=phases[1]
    need(construction['equal_layers']==[True]*28 and construction['integer_planes']==[True]*3 and
        construction['construction_h2d_bytes']==660602880 and construction['construction_d2h_bytes']==20643840 and
        construction['equality_d2h_bytes']==629637120 and construction['plane_snapshot_d2h_bytes']==10321920,'Construction accounting')
    account=json.loads((run/'allocation.json').read_text())
    expected=dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
        draft_host_parameters_bytes=4624220160,charged_baseline_bytes=7725494272,shared_bytes=0,
        resident_bytes=639959040,workspace_bytes=254607360,layer_pool_bytes=20643840,page_pool_bytes=589824,
        layer_staging_bytes=10321920,page_staging_bytes=294912,original_host_increment_bytes=578027520,
        final_host_base4_bytes=20643840,new_host_plane_bytes=10321920,new_device_base2_bytes=10321920)
    need(all(account[k]==v for k,v in expected.items()),'Allocation mismatch')
    need(all(r['extra_cuda_peak_bytes']==r['cuda']['peak_allocated_bytes']-account['charged_baseline_bytes'] and
        0<=r['extra_cuda_peak_bytes']<=1024*2**20 and 0<r['wall_seconds']<300 for r in phases),'Phase time/memory failure')
    mechanical=phases[2]
    need(mechanical['input_h2d_bytes']==mechanical['output_d2h_bytes']==28*1536*4 and
         mechanical['correction_scalar_d2h_bytes']==27*8,'Mechanical copy accounting')
    old,new=load_file(run/'parent-mechanics.safetensors'),load_file(run/'mechanics.safetensors')
    need(set(new)=={f'q8.{i}' for i in range(28)},'Mechanics inventory')
    errors=[]
    for key,value in new.items():
        need(value.shape==old[key].shape==(1,1,1536) and value.dtype==old[key].dtype==np.float32,'Mechanical tensor geometry')
        errors.append(relative(old[key],value))
    need(max(errors)<=.01 and close(mechanical['relative_l2'],errors),'Mechanical numerical gate')
    planes=load_file(run/'final-planes.safetensors')
    need(set(planes)=={f'{name}.{i}' for i in range(3) for name in ('low','next','parent4')},'Final-plane inventory')
    for j in range(3):
        need(all(planes[f'{name}.{j}'].dtype==np.uint8 and planes[f'{name}.{j}'].shape==(8960,768 if name=='parent4' else 384)
             for name in ('low','next','parent4')),'Final-plane dimensions')
        integer_identity(planes[f'parent4.{j}'],planes[f'low.{j}'],planes[f'next.{j}'],FROZEN['final_base_sha256'][j])
    frames=[]
    for di,(doc,split) in enumerate(docs):
        pre=phases[3+17*di]
        need(pre['document']==doc and pre['token_h2d_bytes']==160 and pre['kv_storage_bytes']==8*57344 and
             pre['correction_scalar_d2h_bytes']==4*27*8 and pre['kv_sha']==rows[16*di]['prior_sha'] and
             pre['kv_fingerprint_d2h_bytes']==4*57344,'Prefill accounting')
        for pos in range(16):
            phase=phases[4+17*di+pos]
            need((phase['document'],phase['split'],phase['position'])==(doc,split,pos),'Frame phase drift')
            row=rows[16*di+pos]
            measured=sum(row[k] for k in ('base_seconds','target_seconds','high_readout_seconds',
                'oracle_selection_seconds','fixed_seconds','oracle_seconds'))
            measured += sum(o['wall_seconds'] for o in row['observations'])+(row['replay']['wall_seconds'] if row['replay'] else 0)
            need(measured<=phase['wall_seconds'],'Frame time containment')
            frames.append(metrics(load_file(run/f'frame-{di:02}-{pos:02}'/'observations.safetensors'),row))
    stats={split:summarize([r for r in frames if r['split']==split]) for split in ('fit','diagnostic')}
    observation_mean=sum(o['wall_seconds'] for r in rows for o in r['observations'])/(128*35)
    replay_mean=sum(r['replay']['wall_seconds'] for r in rows if r['replay'])/8
    samples=list(read_rows(run/'resources.jsonl'))
    need(bool(samples) and all(r['gpu_used']<=15000*2**20 and r['host_available']>=2048*2**20 for r in samples),'Resource cap')
    resources=dict(samples=len(samples),peak_gpu_used=max(r['gpu_used'] for r in samples),
        min_host_available=min(r['host_available'] for r in samples),peak_rss=max(r['process']['rss'] for r in samples),passed=True,error=None)
    need(resources==json.loads((run/'final-resources.json').read_text()),'Resource receipt drift')
    d2h=construction['construction_d2h_bytes']+construction['equality_d2h_bytes']+construction['plane_snapshot_d2h_bytes']
    d2h += mechanical['output_d2h_bytes']+mechanical['correction_scalar_d2h_bytes']+8*(4*27*8+4*57344)
    for row in rows:
        d2h += row['kv_fingerprint_d2h_bytes']+row['full_logit_d2h_bytes']+row['auxiliary_d2h_bytes']+row['correction_scalar_d2h_bytes']
        d2h += sum(o['d2h_bytes'] for o in row['observations'])  # Includes delta vectors; do not add them twice.
        if row['replay']:
            d2h += row['replay']['kv_d2h_bytes']+row['replay']['logit_d2h_bytes']
    return dict(source_commit=manifest['source_commit'],statistics=stats,frames=frames,
        mean_observation_seconds=observation_mean,mean_full_replay_seconds=replay_mean,
        action_wall_seconds={k:sum(r[k] for r in rows) for k in ('base_seconds','target_seconds',
            'high_readout_seconds','oracle_selection_seconds','fixed_seconds','oracle_seconds')},
        traffic=traffic,allocation=account,resources=resources,max_ffn_relative_l2=max(errors),
        recorded_d2h_bytes=d2h,construction_h2d_bytes=construction['construction_h2d_bytes'],token_h2d_bytes=8*160,
        peak_extra_cuda_bytes=max(r['extra_cuda_peak_bytes'] for r in phases),
        **decision(stats,observation_mean,replay_mean),
        note='Controlled final-layer low-precision challenge; oracle sees all pages. No causal policy, total inference gain or capacity claim.')

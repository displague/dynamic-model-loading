"""Model-free audit of exact-zero acquisition, greedy histories and real bytes."""
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from safetensors.numpy import load_file
from .debt_analysis import demand, read_rows, audit_cuda
from .fault_screen import ROOT, require_supervisor
from .provenance import verify_snapshot, frozen_environment
from .specialist_analysis import audit_resources
from .specialist_screen import sha
from .sparse_screen import validate_config
from .sparse_down import active_extents


def compare(reference,candidate):
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    demand(a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all(),
           'Invalid full-logit comparison')
    rel=float(np.sqrt(np.max(np.sum((a-b)**2,axis=1)/np.maximum(np.sum(a*a,axis=1),1e-30))))
    same=bool(np.array_equal(a.argmax(1),b.argmax(1)))
    demand(rel<=1e-5 and same,'Numerical or argmax reference failure')
    return dict(relative_l2=rel,argmax_identical=same)


def bounded_cuda(receipt):
    audit_cuda(receipt)
    demand(receipt['peak_allocated_bytes']<=15000*2**20 and
           receipt['peak_reserved_bytes']<=15000*2**20,'Allocator peak exceeds GPU cap')


def audit_pages(pages,tensors,calls,episode):
    mode=episode['condition']; count=len(calls)
    if mode not in ('stream','sparse'):
        demand(not pages,'Resident path unexpectedly acquired pages')
        return dict(h2d_bytes=0,d2h_bytes=0,extents=0,prefill_bytes=0,decode_bytes=0,
                    active_neurons=0,observed_neurons=0)
    demand(len(pages)==count*24,'Missing physical layer calls')
    stats=dict(h2d_bytes=0,d2h_bytes=0,extents=0,prefill_bytes=0,decode_bytes=0,
               active_neurons=0,observed_neurons=0)
    previous=calls[0]['started']
    for i,row in enumerate(pages):
        step,layer=divmod(i,24); c=calls[step]; n=len(c['input_ids'])
        demand((row['call'],row['layer'],row['tokens'],row['episode'],row['condition'])==
               (i+1,layer,n,episode['episode'],mode),'Physical call order changed')
        times=[row[k] for k in ('started','selection_finished','acquisition_finished','compute_finished')]
        demand(all(math.isfinite(t) for t in times) and times==sorted(times)
               and max(previous,c['started'])<=times[0]<=times[-1]<=c['finished'],'Physical clocks invalid')
        previous=times[-1]
        if mode=='sparse':
            packed=tensors[f'activity.{i+1}']
            demand(packed.shape==(n,1024) and packed.dtype==np.uint8,'Activity shape changed')
            activity=np.unpackbits(packed,axis=1).astype(bool)
            selected,extents=active_extents(activity,128)
            d2h=n*8192+1
            stats['active_neurons']+=int(activity.sum()); stats['observed_neurons']+=activity.size
        else:
            selected,extents=list(range(64)),[[0,8192]]; d2h=0
        h2d=sum(b-a for a,b in extents)*2048*4
        demand(row['pages']==selected and row['extents']==extents and
               row['weight_h2d_bytes']==h2d and row['activity_d2h_bytes']==d2h,
               'Acquisition decision/copy reconciliation failed')
        stats['h2d_bytes']+=h2d; stats['d2h_bytes']+=d2h; stats['extents']+=len(extents)
        stats['prefill_bytes' if step==0 else 'decode_bytes']+=h2d
    return stats


def decision(conditions):
    a,b=conditions['stream'],conditions['sparse']
    demand(a['h2d_bytes']>0 and a['wall_seconds']>0,'Missing baseline denominator')
    traffic=1-b['h2d_bytes']/a['h2d_bytes']; wall=1-b['wall_seconds']/a['wall_seconds']
    checks=dict(Htraffic=10*b['h2d_bytes']<=9*a['h2d_bytes'],
                Hruntime=20*b['wall_seconds']<=19*a['wall_seconds'])
    return dict(checks=checks,weight_traffic_saving=traffic,wall_saving=wall,
        decision='eligible_for_separately_frozen_followup' if all(checks.values()) else 'stop_128_row_exact_zero_candidate',
        full_suite_launched=False,native_admission_evaluated=False)


def analyze(output):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    output=Path(output); supervisor=require_supervisor(output)
    cfg=json.loads((output/'config.json').read_text()); validate_config(cfg)
    manifest=json.loads((output/'manifest.json').read_text())
    verify_snapshot(output,manifest,'configs/sparse-down-screen.json',cfg['protocol'],__file__)
    env=manifest['environment']; frozen_environment(env)
    demand(env['device']=='cuda:0' and env['cpu_threads']==4 and not env['tf32_matmul']
        and not env['tf32_cudnn'] and manifest['cuda_execution'] is True,'Execution settings changed')
    inventory=json.loads((output/'files.json').read_text())
    demand(set(inventory)=={p.relative_to(output).as_posix() for p in output.rglob('*')
        if p.is_file() and p!=output/'files.json'},'Raw inventory changed')
    for name,digest in inventory.items(): demand(sha(output/name)==digest,'Raw hash changed: '+name)
    parent=output/'artifact-manifest.json'
    original=subprocess.check_output(['git','show',manifest['source_commit']+':'+cfg['artifact_manifest']],cwd=ROOT)
    demand(original.replace(b'\r\n',b'\n')==parent.read_bytes().replace(b'\r\n',b'\n')
        and sha(parent)==manifest['artifact_manifest_sha256'],'Parent artifact provenance changed')
    expected={n:dict(bytes=cfg['artifact_bytes'][n],sha256=d)
        for n,d in json.loads(parent.read_text())['checkpoint_files'].items()}
    demand(manifest['artifacts']==expected,'Artifact inventory changed')
    path=Path(snapshot_download(cfg['repo'],revision=cfg['revision'],local_files_only=True))
    for name in ('config.json','merges.txt','vocab.json','tokenizer_config.json','special_tokens_map.json'):
        demand(sha(path/name)==expected[name]['sha256'],'Tokenizer artifact changed')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    prefixes=json.loads((output/'tokens.json').read_text())
    demand(prefixes==[tok.encode(p,add_special_tokens=False)[:16] for p in cfg['prompts']],
           'Independent tokenization mismatch')
    episodes=json.loads((output/'episodes.json').read_text())
    order=[(0,'resident'),(1,'resident'),(0,'stream'),(0,'sparse'),(1,'sparse'),(1,'stream'),(0,'repeat'),(1,'repeat')]
    demand([(r['document'],r['condition']) for r in episodes]==order,'Episode matrix changed')
    refs={}; conditions={}; controls=[]; allcopies=dict(h2d_bytes=0,d2h_bytes=0)
    completion=json.loads((output/'completion.json').read_text())
    samples=list(read_rows(output/'resources.jsonl')); audit_resources(samples,completion['resources'])
    last=samples[0]['monotonic']
    for i,row in enumerate(episodes):
        folder=output/f'episode-{i}'; mode=row['condition']; doc=row['document']
        demand(row['episode']==folder.name and row==json.loads((folder/'episode.json').read_text()),'Episode receipt drift')
        demand(last<=row['started']<row['finished']<=samples[-1]['monotonic'] and
            row['wall_seconds']==row['finished']-row['started'] and 0<row['ttft_seconds']<row['wall_seconds'],
            'Episode clocks changed')
        last=row['finished']; bounded_cuda(row['cuda'])
        calls=json.loads((folder/'calls.json').read_text()); tensors=load_file(folder/'tensors.safetensors')
        count=len(calls); demand(1<=count<=8,'Invalid output count')
        keys={f'logits.{j}' for j in range(count)}
        if mode=='sparse': keys|={f'activity.{j+1}' for j in range(count*24)}
        demand(set(tensors)==keys,'Tensor inventory changed')
        demand(row['raw_tensor_bytes']==sum(a.nbytes for a in tensors.values()),'Raw array accounting changed')
        ids=[]; copy_h2d=128+8*(count-1); copy_d2h=0; previous=row['started']
        for j,c in enumerate(calls):
            logits=tensors[f'logits.{j}']
            demand(logits.shape==(50272,) and logits.dtype==np.float32 and np.isfinite(logits).all(),'Invalid logits')
            demand(c['step']==j and c['input_ids']==(prefixes[doc] if j==0 else [ids[-1]]) and
                c['kv_length']==16+j and c['kv_bytes']==(16+j)*393216,'Input/KV contract failed')
            demand(previous<=c['started']<c['finished']<=row['finished'],'Forward timing changed')
            previous=c['finished']; ids.append(int(logits.argmax()))
            demand(c['token']==ids[-1] and (j==count-1 or ids[-1]!=2),'Greedy/EOS reconstruction failed')
            copy_d2h+=len(c['input_ids'])*8+50272*4
        stop='eos' if ids[-1]==2 else 'length'
        demand(ids==row['ids'] and row['stop_reason']==stop and (stop=='eos' or count==8),'Output receipt changed')
        demand(row['ttft_seconds']==calls[0]['finished']-row['started'],'TTFT changed')
        demand(row['explicit_copies']==dict(h2d_bytes=copy_h2d,d2h_bytes=copy_d2h),'Readout/token copy reconciliation failed')
        arrays=np.stack([tensors[f'logits.{j}'] for j in range(count)])
        if mode=='resident': refs[doc]=(ids,stop,arrays)
        else:
            ref,reason,a=refs[doc]; demand(ids==ref and stop==reason,'Reference output mismatch')
            controls.append(dict(episode=row['episode'],**compare(a,arrays)))
        pages=json.loads((folder/'pages.json').read_text()); stats=audit_pages(pages,tensors,calls,row)
        demand(row['weight_h2d_bytes']==stats['h2d_bytes'] and row['activity_d2h_bytes']==stats['d2h_bytes']
               and row['load_extents']==stats['extents'],'Episode acquisition totals changed')
        group=conditions.setdefault(mode,dict(tokens=0,wall_seconds=0.,ttft_seconds=0.,**{k:0 for k in stats}))
        group['tokens']+=count; group['wall_seconds']+=row['wall_seconds']; group['ttft_seconds']+=row['ttft_seconds']
        for k,v in stats.items(): group[k]+=v
        allcopies['h2d_bytes']+=copy_h2d+stats['h2d_bytes']; allcopies['d2h_bytes']+=copy_d2h+stats['d2h_bytes']
    allocation=json.loads((output/'allocation.json').read_text())
    constants=dict(parameters_bytes=5263032320,registered_buffers_bytes=0,construction_h2d_bytes=5263032320,
        host_down_bytes=1610612736,workspace_bytes=67108864,pinned_staging_bytes=67108864,
        construction_d2h_bytes=1610612736,restore_h2d_bytes=1610612736,
        hybrid_cuda_parameters_bytes=3652419584,host_weight_aliases=True)
    demand(all(allocation.get(k)==v for k,v in constants.items()),'Persistent allocation/copy accounting changed')
    for k in ('resident_cuda','hybrid_cuda','restore_cuda'): bounded_cuda(allocation[k])
    demand(allocation['resident_cuda']['allocated_bytes']>=constants['parameters_bytes'] and
        allocation['hybrid_cuda']['allocated_bytes']>=constants['hybrid_cuda_parameters_bytes']+67108864,
        'Persistent CUDA allocation undercharged')
    demand(all(math.isfinite(allocation[k]) and allocation[k]>0 for k in ('conversion_seconds','restore_seconds')),
           'Setup timing invalid')
    demand(completion['complete'] is True and completion['full_suite_launched'] is False and
        0<completion['worker_inner_seconds']<=supervisor['worker_wall_seconds'],'Invalid completion')
    return dict(source_commit=manifest['source_commit'],conditions=conditions,controls=controls,
        allocation=allocation,episode_explicit_copies=allcopies,resources=completion['resources'],
        max_kv_bytes=max((16+len(r['ids'])-1)*393216 for r in episodes),
        all_output_ids_match=True,**decision(conditions))

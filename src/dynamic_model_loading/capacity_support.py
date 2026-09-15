"""Bounded-residency controls and immutable external dense-reference receipts."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from safetensors.numpy import load_file
from .debt_analysis import demand, audit_cuda
from .fault_screen import ROOT
from .specialist_screen import sha

LIMIT=4800*2**20
REFERENCE_COMMIT='1e33e76e0aaa56121ed5d6824da7619c98220a51'
REFERENCE_SOURCE='03e1414686b1a5bbf1ca05c2312732b3ee64c5f8'
RECEIPT='results/row-packet-screen-20260915/packet-v033-archive.json'
REFERENCE=ROOT/'runs/row-packet-screen-20260915-v2/worker'
FILES=['manifest.json','tokens.json']+[f'episode-{i}/{n}' for i in (0,1)
    for n in ('episode.json','tensors.safetensors')]


def bounded(receipt):
    audit_cuda(receipt)
    demand(receipt['peak_allocated_bytes']<=LIMIT and receipt['peak_reserved_bytes']<=LIMIT,
           'Allocator exceeds declared capacity ceiling')


def parent_receipt():
    return subprocess.check_output(['git','show',REFERENCE_COMMIT+':'+RECEIPT],cwd=ROOT)


def copy_references(output):
    folder=Path(output)/'reference'; folder.mkdir()
    original=parent_receipt()
    demand((ROOT/RECEIPT).read_bytes()==original,'Published reference inventory changed')
    shutil.copyfile(ROOT/RECEIPT,folder/'archive.json')
    inventory=json.loads(original)['inventory']
    for name in FILES:
        source=REFERENCE/name
        demand(sha(source)==inventory['row-packet-screen-20260915-v2/worker/'+name],
               'Reference source hash changed')
        (folder/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,folder/name)


def load_references(output,prefixes):
    folder=Path(output)/'reference'
    demand((folder/'archive.json').read_bytes()==parent_receipt(),'External reference inventory drift')
    inventory=json.loads((folder/'archive.json').read_text())['inventory']
    demand({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(FILES+['archive.json']),
           'Reference file set changed')
    for name in FILES:
        demand(sha(folder/name)==inventory['row-packet-screen-20260915-v2/worker/'+name],'Reference hash changed')
    manifest=json.loads((folder/'manifest.json').read_text())
    demand(manifest['source_commit']==REFERENCE_SOURCE,'Reference source identity changed')
    demand(json.loads((folder/'tokens.json').read_text())==prefixes,'Reference prefix mismatch')
    refs={}
    for doc in (0,1):
        row=json.loads((folder/f'episode-{doc}/episode.json').read_text())
        tensors=load_file(folder/f'episode-{doc}/tensors.safetensors')
        demand(row['condition']=='resident' and row['document']==doc and row['stop_reason']=='length',
               'Not the pinned complete dense reference')
        demand(set(tensors)=={f'logits.{s}' for s in range(8)},'Reference tensor set changed')
        a=np.stack([tensors[f'logits.{s}'] for s in range(8)])
        demand(a.shape==(8,50272) and a.dtype==np.float32 and np.isfinite(a).all()
               and a.argmax(1).tolist()==row['ids'],'Reference output binding failed')
        refs[doc]=(row,a)
    return refs


def audit_allocation(a,env):
    expected=dict(parameters_bytes=5263032320,registered_buffers_bytes=0,cpu_first=True,
        cpu_loaded_parameter_bytes=5263032320,construction_h2d_bytes=3652419584,
        host_down_bytes=1610612736,workspace_bytes=67108864,pinned_staging_bytes=67108864,
        construction_d2h_bytes=0,hybrid_cuda_parameters_bytes=3652419584,host_weight_aliases=True,
        packet_host_bytes=67174400,packet_cuda_bytes=67174400,total_pinned_bytes=134283264,
        original_workspace_layout=True,weight_workspace_contiguous=True,workspace_strides=[1,8192],
        extra_host_linear_bytes=1610612736,direct_dense_host_contiguous=True,allocator_limit_bytes=LIMIT)
    demand(all(a.get(k)==v for k,v in expected.items()),'Capacity allocation/copy accounting changed')
    demand(not any(k in a for k in ('resident_cuda','restore_cuda','restore_h2d_bytes')),
           'Capacity worker unexpectedly used full GPU residency')
    demand(a['parameters_bytes']>LIMIT and a['allocator_fraction']==LIMIT/env['gpu']['total_bytes'],
           'Capacity exclusion or allocator fraction changed')
    for k in ('before_residency_cuda','hybrid_cuda'): bounded(a[k])
    demand(a['before_residency_cuda']['allocated_bytes']==0,'CPU-first path has hidden CUDA allocation')
    demand(a['hybrid_cuda']['allocated_bytes']>=3652419584+67108864+67174400,'Missing resident scratch')
    demand(all(math.isfinite(a[k]) and a[k]>0 for k in ('conversion_seconds','setup_seconds'))
           and a['setup_seconds']>=a['conversion_seconds'],'Construction clock invalid')

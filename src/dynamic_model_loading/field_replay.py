"""Frozen, CPU-only replay infrastructure for page-field hypothesis screens."""
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from .fault_screen import ROOT,write,require_supervisor
from .provenance import committed_inputs,verify_snapshot

PARENT=ROOT/'runs/output-sensors-20260915-v1/worker'
PARENT_SHA='ea879f8fb8389be7d2df2a299c608acbf8c7285252b0b345c6701812d430f8c7'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def parent_data():
    if sha(PARENT/'files.json')!=PARENT_SHA:
        raise ValueError('Published v0.26 inventory changed')
    inventory=json.loads((PARENT/'files.json').read_text())
    rows=[]
    if sha(PARENT/'frames.jsonl')!=inventory['frames.jsonl']:
        raise ValueError('Parent frames changed')
    frames=[json.loads(line) for line in (PARENT/'frames.jsonl').read_text().splitlines()]
    if len(frames)!=128 or [r['split'] for r in frames]!=['fit']*96+['diagnostic']*32:
        raise ValueError('Frozen document split changed')
    for index,row in enumerate(frames):
        name=f'frame-{index//16:02d}-{index%16:02d}/observations.safetensors'
        if sha(PARENT/name)!=inventory[name]:
            raise ValueError('Published sensor tensor changed')
        data=load_file(PARENT/name)
        h=data['h'].astype(np.float64)
        axis=data['basis'].astype(np.float64)/np.sqrt(np.mean(h*h)+row['epsilon'])
        delta=data['deltas'].astype(np.float64)
        rows.append(dict(z=delta@axis,axis=axis,delta=delta,margin=np.array(axis@h),
                         ids=np.array([int(data[k].argmax()) for k in ('base','high','fixed','oracle','target')],dtype=np.int64)))
    return {key:np.stack([r[key] for r in rows]) for key in rows[0]}


def begin(output,name,config):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    path=ROOT/f'configs/{name}.json'
    actual=json.loads(path.read_text())
    if actual!=config:
        raise ValueError('Frozen field-screen configuration changed')
    manifest,protocol=committed_inputs(__file__,path,config['protocol'])
    shutil.copyfile(path,output/'config.json')
    shutil.copyfile(protocol,output/'protocol.md')
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    import importlib.metadata,psutil
    manifest.update(parent_inventory_sha256=PARENT_SHA,python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ('numpy','safetensors','torch','transformers')},
        cuda_execution=False,process_rss_start=psutil.Process().memory_info().rss)
    write(output/'manifest.json',manifest)
    return parent_data()


def finish(output,inputs,predictions,summary,elapsed):
    import psutil
    output=Path(output)
    save_file(inputs,output/'inputs.safetensors')
    save_file(predictions,output/'predictions.safetensors')
    write(output/'calculation.json',summary)
    write(output/'completion.json',dict(complete=True,calculation_seconds=elapsed,
        process_peak_rss=psutil.Process().memory_info().peak_wset,
        input_tensor_bytes=sum(a.nbytes for a in inputs.values()),
        prediction_tensor_bytes=sum(a.nbytes for a in predictions.values()),new_inference=False))
    write(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


def validate_completion(receipt,inputs,predictions,supervisor):
    sizes=[sum(a.nbytes for a in arrays.values()) for arrays in (inputs,predictions)]
    if (receipt.get('complete') is not True or receipt.get('new_inference') is not False or
        receipt.get('input_tensor_bytes')!=sizes[0] or receipt.get('prediction_tensor_bytes')!=sizes[1] or
        not np.isfinite(receipt.get('calculation_seconds',np.nan)) or
        not 0<receipt['calculation_seconds']<=supervisor['worker_wall_seconds'] or
        not isinstance(receipt.get('process_peak_rss'),int) or receipt['process_peak_rss']<sum(sizes)):
        raise ValueError('Incomplete or inconsistent CPU resource receipt')


def audit(output,name,config,recompute,summarize):
    output=Path(output)
    supervisor=require_supervisor(output)
    manifest=json.loads((output/'manifest.json').read_text())
    verify_snapshot(output,manifest,f'configs/{name}.json',config['protocol'],__file__)
    if json.loads((output/'config.json').read_text())!=config or manifest['parent_inventory_sha256']!=PARENT_SHA:
        raise ValueError('Configuration or parent binding changed')
    if (manifest['python'].split()[0]!='3.14.3' or manifest['packages']['torch']!='2.10.0+cu130' or
        manifest['packages']['transformers']!='5.13.1' or manifest['cuda_execution'] is not False):
        raise ValueError('Baseline environment changed')
    inventory=json.loads((output/'files.json').read_text())
    if set(inventory)!={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='files.json'}:
        raise ValueError('Inventory changed')
    for name_,digest in inventory.items():
        if sha(output/name_)!=digest:
            raise ValueError('Raw receipt changed: '+name_)
    inputs=load_file(output/'inputs.safetensors')
    parent=parent_data()
    if inputs.keys()!=parent.keys() or any(not np.array_equal(inputs[k],parent[k]) for k in inputs):
        raise ValueError('Derived observations differ from published parent')
    stored=load_file(output/'predictions.safetensors')
    resources=json.loads((output/'completion.json').read_text())
    validate_completion(resources,inputs,stored,supervisor)
    replay=recompute(inputs)
    if stored.keys()!=replay.keys() or any(not np.array_equal(stored[k],replay[k]) for k in stored):
        raise ValueError('Prediction replay differs')
    result=summarize(inputs,replay)
    if result!=json.loads((output/'calculation.json').read_text()):
        raise ValueError('Summary replay differs')
    return dict(**result,source_commit=manifest['source_commit'],replay_exact=True,
        resources=resources)

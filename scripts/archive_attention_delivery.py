"""Archive all attempts and source snapshots, then restore and reproduce both stages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def create(project,candidate,archive,inventory,restore,receipt,analysis):
    for p in [archive,inventory,restore,receipt]:
        if p.exists(): raise ValueError('output already exists: '+str(p))
    files={}
    def add(path,name):
        if name in files: raise ValueError('duplicate archive member')
        files[name]=path
    groups=[('stock-placement-20260913','stock-placement-source-v1','original-source'),
            ('fixed-attention-20260913','fixed-attention-source-v1','fixed-initial-source'),
            ('fixed-attention-corrected-20260913','fixed-attention-source-v2','fixed-source')]
    heads={}
    for raw,source,dest in groups:
        run_root=project/'runs'/raw; src=project/'runs'/source
        if subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True).strip(): raise ValueError('dirty source')
        heads[dest]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
        manifests=list(run_root.glob('*/manifest.json'))
        identities=json.loads(manifests[0].read_text(encoding='utf-8'))['source_sha256']
        for name,digest in identities.items():
            if sha(src/name)!=digest: raise ValueError('source snapshot identity')
            add(src/name,dest+'/'+name)
        for path in sorted(run_root.rglob('*')):
            if path.is_file(): add(path,'runs/'+raw+'/'+path.relative_to(run_root).as_posix())
    for name in ['analyze_attention_delivery.py','archive_attention_delivery.py']:
        add(candidate/'scripts'/name,'analysis-code/'+name)
    native=project/'runs/llama-b10919-source'
    native_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()
    if native_head!='d3146f2b56c2db4711ac8391871c9e529d1946d7' or subprocess.check_output(
            ['git','status','--porcelain'],cwd=native,text=True).strip(): raise ValueError('native source identity')
    for name in ['common/common.cpp','common/common.h','common/speculative.cpp','common/arg.cpp',
                 'tools/server/server-context.cpp','tools/server/server-task.cpp','tools/server/server-common.h',
                 'src/llama-model.cpp','src/llama-model-loader.cpp','ggml/src/ggml-backend.cpp','ggml/src/ggml-cuda/ggml-cuda.cu']:
        add(native/name,'native-source/'+name)
    for pattern in ['placement-protocol-review-*','fixed-attention-protocol-review-*','fixed-attention-copy-review-*']:
        for path in sorted((project/'runs').glob(pattern)):
            if path.is_file(): add(path,'helpers/'+path.name)
    for name in ['run-placement-review.py','test-continuing-candidate.py','stock-publication-review-schema.json','collect-placement-results.py']:
        add(project/'runs'/name,'helpers/'+name)
    add(analysis,'analysis.json')
    add(project/'runs/placement-v0170-environment.json','environment.json')
    entries=[{'path':name,'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(files.items())]
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()): z.write(path,name)
    inv={'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
         'source_heads':heads,'native_source_head':native_head,'entries':entries}
    inventory.write_text(json.dumps(inv,indent=2)+'\n',encoding='utf-8',newline='\n')
    restore.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=[e['path'] for e in entries]: raise ValueError('archive member order/set')
        for name in z.namelist():
            if not (restore/name).resolve().is_relative_to(restore.resolve()): raise ValueError('unsafe member')
        z.extractall(restore)
    for e in entries:
        path=restore/e['path']
        if path.stat().st_size!=e['bytes'] or sha(path)!=e['sha256']: raise ValueError('restoration identity')
    reproduced=restore/'reproduced-analysis.json'
    command=[sys.executable,str(restore/'analysis-code/analyze_attention_delivery.py'),
        '--source',str(restore/'original-source'),'--runs',str(restore/'runs/stock-placement-20260913'),
        '--initial-source',str(restore/'fixed-initial-source'),'--initial-runs',str(restore/'runs/fixed-attention-20260913'),
        '--fixed-source',str(restore/'fixed-source'),'--fixed-runs',str(restore/'runs/fixed-attention-corrected-20260913'),
        '--output',str(reproduced)]
    subprocess.run(command,check=True)
    if reproduced.read_bytes()!=analysis.read_bytes(): raise ValueError('restored analysis differs')
    result={'archive_sha256':sha(archive),'inventory_sha256':sha(inventory),'restored_members':len(entries),
        'restored_bytes':sum(e['bytes'] for e in entries),'analysis_sha256':sha(analysis),
        'reproduced_analysis_sha256':sha(reproduced),'analysis_byte_identical':True,'command':command}
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','candidate','archive','inventory','restore','receipt','analysis']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); create(**{k:v.resolve() for k,v in vars(a).items()})

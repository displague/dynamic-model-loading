"""Archive exact continuing-agent receipts and prove restored analysis equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def create(project,source,run_root,analysis,archive,inventory,restore,receipt):
    for p in [archive,inventory,restore,receipt]:
        if p.exists(): raise ValueError(f'output already exists: {p}')
    files={}
    def add(path,name):
        if name in files: raise ValueError('duplicate archive member')
        files[name]=path
    for path in sorted(run_root.rglob('*')):
        if path.is_file(): add(path,'runs/'+run_root.name+'/'+path.relative_to(run_root).as_posix())
    names=['scripts/continuing_agent.py','scripts/stock_benchmark.py','scripts/verification_offload.py',
           'scripts/analyze_continuing.py','scripts/archive_continuing.py','tests/test_continuing_agent.py',
           'data/committed-replay.json','data/continuing-agent-turns.json','docs/continuing-agent-protocol.md',
           'docs/adr/0003-verified-speculation-boundary.md','configs/stock-speculation-artifacts.json']
    for name in names: add(source/name,'source/'+name)
    fixture=json.loads((source/'data/committed-replay.json').read_text(encoding='utf-8'))
    historical={name:digest for c in fixture['cases'].values() for name,digest in c['source_files'].items()}
    for name,digest in sorted(historical.items()):
        path=project/name
        if sha(path)!=digest: raise ValueError('historical source identity')
        add(path,name)
    native=project/'runs/llama-b10919-source'
    native_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()
    if native_head!='d3146f2b56c2db4711ac8391871c9e529d1946d7' or subprocess.check_output(
            ['git','status','--porcelain'],cwd=native,text=True).strip():
        raise ValueError('native source snapshot is not the clean pinned commit')
    for name in ['common/common.cpp','common/common.h','common/speculative.cpp',
                 'tools/server/server-context.cpp','tools/server/server-task.cpp']:
        add(native/name,'native-source/'+name)
    for name in ['prepare-continuing-fixtures.py','run-continuing-matrix.py','run-continuing-review.py',
                 'test-continuing-candidate.py','collect-continuing-results.py',
                 'continuing-review-v1.txt','continuing-review-v2.txt','stock-publication-review-schema.json',
                 'continuing-protocol-review-v1.jsonl','continuing-protocol-review-v1.stderr.log',
                 'continuing-protocol-review-v1-result.json','continuing-protocol-review-v1-receipt.json',
                 'continuing-protocol-review-v2.jsonl','continuing-protocol-review-v2.stderr.log',
                 'continuing-protocol-review-v2-result.json','continuing-protocol-review-v2-receipt.json']:
        add(project/'runs'/name,'helpers/'+name)
    add(analysis,'analysis.json')
    entries=[{'path':name,'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(files.items())]
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()): z.write(path,name)
    inv={'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
         'entries':entries,'native_source_head':native_head}
    inventory.write_text(json.dumps(inv,indent=2)+'\n',encoding='utf-8')
    restore.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=[e['path'] for e in entries]: raise ValueError('member set/order')
        for name in z.namelist():
            target=(restore/name).resolve()
            if not target.is_relative_to(restore.resolve()): raise ValueError('unsafe archive path')
        z.extractall(restore)
    for entry in entries:
        path=restore/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']: raise ValueError('restoration mismatch')
    reproduced=restore/'reproduced-analysis.json'
    command=[str(project/'.venv/Scripts/python.exe'),str(restore/'source/scripts/analyze_continuing.py'),
             '--runs',str(restore/'runs'/run_root.name),'--output',str(reproduced)]
    subprocess.run(command,check=True)
    if analysis.read_bytes()!=reproduced.read_bytes(): raise ValueError('restored analysis differs')
    result={'archive_sha256':sha(archive),'inventory_sha256':sha(inventory),'restored_members':len(entries),
            'restored_bytes':sum(e['bytes'] for e in entries),'analysis_sha256':sha(analysis),
            'reproduced_analysis_sha256':sha(reproduced),'analysis_byte_identical':True,'command':command}
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','source','runs','analysis','archive','inventory','restore','receipt']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    create(a.project.resolve(),a.source.resolve(),a.runs.resolve(),a.analysis.resolve(),a.archive.resolve(),
           a.inventory.resolve(),a.restore.resolve(),a.receipt.resolve())

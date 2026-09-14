"""Package a completed run into bounded ZIP assets and verify every archived byte."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest_stream(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(2**20),b''):
        h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--analysis',required=True,type=Path)
    parser.add_argument('--assets',required=True,type=Path)
    parser.add_argument('--compact',required=True,type=Path)
    args=parser.parse_args()
    if not (args.run/'completion.json').exists() or not (args.analysis/'summary.json').exists():
        raise ValueError('Only complete analyzed runs can be published')
    args.assets.mkdir(parents=True,exist_ok=False)
    args.compact.mkdir(parents=True,exist_ok=False)
    inventory=json.loads((args.analysis/'verified-files.json').read_text(encoding='utf-8'))
    files=sorted(p for p in args.run.rglob('*') if p.is_file())
    if set(inventory)!={p.relative_to(args.run).as_posix() for p in files}:
        raise ValueError('Run files differ from the analyzed inventory')
    groups=[[]]
    size=0
    for p in files:
        n=p.stat().st_size
        if n>1500*2**20:
            raise ValueError('One file exceeds the archive partition limit')
        if size+n>1500*2**20:
            groups.append([])
            size=0
        groups[-1].append(p)
        size+=n
    assets=[]
    for i,group in enumerate(groups,1):
        archive=args.assets/f'fault-pager-v020-raw-{i:02}.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
            for p in group:
                out.write(p,p.relative_to(args.run).as_posix())
        with zipfile.ZipFile(archive) as restored:
            for info in restored.infolist():
                with restored.open(info) as stream:
                    if digest_stream(stream)!=inventory[info.filename]:
                        raise ValueError('Archive byte restoration failed: '+info.filename)
        with archive.open('rb') as stream:
            assets.append(dict(name=archive.name,bytes=archive.stat().st_size,sha256=digest_stream(stream),members=len(group)))
        print(f'Verified {archive.name}: {len(group)} members',flush=True)
    for name in ('results.jsonl','reference.json','token-ids.json','allocation.json','manifest.json','config.json',
                 'protocol.md','amendment.md','condition-order.json','completion.json'):
        shutil.copyfile(args.run/name,args.compact/name)
    shutil.copyfile(args.run/'calibration/index.json',args.compact/'index.json')
    shutil.copyfile(args.analysis/'summary.json',args.compact/'summary.json')
    shutil.copyfile(args.analysis/'verified-files.json',args.compact/'raw-files.json')
    (args.compact/'archive.json').write_text(json.dumps(dict(assets=assets,members=len(files),
        restoration='Every ZIP member decompressed and SHA256 checked against the verified run.'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()

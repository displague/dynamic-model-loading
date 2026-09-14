"""Archive a supervised screen, including failures, and verify restored bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(stream):
    result=hashlib.sha256()
    for block in iter(lambda:stream.read(2**20),b''):
        result.update(block)
    return result.hexdigest()


def archive_screen(run,archive,receipt):
    run,archive,receipt=(Path(p).resolve() for p in (run,archive,receipt))
    if not (run/'supervisor.json').is_file() or not (run/'decision.json').is_file():
        raise ValueError('Missing supervisor or decision')
    if archive.is_relative_to(run) or receipt.is_relative_to(run) or archive==receipt:
        raise ValueError('Archive outputs must be separate from the raw run')
    if archive.exists() or receipt.exists():
        raise FileExistsError('Preserve existing archive receipts')
    files=sorted(p for p in run.rglob('*') if p.is_file())
    inventory={}
    total=0
    for path in files:
        name=path.relative_to(run.parent).as_posix()
        with path.open('rb') as stream:
            inventory[name]=digest(stream)
        total+=path.stat().st_size
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for path in files:
            out.write(path,path.relative_to(run.parent).as_posix())
    with zipfile.ZipFile(archive) as restored:
        if restored.namelist()!=list(inventory):
            raise ValueError('Archive inventory differs')
        for name,expected in inventory.items():
            with restored.open(name) as stream:
                if digest(stream)!=expected:
                    raise ValueError('Restored bytes differ: '+name)
    # Detect source changes during compression as well as archive corruption.
    if files!=sorted(p for p in run.rglob('*') if p.is_file()):
        raise ValueError('Raw inventory changed during archival')
    for path in files:
        with path.open('rb') as stream:
            if digest(stream)!=inventory[path.relative_to(run.parent).as_posix()]:
                raise ValueError('Raw source changed during archival')
    with archive.open('rb') as stream:
        result=dict(name=archive.name,bytes=archive.stat().st_size,sha256=digest(stream),
            members=len(files),raw_bytes=total,run=run.name,
            restoration='Every ZIP member decompressed and SHA256 checked; raw sources rechecked.',
            inventory=inventory)
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args()
    report=archive_screen(args.run,args.archive,args.receipt)
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'},indent=2))

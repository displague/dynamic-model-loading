"""Restore this indexed archive without executing any archived source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(name):
    path=Path(name)
    if path.is_absolute() or path.drive or '..' in path.parts:raise ValueError('Unsafe archive path')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent;out=Path(args.output)
    if out.exists():raise ValueError('Use a fresh output directory')
    index=json.loads((root/'archive.json').read_text(encoding='utf-8'))
    for name,item in index['files'].items():
        path=root/relative(name)
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Git artifact mismatch: '+name)
    # Verify every container and payload before creating the restored run.
    for asset in index['assets']:
        path=Path(args.assets)/relative(asset['name'])
        if path.stat().st_size!=asset['bytes'] or digest(path)!=asset['sha256']:raise ValueError('Asset mismatch')
        with zipfile.ZipFile(path) as zipped:
            names=zipped.namelist()
            if len(names)!=len(set(names)) or set(names)!={p['file'] for p in asset['payloads']}:raise ValueError('Asset inventory mismatch')
            for item in asset['payloads']:
                relative(item['file']);data=zipped.read(item['file'])
                if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Payload mismatch')
    shutil.copytree(root/'run',out)
    for asset in index['assets']:
        with zipfile.ZipFile(Path(args.assets)/asset['name']) as zipped:
            for item in asset['payloads']:
                path=out/relative(item['file']);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(zipped.read(item['file']))
    print('Restored',out)


if __name__=='__main__':main()

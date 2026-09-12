"""Create and actually restore a lossless archive of explicit completed receipt paths."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, default=root)
parser.add_argument('--name', required=True)
parser.add_argument('--paths', nargs='+', required=True)
args = parser.parse_args()
root = args.workspace.resolve(strict=True)
if not args.name.replace('-', '').isalnum():
    raise ValueError('invalid archive label')
archive = root/'runs/release-assets'/(args.name+'.zip')
inventory_path = root/'runs'/(args.name+'-inventory.json')
restore = root/'runs'/('restore-'+args.name)
if archive.exists() or inventory_path.exists() or restore.exists():
    raise FileExistsError('archive, inventory and restoration destinations must be fresh')
archive.parent.mkdir(exist_ok=True)
members = {}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for relative in args.paths:
    source = (root/relative).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError('source is outside the explicit repository workspace')
    files = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
    for file in files:
        absolute = file.resolve(strict=True)
        boundary = source if source.is_dir() else source.parent
        if not absolute.is_relative_to(boundary):
            raise ValueError('source member escapes its selected receipt directory')
        name = file.relative_to(root).as_posix()
        if name in members:
            raise ValueError('duplicate archive member')
        members[name] = {'path': name, 'bytes': file.stat().st_size, 'sha256': digest(file)}
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for name in sorted(members):
        output.write(root/name, arcname=name)

# The destination is resolved and checked before extracting any nested paths.
intended = (root/'runs').resolve()
if not restore.resolve().is_relative_to(intended):
    raise ValueError('restoration destination escapes runs')
restore.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as source:
    if set(source.namelist()) != set(members) or len(source.namelist()) != len(members):
        raise ValueError('archive member coverage mismatch')
    for member in source.infolist():
        target = (restore/member.filename).resolve()
        if not target.is_relative_to(restore.resolve()):
            raise ValueError('archive member escapes fresh restoration')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(member) as inp, target.open('xb') as out:
            while chunk := inp.read(1024*1024):
                out.write(chunk)
        expected = members[member.filename]
        if target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
            raise ValueError('restored member differs from original inventory')
        if digest(root/member.filename) != expected['sha256']:
            raise ValueError('source changed during archive/restoration')
inventory = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive),
             'source_paths': args.paths, 'members': [members[k] for k in sorted(members)],
             'restoration': {'verified': True, 'member_count': len(members), 'directory': str(restore)},
             'builder_sha256': digest(Path(__file__))}
inventory_path.write_text(json.dumps(inventory, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in inventory.items() if k != 'members'}), flush=True)

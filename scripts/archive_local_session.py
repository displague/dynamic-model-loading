"""Archive the two integration attempts and verify an actual restored analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

import stock_benchmark as stock


def archive(project, source):
    runs = project/'runs'
    target = runs/'release-assets/local-session-v0190-raw.zip'
    inventory = runs/'local-session-v0190-inventory.json'
    restored = runs/'restore-local-session-v0190'
    restoration = runs/'local-session-v0190-restoration.json'
    if any(p.exists() for p in (target, inventory, restored, restoration)):
        raise ValueError('fresh archive/restoration paths required')
    files = {}
    for label in ['v1', 'v2']:
        base = runs/('local-session-20260913-'+label)
        for p in sorted(base.rglob('*')):
            if p.is_file() and not p.relative_to(base).as_posix().startswith('measured/codex/state/'):
                files[p.relative_to(runs).as_posix()] = p
    for p in sorted((source/'scripts').glob('*.py')):
        files['analysis-source/scripts/'+p.name] = p
    for version in ['v1', 'v2']:
        base = runs/('local-session-source-'+version)
        for rel in ['scripts/local_session.py', 'scripts/local_session_smoke.py',
                    'docs/local-session-protocol.md', 'configs/stock-speculation-artifacts.json']:
            files['measured-source-'+version+'/'+rel] = base/rel
        amendment = base/'docs/local-session-amendment-1.md'
        if amendment.exists(): files['measured-source-'+version+'/docs/local-session-amendment-1.md'] = amendment
    target.parent.mkdir(exist_ok=True)
    entries = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, p in sorted(files.items()):
            entries.append({'name': name, 'bytes': p.stat().st_size, 'sha256': stock.digest(p)})
            z.write(p, name)
    stock.write_json(inventory, {'archive': target.name, 'archive_bytes': target.stat().st_size,
        'archive_sha256': stock.digest(target), 'entries': entries,
        'exclusion': 'Machine-specific Codex state databases retained locally; CLI events and all analysis inputs are archived.'})
    restored.mkdir()
    with zipfile.ZipFile(target) as z:
        if set(z.namelist()) != set(files) or len(z.namelist()) != len(files):
            raise ValueError('archive coverage')
        for name in z.namelist():
            if not (restored/name).resolve().is_relative_to(restored.resolve()):
                raise ValueError('unsafe archive path')
        z.extractall(restored)
    for e in entries:
        p = restored/e['name']
        if p.stat().st_size != e['bytes'] or stock.digest(p) != e['sha256']:
            raise ValueError('restoration mismatch')
    analysis = runs/'local-session-v0190-analysis.json'
    reproduced = restored/'analysis.json'
    subprocess.run([str(project/'.venv/Scripts/python.exe'),
                    str(restored/'analysis-source/scripts/analyze_local_session.py'),
                    '--root', str(restored), '--output', str(reproduced)], check=True)
    if analysis.read_bytes() != reproduced.read_bytes():
        raise ValueError('restored analysis differs')
    stock.write_json(restoration, {'archive_sha256': stock.digest(target),
        'inventory_sha256': stock.digest(inventory), 'restored_members': len(entries),
        'analysis_sha256': stock.digest(analysis), 'reproduced_analysis_sha256': stock.digest(reproduced),
        'analysis_byte_identical': True})
    print(json.dumps({'archive_bytes': target.stat().st_size, 'members': len(entries), 'sha256': stock.digest(target)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    archive(a.project.resolve(), a.source.resolve())

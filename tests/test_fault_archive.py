import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def test_archive_restores_nested_files_with_portable_manifest_paths(tmp_path):
    run=tmp_path/'run'
    analysis=tmp_path/'analysis'
    (run/'calibration').mkdir(parents=True)
    analysis.mkdir()
    for name in ('results.jsonl','reference.json','token-ids.json','allocation.json','manifest.json',
                 'config.json','protocol.md','amendment.md','condition-order.json','completion.json',
                 'calibration/index.json'):
        (run/name).write_text('{}',encoding='utf-8')
    (analysis/'summary.json').write_text('{}',encoding='utf-8')
    inventory={p.relative_to(run).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in run.rglob('*') if p.is_file()}
    (analysis/'verified-files.json').write_text(json.dumps(inventory),encoding='utf-8')
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/archive_fault_pager.py'),
             '--run',str(run),'--analysis',str(analysis),'--assets',str(tmp_path/'assets'),
             '--compact',str(tmp_path/'compact')]
    subprocess.run(command,check=True,capture_output=True)
    with zipfile.ZipFile(next((tmp_path/'assets').glob('*.zip'))) as archive:
        assert set(archive.namelist())==set(inventory)
        assert archive.read('calibration/index.json')==b'{}'

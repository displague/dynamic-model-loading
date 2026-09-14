import importlib.util
from pathlib import Path
import zipfile

import pytest


spec=importlib.util.spec_from_file_location('archive_screen',Path(__file__).parents[1]/'scripts/archive_screen.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_preserves_failed_screen_and_nested_receipts(tmp_path):
    run=tmp_path/'failed-run'
    run.mkdir()
    (run/'supervisor.json').write_text('{"status":"timeout"}')
    (run/'decision.json').write_text('{"decision":"inconclusive"}')
    (run/'worker').mkdir()
    (run/'worker'/'partial.bin').write_bytes(b'raw\x00data')
    archive,receipt=tmp_path/'raw.zip',tmp_path/'archive.json'
    result=module.archive_screen(run,archive,receipt)
    assert result['members']==3 and result['bytes']==archive.stat().st_size
    with zipfile.ZipFile(archive) as restored:
        assert restored.read('failed-run/worker/partial.bin')==b'raw\x00data'
    with pytest.raises(FileExistsError):
        module.archive_screen(run,archive,receipt)


def test_archive_requires_terminal_receipts_and_separate_outputs(tmp_path):
    with pytest.raises(ValueError,match='Missing'):
        module.archive_screen(tmp_path,tmp_path/'raw.zip',tmp_path/'archive.json')
    (tmp_path/'supervisor.json').write_text('{}')
    (tmp_path/'decision.json').write_text('{}')
    with pytest.raises(ValueError,match='separate'):
        module.archive_screen(tmp_path,tmp_path/'raw.zip',tmp_path/'archive.json')

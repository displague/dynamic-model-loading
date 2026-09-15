import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


@pytest.mark.parametrize('executed',[1,2])
def test_full_tip_receipt_clears_filter_and_reconciles_collection(monkeypatch,tmp_path,executed):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/validate_release_tip.py'))
    commit = 'a'*40
    def output(command,**kwargs):
        if command[1]=='status':
            return ''
        return commit+'\n'
    def execute(command,**kwargs):
        if command[0]=='git':
            return subprocess.CompletedProcess(command,0)
        assert not any(k.upper().startswith('PYTEST_') for k in kwargs['env'])
        assert command[command.index('-o')+1]=='addopts='
        if '--collect-only' in command:
            return subprocess.CompletedProcess(command,0,'tests/test_a.py::test_1\ntests/test_a.py::test_2\n')
        xml = Path(next(x.split('=',1)[1] for x in command if x.startswith('--junitxml=')))
        xml.write_text(f'<testsuites><testsuite tests="{executed}" failures="0" errors="0" skipped="0" time="1" timestamp="test"/></testsuites>')
        return subprocess.CompletedProcess(command,0,'passed')
    monkeypatch.setattr(subprocess,'check_output',output)
    monkeypatch.setattr(subprocess,'run',execute)
    monkeypatch.setenv('PYTEST_ADDOPTS','-k narrowed')
    prefix = tmp_path/'validation'
    monkeypatch.setattr(sys,'argv',['validate','--release','v0.25.0','--prefix',str(prefix),'--source-freeze',commit])
    if executed==1:
        with pytest.raises(SystemExit,match='validation failed'):
            module['main']()
        assert not prefix.with_suffix('.json').exists()
    else:
        module['main']()
        result = json.loads(prefix.with_suffix('.json').read_text())
        assert result['tests']==result['collected_tests']==2
        assert result['pytest_environment_cleared'] and result['commit']==commit

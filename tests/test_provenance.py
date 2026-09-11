import shutil
import subprocess

import pytest

from dynamic_model_loading.provenance import committed_inputs, frozen_environment, verify_snapshot


@pytest.fixture
def committed_repo(tmp_path):
    repo=tmp_path/'source';repo.mkdir()
    source=repo/'src/dynamic_model_loading/sample.py';source.parent.mkdir(parents=True)
    source.write_text('VALUE = 1\n',encoding='utf-8')
    (source.parent/'metrics.py').write_text('DEPENDENCY = 1\n',encoding='utf-8')
    cfg=repo/'configs/refinement-feasibility.json';cfg.parent.mkdir();cfg.write_text('{}\n',encoding='utf-8')
    protocol=repo/'docs/refinement-feasibility-protocol.md';protocol.parent.mkdir();protocol.write_text('Frozen.\n',encoding='utf-8')
    (repo/'.gitattributes').write_text('* text=auto eol=lf\n',encoding='utf-8')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                    'commit','-qm','Freeze fixture'],check=True)
    return repo,source,cfg,protocol


def test_provenance_checks_source_owner_even_from_another_clean_checkout(committed_repo,tmp_path,monkeypatch):
    repo,source,cfg,protocol=committed_repo
    other=tmp_path/'launch';other.mkdir();subprocess.run(['git','init','-q',str(other)],check=True)
    monkeypatch.chdir(other)
    receipt,resolved=committed_inputs(source,cfg,'docs/refinement-feasibility-protocol.md')
    assert resolved==protocol and receipt['source_commit']==subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
    source.write_text('VALUE = 2\n',encoding='utf-8')
    with pytest.raises(ValueError,match='owning source'):committed_inputs(source,cfg,'docs/refinement-feasibility-protocol.md')


def test_provenance_rejects_external_configuration(committed_repo,tmp_path):
    _,source,cfg,_=committed_repo
    external=tmp_path/'external.json';shutil.copyfile(cfg,external)
    with pytest.raises(ValueError,match='outside'):committed_inputs(source,external,'docs/refinement-feasibility-protocol.md')


def test_archived_source_cannot_be_replaced_by_new_self_reported_hashes(committed_repo,tmp_path):
    _,source,cfg,protocol=committed_repo
    receipt,_=committed_inputs(source,cfg,'docs/refinement-feasibility-protocol.md')
    run=tmp_path/'archive';(run/'source').mkdir(parents=True)
    for path in source.parent.glob('*.py'):shutil.copyfile(path,run/'source'/path.name)
    shutil.copyfile(cfg,run/'config.json');shutil.copyfile(protocol,run/'protocol.md')
    verify_snapshot(run,receipt,'configs/refinement-feasibility.json','docs/refinement-feasibility-protocol.md',source)
    (run/'source/sample.py').write_text('VALUE = 3\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Git object'):
        verify_snapshot(run,receipt,'configs/refinement-feasibility.json','docs/refinement-feasibility-protocol.md',source)


def test_omitted_dependency_cannot_disappear_from_archive_and_claimed_inventory(committed_repo,tmp_path):
    _,source,cfg,protocol=committed_repo
    receipt,_=committed_inputs(source,cfg,'docs/refinement-feasibility-protocol.md')
    run=tmp_path/'archive';(run/'source').mkdir(parents=True)
    shutil.copyfile(source,run/'source/sample.py');shutil.copyfile(cfg,run/'config.json');shutil.copyfile(protocol,run/'protocol.md')
    del receipt['committed_files']['src/dynamic_model_loading/metrics.py']
    with pytest.raises(ValueError,match='source inventory'):
        verify_snapshot(run,receipt,'configs/refinement-feasibility.json','docs/refinement-feasibility-protocol.md',source)


@pytest.mark.parametrize('change',['python','torch','transformers'])
def test_frozen_environment_rejects_unregistered_versions(change):
    env={'python':'3.14.3 (fixture)','torch':'2.10.0+cu130','versions':{'transformers':'5.13.1'}}
    frozen_environment(env)
    if change=='transformers':env['versions'][change]='5.14.0'
    else:env[change]='different'
    with pytest.raises(ValueError,match='Frozen'):frozen_environment(env)

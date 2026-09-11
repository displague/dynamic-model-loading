"""Bind measurements to the owning committed worktree and frozen environment."""

from pathlib import Path
import re
import subprocess


def owning_repository(source_file):
    location = Path(source_file).resolve().parent
    return Path(subprocess.check_output(['git','-C',str(location),'rev-parse','--show-toplevel']).decode().strip()).resolve()


def committed_inputs(source_file, config_path, protocol_relative):
    source_file = Path(source_file).resolve()
    repository = owning_repository(source_file)
    def git(*args, **kwargs):
        return subprocess.check_output(['git','-C',str(repository),*args],**kwargs)
    if git('status','--porcelain').strip():
        raise ValueError('Commit the owning source worktree before measurement')
    protocol = repository / protocol_relative
    paths = [*source_file.parent.rglob('*.py'),Path(config_path).resolve(),protocol.resolve()]
    files = {}
    for path in paths:
        try:relative = path.relative_to(repository).as_posix()
        except ValueError as error:raise ValueError('Input is outside the owning source worktree') from error
        expected = git('rev-parse','HEAD:'+relative).decode().strip()
        actual = git('hash-object','--path',relative,'--stdin',input=path.read_bytes()).decode().strip()
        if actual != expected:raise ValueError('Input differs from committed snapshot: '+relative)
        files[relative] = expected
    return {'source_commit':git('rev-parse','HEAD').decode().strip(),'committed_files':files},protocol


def verify_snapshot(run, receipt, config_relative, protocol_relative, source_file):
    """Check archived bytes against the actual Git objects, not claimant digests alone."""
    run = Path(run)
    repository = owning_repository(source_file)
    commit = receipt['source_commit']
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Invalid committed source identity')
    package = 'src/dynamic_model_loading'
    tracked = subprocess.check_output(['git','-C',str(repository),'ls-tree','-r','--name-only',commit,'--',package]).decode().splitlines()
    required_sources = {name for name in tracked if name.endswith('.py')}
    archived_sources = {package+'/'+p.relative_to(run/'source').as_posix() for p in (run/'source').rglob('*.py')}
    if not required_sources or archived_sources != required_sources:
        raise ValueError('Archived source inventory differs from recorded Git commit')
    expected_paths = required_sources | {config_relative,protocol_relative}
    if set(receipt['committed_files']) != expected_paths:
        raise ValueError('Committed input inventory mismatch')
    for relative,blob in receipt['committed_files'].items():
        resolved = subprocess.check_output(['git','-C',str(repository),'rev-parse',receipt['source_commit']+':'+relative]).decode().strip()
        if resolved != blob:raise ValueError('Committed blob identity mismatch')
        if relative==config_relative:path=run/'config.json'
        elif relative==protocol_relative:path=run/'protocol.md'
        else:path=run/'source'/Path(relative).relative_to(package)
        original = subprocess.check_output(['git','-C',str(repository),'cat-file','blob',blob])
        if path.read_bytes().replace(b'\r\n',b'\n') != original.replace(b'\r\n',b'\n'):
            raise ValueError('Archived input differs from committed Git object: '+relative)


def frozen_environment(env):
    if (env['python'].split()[0] != '3.14.3' or env['torch'] != '2.10.0+cu130'
            or env['versions']['transformers'] != '5.13.1'):
        raise ValueError('Frozen Python, PyTorch and Transformers versions required')

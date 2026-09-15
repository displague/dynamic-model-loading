"""Run full CPU tests on an immutable clean tip and bind the receipt to that commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True)
    parser.add_argument('--prefix',required=True,type=Path)
    parser.add_argument('--source-freeze',required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=root,text=True).strip()
    if git('status','--porcelain'):
        raise SystemExit('Final tip must be clean')
    commit = git('rev-parse','HEAD')
    source = git('rev-parse',args.source_freeze+'^{commit}')
    subprocess.run(['git','merge-base','--is-ancestor',source,commit],cwd=root,check=True)
    xml = args.prefix.with_suffix('.xml').resolve()
    receipt = args.prefix.with_suffix('.json').resolve()
    if xml.exists() or receipt.exists():
        raise SystemExit('Preserve existing validation receipts')
    started = time.perf_counter()
    env = dict(os.environ)
    for name in list(env):
        if name.upper().startswith('PYTEST_'):
            del env[name]
    base = [sys.executable,'-m','pytest','-q','-o','addopts=','tests']
    collection = subprocess.run([*base,'--collect-only'],cwd=root,env=env,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    nodes = [line for line in collection.stdout.splitlines() if line.startswith('tests/') and '::' in line]
    if collection.returncode or not nodes or len(set(nodes))!=len(nodes):
        raise SystemExit('Full collection failed; receipt not issued')
    command = [*base,'--junitxml='+str(xml)]
    proc = subprocess.run(command,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(proc.stdout,flush=True)
    counts = {k:0 for k in ('tests','failures','errors','skipped')}
    suites = ET.parse(xml).getroot().findall('testsuite')
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key,0))
    clean,unchanged = not git('status','--porcelain'),git('rev-parse','HEAD')==commit
    if proc.returncode or not clean or not unchanged or counts['tests']!=len(nodes) or any(counts[k] for k in ('failures','errors','skipped')):
        raise SystemExit('Final-tip validation failed; receipt not issued')
    result = dict(release=args.release,commit=commit,source_freeze=source,command=command,
        worktree_clean_before=True,worktree_clean_after=clean,head_unchanged=unchanged,
        exit_code=proc.returncode,**counts,wall_seconds=time.perf_counter()-started,
        junit_seconds=sum(float(s.get('time',0)) for s in suites),
        junit_timestamp=suites[0].get('timestamp'),receipt=xml.name,
        collected_tests=len(nodes),collection_sha256=hashlib.sha256('\n'.join(nodes).encode()).hexdigest(),
        pytest_environment_cleared=True,configured_addopts_cleared=True,
        receipt_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),inference_rerun=False)
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

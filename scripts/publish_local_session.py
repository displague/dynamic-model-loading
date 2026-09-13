"""Validate final candidate, evidence and exact draft assets before publishing v0.19.0."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

TAG='v0.19.0'
REPO='displague/dynamic-model-loading'


def require(value,message):
    if not value: raise ValueError(message)


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def run(project,args): return subprocess.check_output(args,cwd=project,encoding='utf-8').strip()


def verify_remote_tag(text,local_object,head):
    pairs=[line.split() for line in text.splitlines() if line.strip()]
    expected={f'refs/tags/{TAG}':local_object,f'refs/tags/{TAG}^{{}}':head}
    require(len(pairs)==2 and all(len(p)==2 for p in pairs) and
            {ref:object_id for object_id,ref in pairs}==expected,'remote annotated tag identity')


def publish(project,candidate):
    git=lambda *a:run(project,['git',*a])
    head=run(candidate,['git','rev-parse','HEAD']); tree=run(candidate,['git','rev-parse','HEAD^{tree}'])
    require(not git('status','--porcelain') and not run(candidate,['git','status','--porcelain']),'dirty candidate')
    require(git('rev-parse','HEAD')==head==git('rev-parse','origin/main')==git('rev-parse',TAG+'^{commit}'),'candidate/main/tag')
    require(git('ls-remote','origin','refs/heads/main').split()[0]==head,'remote main')
    require(git('cat-file','-t',TAG)=='tag','tag is not annotated')
    verify_remote_tag(git('ls-remote','origin',f'refs/tags/{TAG}',f'refs/tags/{TAG}^{{}}'),
                      git('rev-parse',TAG),head)
    notes=(project/'docs/releases'/f'{TAG}.md').read_text(encoding='utf-8').strip()
    tag_body=git('cat-file','tag',TAG).split('\n\n',1)[1].strip()
    require(notes==tag_body,'tag notes differ')
    evidence=project/'results/local-session-20260913'
    archive=project/'runs/release-assets/local-session-v0190-raw.zip'
    inventory=project/'runs/local-session-v0190-inventory.json'
    restoration=project/'runs/local-session-v0190-restoration.json'
    analysis=project/'runs/local-session-v0190-analysis.json'
    require(inventory.read_bytes()==(evidence/'archive-inventory.json').read_bytes(),'reviewed inventory differs')
    require(restoration.read_bytes()==(evidence/'restoration.json').read_bytes(),'reviewed restoration differs')
    require(analysis.read_bytes()==(evidence/'analysis.json').read_bytes(),'reviewed analysis differs')
    inv=read(inventory); restored=read(restoration)
    require(archive.stat().st_size==inv['archive_bytes'] and sha(archive)==inv['archive_sha256'],'archive identity')
    require(restored['archive_sha256']==sha(archive) and restored['inventory_sha256']==sha(inventory) and
            restored['analysis_sha256']==sha(analysis)==restored['reproduced_analysis_sha256'] and
            restored['analysis_byte_identical'] is True and restored['restored_members']==len(inv['entries']),
            'restoration identities')
    review_path=project/'runs/local-session-v0190-review-receipt.json'
    review=read(review_path)
    require(review['exit_code']==0 and review['reviewed_tree']==tree==review['tree_after'] and
            review['result']['clearance'] is True and review['result']['findings']==[] and
            not review['unstaged_after'] and not review['untracked_after'],'final review clearance')
    require(review['helpers_sha256']==review['helpers_after_sha256'],'review helper mutation')
    require(set(review['helpers_sha256'])=={'run-local-session-review.py','local-session-review.txt',
            'stock-publication-review-schema.json','test-continuing-candidate.py'},'review helper coverage')
    for name,digest in review['helpers_sha256'].items():
        require(sha(project/'runs'/name)==digest,'review helper identity: '+name)
    result_path=Path(review['command'][review['command'].index('--output-last-message')+1])
    require(sha(result_path)==review['result_sha256'] and read(result_path)==review['result'],'completed review result')
    prefix=str(result_path).removesuffix('-result.json')
    events=Path(prefix+'.jsonl'); stderr=Path(prefix+'.stderr.log')
    require(sha(events)==review['events_sha256'] and sha(stderr)==review['stderr_sha256'],'review transcript identities')
    require(any(json.loads(line).get('type')=='turn.completed' for line in events.read_text(encoding='utf-8').splitlines()),
            'review never completed')
    tests=[]; assets=[archive,inventory,restoration,analysis,review_path,events,result_path]
    for key,version in [('210','2.10.0+cu130'),('212','2.12.0+cu130')]:
        path=project/f'runs/local-session-v0190-tests-{key}.json'; log=path.with_suffix('.log')
        test=read(path)
        require(test['head']==head==test['head_after'] and test['tree']==tree and test['clean_after'] and
                test['exit_code']==0 and test['passed']==291,'full clean-candidate suite')
        require(test['packages']['packages']['torch']==version and test['log_sha256']==sha(log) and
                test['helper_sha256']==sha(project/'runs/test-continuing-candidate.py'),'test identities')
        require(test['command']==[test['packages']['executable'],'-m','pytest','-q'],'test command')
        require(re.findall(r'(\d+) passed in [\d.]+s',log.read_text(encoding='utf-8'))==['291'],'test log count')
        tests.append({'receipt':path.name,'sha256':sha(path),'passed':test['passed']})
        assets.extend([path,log])
    api=lambda endpoint:json.loads(run(project,['gh','api','repos/'+REPO+'/'+endpoint]))
    release_id=json.loads(run(project,['gh','release','view',TAG,'--json','databaseId']))['databaseId']
    release=api('releases/'+str(release_id))
    require(release['tag_name']==TAG and release['draft'] and release['prerelease'] and
            release['body'].replace('\r\n','\n').strip()==notes,'draft release metadata')
    expected={p.name:sha(p) for p in assets}
    actual={a['name']:a for a in release['assets']}
    require(set(actual)==set(expected),'exact draft asset set')
    for name,digest in expected.items(): require(actual[name]['digest']=='sha256:'+digest,'server asset digest: '+name)
    require(api('issues/27')['state']=='open','fidelity issue must remain open')
    issue=api('issues/32')
    require(issue['state']=='closed' and issue['state_reason']=='completed','bounded issue completion')
    require(all(api(f'issues/{n}')['state']=='open' for n in [25,26]),'deferred/pending issue state')
    require(api('milestones/9')['state']=='open','milestone remains open')
    validation=project/'runs/local-session-v0190-publication-validation.json'
    require(not validation.exists(),'publication validation already exists')
    validation.write_text(json.dumps({'tag':TAG,'commit':head,'tree':tree,'release_id':release_id,
        'assets_sha256':expected,'tests':tests,'review_sha256':sha(review_path),
        'notes_sha256':sha(project/'docs/releases'/f'{TAG}.md'),'validated':True},indent=2)+'\n',encoding='utf-8')
    subprocess.run(['gh','release','upload',TAG,str(validation)],cwd=project,check=True)
    expected[validation.name]=sha(validation)
    staged=api('releases/'+str(release_id))
    require({a['name']:a['digest'] for a in staged['assets']}=={k:'sha256:'+v for k,v in expected.items()},'final asset set')
    subprocess.run(['gh','release','edit',TAG,'--draft=false','--prerelease'],cwd=project,check=True)
    public=api('releases/tags/'+TAG)
    require(public['id']==release_id and not public['draft'] and public['prerelease'] and
            public['body'].replace('\r\n','\n').strip()==notes,'public metadata')
    require({a['name']:a['digest'] for a in public['assets']}=={k:'sha256:'+v for k,v in expected.items()},'public assets')
    print(public['html_url'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True); p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args(); publish(a.project.resolve(),a.candidate.resolve())

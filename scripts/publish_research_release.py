"""Publish an already reviewed, committed and validated research delivery.

This does not commit code or manufacture a review. Existing tags/releases are
never overwritten. The final validation receipt and archive must already exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True)
    parser.add_argument('--title',required=True)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--archive-receipt',required=True,type=Path)
    parser.add_argument('--receipt',required=True,type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def run(*cmd):
        return subprocess.check_output(cmd,cwd=root,encoding='utf-8').strip()
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    note = root/'docs/releases'/f'{args.version}.md'
    validation = json.loads(args.validation.read_text(encoding='utf-8'))
    archive = json.loads(args.archive_receipt.read_text(encoding='utf-8'))
    xml = args.validation.parent/validation['receipt']
    tip = run('git','rev-parse','HEAD')
    if args.receipt.exists() or run('git','status','--porcelain'):
        raise SystemExit('Need clean tip and a fresh publication receipt')
    if (validation['commit']!=tip or validation['release']!=args.version or validation['exit_code']!=0 or
        not all(validation[k] is True for k in ('worktree_clean_before','worktree_clean_after','head_unchanged')) or
        validation['tests']<=0 or any(validation[k] for k in ('failures','errors','skipped')) or
        sha(xml)!=validation['receipt_sha256']):
        raise SystemExit('Final-tip validation does not qualify this release')
    if (archive['name']!=args.archive.name or archive['bytes']!=args.archive.stat().st_size or
        archive['sha256']!=sha(args.archive)):
        raise SystemExit('Archive receipt mismatch')
    run('git','fetch','origin','main')
    if run('git','rev-parse','origin/main')!=tip:
        raise SystemExit('Push this reviewed tip first')
    if run('git','tag','--list',args.version):
        raise SystemExit('Preserve existing tag; inspect publication state manually')
    body = note.read_text(encoding='utf-8')
    original = subprocess.check_output(['git','show',tip+':docs/releases/'+args.version+'.md'],cwd=root)
    if note.read_bytes().replace(b'\r\n',b'\n')!=original.replace(b'\r\n',b'\n'):
        raise SystemExit('Release notes differ from the validated commit')
    run('git','tag','-a',args.version,tip,'--cleanup=verbatim','-F',str(note))
    if run('git','rev-parse',args.version+'^{commit}')!=tip:
        raise SystemExit('Release tag differs from the validated commit')
    tagged = subprocess.check_output(['git','cat-file','tag',args.version],cwd=root).split(b'\n\n',1)[1]
    if tagged.replace(b'\r\n',b'\n')!=original.replace(b'\r\n',b'\n'):
        raise SystemExit('Tag body mismatch; do not push')
    run('git','push','origin','refs/tags/'+args.version)
    assets = [args.archive,args.validation,xml]
    url = run('gh','release','create',args.version,'--verify-tag','--prerelease','--title',args.title,
              '--notes',body,*[str(p) for p in assets])
    remote = json.loads(run('gh','api',f'repos/displague/dynamic-model-loading/releases/tags/{args.version}'))
    if remote['draft'] or not remote['prerelease'] or remote['body'].replace('\r\n','\n')!=body.replace('\r\n','\n'):
        raise SystemExit('Published release body/status mismatch')
    actual = {a['name']:a for a in remote['assets']}
    for path in assets:
        item = actual[path.name]
        if item['size']!=path.stat().st_size or item.get('digest')!='sha256:'+sha(path):
            raise SystemExit('Remote release asset mismatch: '+path.name)
    result = dict(version=args.version,commit=tip,url=url,release_id=remote['id'],
                  tag_and_release_bodies_match=True,assets_verified=True)
    with args.receipt.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

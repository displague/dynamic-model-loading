"""Independent ledger reconciliation and frozen fault-pager summaries; no inference."""
import argparse
from collections import Counter, OrderedDict
import gzip
import json
from pathlib import Path
import statistics
import subprocess

import torch
from safetensors.torch import load_file

from .experiment import digest, write_json
from .metrics import compare_logits, relative_l2
from .provenance import verify_snapshot, owning_repository
from .fault_pager_study import shuffled_conditions


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as src:
        for line in src:
            yield json.loads(line)


def audit_pages(path,budget,mode='lru'):
    resident, prefetched, stats = OrderedDict(), set(), Counter()
    demands, evicted = [], []
    eager_release = None
    used = 0
    for row in read_rows(path):
        outcome = row['outcome']
        key = (row['key']['layer'],row['key']['page']) if 'key' in row else None
        demand(outcome in ('load','hit','release','cancelled_prefetch','lookup','selection_copy','layer'),
               'unknown page event')
        if eager_release is not None:
            demand(outcome=='release' and key==eager_release and row['request']=='eager_release',
                   'eager page retained beyond execution')
            eager_release=None
        if outcome in ('load','hit'):
            demand(row['request'] in ('demand','prefetch'),'unknown page request')
            demand(row['request']!='prefetch' or mode=='prefetch','prefetch outside declared policy')
            if row['request']=='demand':
                demands.append(key)
                if mode=='eager':
                    demand(outcome=='load','eager cache hit')
                    eager_release=key
        if outcome=='load':
            demand(key not in resident,'duplicate page load')
            demand(0<=key[0]<28 and 0<=key[1]<35,'unknown page')
            demand(row['bytes']==3*1536*256*4==row['h2d_bytes'],'wrong physical CUDA payload')
            listed=[(v['layer'],v['page']) for v in row.get('evicted',[])]
            demand(listed==evicted,'eviction list mismatch')
            demand(not evicted or used+row['bytes']+3*1536*256*4>budget,'unnecessary eviction')
            evicted=[]
            resident[key] = row['bytes']
            used += row['bytes']
            stats['h2d_bytes'] += row['h2d_bytes']
            stats[row['request']+'_bytes'] += row['bytes']
            stats['transfer_wall_ms'] += row['wall_ms']
            stats['transfer_cuda_ms'] += row['cuda_ms']
            if row['request']=='prefetch':
                prefetched.add(key)
        elif outcome=='release':
            demand(key in resident and resident[key]==row['bytes'],'release of absent/wrong-sized page')
            if row['request']=='eviction':
                demand(mode!='eager' and key==next(iter(resident)),'non-LRU eviction')
                evicted.append(key)
                stats['evictions']+=1
            else:
                demand(row['request'] in ('eager_release','episode_end'),'unknown release reason')
                demand(row['request']!='eager_release' or mode=='eager','eager release outside policy')
            used -= resident.pop(key)
        elif outcome=='hit':
            demand(key in resident and row['bytes']==0,'hit without residency')
            resident.move_to_end(key)
            stats[row['request']+'_hits'] += 1
            if row['request']=='demand':
                prefetched.discard(key)
        elif outcome=='cancelled_prefetch':
            demand(key in prefetched,'unproven prefetch cancellation')
            prefetched.remove(key)
            # Runtime emits cancellation immediately after removal and before release receipt.
            demand(row['used_bytes']==used-resident[key],'cancellation cache state')
        elif outcome=='layer':
            selections=row['selected_pages']
            demand(len(selections)==1 and len(set(selections[0]))==27 and len(selections[0])==27,
                   'draft selection did not execute 27 distinct pages')
            demand(demands==[(row['layer'],p) for p in sorted(selections[0])],
                   'selected pages do not reconcile with demand transactions')
            demands=[]
            stats['peak_host_metadata_bytes']=max(stats['peak_host_metadata_bytes'],row.get('host_metadata_bytes',0))
        demand(0<=used<=budget,'cache exceeds declared budget')
        if outcome!='cancelled_prefetch':
            demand(row['used_bytes']==used,'cache byte ledger drift')
        stats[outcome]+=1
        stats['peak_payload_bytes']=max(stats['peak_payload_bytes'],used)
    demand(not resident and not prefetched,'unreleased page state at episode end')
    demand(not demands and not evicted and eager_release is None,'unfinished page execution')
    return dict(stats)


def audit_rounds(path,episode):
    generated, attempted, accepted = [],0,0
    fallback_consumptions=0
    stopped=False
    for number,row in enumerate(read_rows(path),1):
        demand(not stopped,'round after EOS or cap')
        p,t=row['proposed'],row['target_predictions']
        demand(len(p)==len(t)==4 and row['round']==number,'malformed proposal round')
        demand(row['base']==32+len(generated),'verification prefix boundary')
        a=0
        for proposal,target in zip(p,t):
            if proposal!=target:
                break
            a+=1
            if target in (151643,151645):
                break
        full=p[:a]
        fallback=None
        if a<4 and (not full or full[-1] not in (151643,151645)):
            fallback=t[a]
            full.append(fallback)
        committed=full[:64-len(generated)]
        demand(row['accepted']==a and row['fallback']==fallback and row['committed']==committed,'bad commit provenance')
        demand(row['emitted_accepted']==min(a,len(committed)),'accepted/emitted denominator drift')
        stop='eos' if committed[-1] in (151643,151645) else ('length' if len(generated)+len(committed)==64 else ('rejected' if fallback is not None else 'proposal_exhausted'))
        # The verifier names its nonterminal outcomes; termination is independently derived.
        demand(row['stop_reason']==stop,'incorrect round stop reason')
        stopped=stop in ('eos','length')
        expected_cache=row['base']+(a+(not stopped) if fallback is not None else 4)
        demand(row['target_cache']==row['draft_cache']==expected_cache,'incorrect KV rollback boundary')
        if fallback is not None and not stopped:
            fallback_consumptions+=1
        attempted+=4
        accepted+=min(a,len(committed))
        generated+=committed
    demand(generated==episode['ids'],'round ledger/output mismatch')
    demand(attempted==episode['attempted'] and accepted==episode['accepted'],'acceptance aggregate mismatch')
    demand(stopped and episode.get('stop_reason',stop)==stop,'missing or incorrect episode stop')
    return 28*(32+attempted+fallback_consumptions)


def analyze(run,output):
    run,output=Path(run),Path(output)
    demand((run/'completion.json').exists() and not (run/'failure.json').exists(),'run incomplete or failed')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/fault-pager.json','docs/fault-pager-protocol.md',__file__)
    original=subprocess.check_output(['git','show',manifest['source_commit']+':docs/fault-pager-amendment-1.md'],
                                     cwd=owning_repository(__file__))
    demand(original.replace(b'\r\n',b'\n')==(run/'amendment.md').read_bytes().replace(b'\r\n',b'\n'),
           'amendment differs from committed Git object')
    for file,key in [('config.json','config_sha256'),('corpus.jsonl','corpus_sha256'),
                     ('token-ids.json','tokens_sha256'),('protocol.md','protocol_sha256'),('amendment.md','amendment_sha256')]:
        demand(digest(run/file)==manifest[key],'snapshot hash mismatch: '+file)
    for name,sha in manifest['source_sha256'].items():
        demand(digest(run/'source'/name)==sha,'source snapshot hash mismatch')
    index_info=json.loads((run/'calibration/index.json').read_text(encoding='utf-8'))
    demand(digest(run/'calibration/index.safetensors')==index_info['sha256'],'index digest mismatch')
    index=load_file(run/'calibration/index.safetensors')
    cal_docs=[r['id'] for r in read_rows(run/'corpus.jsonl') if r['split']=='calibration']
    for layer in range(28):
        name=f'layer-{layer:02}.safetensors'
        demand(digest(run/'calibration'/name)==index_info['calibration_files'][name],'calibration file hash drift')
        data=load_file(run/'calibration'/name)
        x,scores=data['inputs'],data['scores']
        demand(x.shape==(10240,1536) and scores.shape==(10240,35),'wrong calibration shape')
        chosen=[0]
        distances=(x-x[0]).square().sum(-1)
        for _ in range(15):
            distances[chosen]=-1
            j=int(distances.argmax())
            chosen.append(j)
            distances=torch.minimum(distances,(x-x[j]).square().sum(-1))
        expected_positions=[dict(document=cal_docs[j//128],position=j%128) for j in chosen]
        demand(expected_positions==index_info['positions'][str(layer)],'wrong calibration source positions')
        demand(torch.equal(x[chosen],index[f'centroids.{layer}']),'wrong index centroid')
        demand(torch.equal(scores[chosen].argsort(descending=True,stable=True),index[f'rankings.{layer}']),
               'wrong index page rankings')
        del data,x,scores,distances
    rows=list(read_rows(run/'results.jsonl'))
    resources=list(read_rows(run/'resources.jsonl'))
    demand(resources and all(r['gpu_used']<=15000*2**20 and r['host_available']>=2048*2**20
                             for r in resources),'raw resource limit failure')
    expected_resources=dict(samples=len(resources),peak_gpu_used=max(r['gpu_used'] for r in resources),
        min_host_available=min(r['host_available'] for r in resources),
        peak_rss=max(r['process']['rss'] for r in resources),passed=True,error=None)
    final=json.loads((run/'completion.json').read_text(encoding='utf-8'))
    demand(final==dict(status='complete',resources=expected_resources),'final resource receipt mismatch')
    demand(rows[-1]==dict(kind='complete',resources=expected_resources),'raw completion mismatch')
    mechanics=[r for r in rows if r['kind'].startswith('mechanics_')]
    demand(len(mechanics)==44 and all(r['passed'] for r in mechanics),'numerical gate inventory')
    for i in range(28):
        ref=load_file(run/'mechanics'/f'ffn-{i:02}.safetensors')['grouped']
        alt=load_file(run/'mechanics'/f'ffn-candidate-{i:02}.safetensors')['candidate']
        errors=[relative_l2(ref[:,j],alt[:,j]) for j in range(2)]
        recorded=next(r for r in mechanics if r.get('layer')==i)
        demand(errors==recorded['per_input_relative_l2'] and max(errors)<=.01,'FFN gate reconstruction failed')
    token_ids=json.loads((run/'token-ids.json').read_text(encoding='utf-8'))
    diagnostic_rows=[r for r in read_rows(run/'corpus.jsonl') if r['split']=='diagnostic']
    for i,row in enumerate(diagnostic_rows):
        ref=load_file(run/'mechanics'/f'reference-{i:02}.safetensors')['logits']
        alt=load_file(run/'mechanics'/f'candidate-{i:02}.safetensors')['logits']
        computed=compare_logits(ref,alt,torch.tensor([token_ids[row['id']][:128]]))
        recorded=next(r for r in mechanics if r.get('document')==row['id'])
        for key,val in computed.items():
            demand(abs(val-recorded[key])<=1e-10*max(1,abs(val)),'mechanics numerical reconstruction drift')
        demand(computed['logit_relative_l2']<=.01 and computed['mean_kl_dense_to_candidate']<=.001,
               'mechanics tolerance failed')
        del ref,alt
    episodes=[r for r in rows if r['kind']=='episode']
    scored=[r for r in episodes if not r['warmup'] and r['mode']!='target']
    demand(len(scored)==72,'missing scored episode')
    demand(len([r for r in episodes if r['warmup'] and r['mode']!='target'])==6,'warmup inventory')
    demand(len([r for r in episodes if r['mode']=='target' and not r['warmup']])==12,'target reference inventory')
    reference=json.loads((run/'reference.json').read_text(encoding='utf-8'))
    documents=[r['id'] for r in diagnostic_rows[:4]]
    demand(set(reference)==set(documents),'reference documents not frozen first four')
    expected_order=[('target',0,-1,cal_docs[0],True)]
    expected_order += [('target',0,r,d,False) for r in range(3) for d in documents]
    order={str(r):[list(c) for c in shuffled_conditions(20260914,r)] for r in range(3)}
    demand(json.loads((run/'condition-order.json').read_text(encoding='utf-8'))==order,'seeded order drift')
    for r in range(3):
        for m,b in order[str(r)]:
            if r==0:
                expected_order.append((m,b,r,cal_docs[0],True))
            expected_order += [(m,b,r,d,False) for d in documents]
    demand([(e['mode'],e['budget_mib'],e['repeat'],e['document'],e['warmup']) for e in episodes]==expected_order,
           'episode identity or execution order drift')
    seen=set()
    for ep in episodes:
        demand(ep['resources']['passed'],'resource failure')
        if ep['mode']=='target':
            if not ep['warmup']:
                demand(ep['ids']==reference[ep['document']]['ids'] and
                       ep['stop_reason']==reference[ep['document']]['stop_reason'],'target reference ledger drift')
            continue
        folder=run/'episodes'/ep['episode']
        demand(json.loads((folder/'episode.json').read_text(encoding='utf-8'))==ep,'episode differs from raw ledger')
        actual=audit_pages(folder/'pages.jsonl.gz',ep['budget_mib']*2**20,ep['mode'])
        for key,value in ep['cache'].items():
            demand(abs(actual.get(key,0)-value)<=1e-6*max(1,abs(value)),'cache aggregate mismatch: '+key)
        layers=audit_rounds(folder/'rounds.jsonl',ep)
        demand(actual['layer']==layers,'omitted or hidden draft layer calls')
        if not ep['warmup']:
            key=(ep['mode'],ep['budget_mib'],ep['repeat'],ep['document'])
            demand(key not in seen,'duplicate episode')
            seen.add(key)
            equal=ep['ids']==reference[ep['document']]['ids'] and ep['stop_reason']==reference[ep['document']]['stop_reason']
            demand(equal==ep['reference_match'],'reference identity misreported')
    expected={(m,b,r,d) for m in ('eager','lru','prefetch') for b in (128,512) for r in range(3) for d in documents}
    demand(seen==expected,'condition matrix mismatch')
    summary={}
    for mode,budget in [('target',0)]+[(m,b) for m in ('eager','lru','prefetch') for b in (128,512)]:
        group=[r for r in episodes if not r['warmup'] and r['mode']==mode and r['budget_mib']==budget]
        wall=sum(r['wall_seconds'] for r in group)
        count=sum(len(r['ids']) for r in group)
        attempted=sum(r['attempted'] for r in group)
        accepted=sum(r['accepted'] for r in group)
        h2d=sum(r.get('cache',{}).get('h2d_bytes',0) for r in group)
        summary[f'{mode}-{budget}']=dict(episodes=len(group),tokens=count,wall_seconds=wall,
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in group),
            median_wall_seconds_per_token=statistics.median(r['wall_seconds']/len(r['ids']) for r in group),
            committed_tokens_per_second=count/wall,attempted=attempted,accepted=accepted,
            acceptance=accepted/attempted if attempted else None,h2d_bytes=h2d,
            h2d_per_proposal=h2d/attempted if attempted else None,h2d_per_committed=h2d/count,
            demand_hits=sum(r.get('cache',{}).get('demand_hits',0) for r in group),
            demand_misses_per_committed=sum((r.get('cache',{}).get('demand_bytes',0)//(3*1536*256*4)) for r in group)/count,
            demand_bytes=sum(r.get('cache',{}).get('demand_bytes',0) for r in group),
            prefetch_bytes=sum(r.get('cache',{}).get('prefetch_bytes',0) for r in group),
            target_seconds=sum(r['target_seconds'] for r in group),
            reference_matches=sum(r.get('reference_match',True) for r in group))
    p,e=summary['prefetch-512'],summary['eager-512']
    saving=1-p['h2d_per_proposal']/e['h2d_per_proposal']
    checks=dict(mechanics=True,all_candidate_ids=all(r['reference_match'] for r in scored),
                acceptance_at_least_half=2*p['accepted']>=p['attempted'],
                byte_saving_at_least_tenth=10*p['h2d_bytes']*e['attempted']<=9*e['h2d_bytes']*p['attempted'])
    report=dict(source_commit=manifest['source_commit'],conditions=summary,
                resources=expected_resources,
                physical_improvement={str(b):bool(summary[f'prefetch-{b}']['reference_matches']==12 and
                    summary[f'prefetch-{b}']['h2d_per_committed']<summary[f'lru-{b}']['h2d_per_committed'] and
                    summary[f'prefetch-{b}']['median_wall_seconds']<summary[f'lru-{b}']['median_wall_seconds'])
                    for b in (128,512)},
                native_admission_checks=checks,native_numeric_predicate=all(checks.values()),
                prefetch512_vs_eager512_h2d_saving=saving,
                note='Native-only justification and a new protocol are also required by ADR 0005.')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',report)
    write_json(output/'verified-files.json',{p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file()})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(analyze(args.run,args.output),indent=2))

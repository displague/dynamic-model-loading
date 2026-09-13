"""Reproduce original completed trials and the separately registered fixed comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,message):
    if not v: raise ValueError(message)


def original(source,root):
    sys.path.insert(0,str(source/'scripts'))
    import analyze_stock_placement as audit
    import stock_placement as study
    threshold=audit.threshold_decision(root)
    require(threshold==read(root/'threshold-decision.json'),'original threshold decision')
    names=[audit.identity('threshold',t,0,r) for r,order in [(1,[8,4,2]),(2,[2,4,8])] for t in order]
    names += [audit.identity('allocation',2,n,1) for n in [36,34,32]]
    completed=[audit.audit(root/n) for n in names]; audit.serial_order(completed)
    for r in completed:
        if r['kind']=='allocation':
            require(r['allocation_eligible'],'completed candidate reserve')
            require(read(root/r['name']/'decision.json')==threshold,'original allocation input binding')
    failed=root/audit.identity('allocation',2,30,1)
    require({p.name for p in root.iterdir() if p.is_dir()}==set(names+[failed.name]),'original matrix')
    require(not (root/'allocation-decision.json').exists() and not (failed/'completion.json').exists()
            and not (failed/'rows.jsonl').exists(),'original failure must remain unscored')
    attempt=audit.attempt_identity(failed); manifest=read(failed/'manifest.json'); failure=read(failed/'failure.json')
    require(failure['phase']=='startup-validation' and failure['type']=='RuntimeError' and
            failure['message']=='startup resource bound','declared original stop')
    require(all(attempt[k]==manifest[k] for k in ['head','kind','repeat','configuration','source_sha256','decision_sha256']),
            'aborted attempt identity')
    require(read(failed/'decision.json')==threshold,'aborted prior binding')
    cfg=manifest['configuration']
    require(manifest['command']==study.command(Path(manifest['command'][0]).parent,manifest['artifacts'],cfg,8105),'aborted command')
    require(manifest['effective_runtime_environment']=={'LLAMA_TRACE':'1','GGML_OP_OFFLOAD_MIN_BATCH':'2','GGML_SCHED_DEBUG':'0'},
            'aborted environment')
    log=(failed/'server.log').read_text(encoding='utf-8',errors='replace')
    require(study.placement_evidence(log,cfg)==read(failed/'startup.json')['placement'],'aborted actual placement')
    samples=audit.prior.lines(failed/'resources.jsonl')
    require(samples and all('error' not in r for r in samples),'aborted resource receipt')
    interval=[attempt['started_monotonic'],failure['ended_monotonic']]
    audit.prior.timing_interval(*interval,3600)
    audit.process_bounds(interval,[],[r['monotonic'] for r in samples])
    require(completed[-1]['interval'][1]<=interval[0],'aborted process order')
    peak=max(r['gpu']['used'] for r in samples); host=min(r['host']['available'] for r in samples)
    require(peak>15000*2**20 and host>=2*2**30,'recorded guard cause')
    require(len({r['head'] for r in completed}|{attempt['head']})==1,'original mixed source')
    return {'threshold':threshold,'completed':completed,'allocation_decision':None,
        'aborted_unscored':{'run':failed.name,'failure':failure,'head':attempt['head'],
            'gpu_peak_bytes':peak,'host_available_min_bytes':host,'resource_samples':len(samples),
            'raw_sha256':{p.name:sha(p) for p in sorted(failed.iterdir()) if p.is_file()}},
        'limitation':'Completed allocation diagnostics are not performance scores. The startup failure neither selects an allocation nor establishes a general infeasibility boundary.'}


def initial_fixed(source,root):
    sys.path.insert(0,str(source/'scripts'))
    import fixed_attention as fixed
    import stock_placement as study
    names=[f'mechanism-t2-f{n}-r1' for n in [0,32]]
    require({p.name for p in root.iterdir() if p.is_dir()}==set(names),'aborted fixed matrix')
    candidate=source/'configs/fixed-attention-candidate.json'; records=[]
    for name in names:
        run=root/name; m=read(run/'manifest.json'); attempt=read(run/'attempt.json')
        require(read(run/'completion.json')['complete'] and not (run/'failure.json').exists(),'initial native completion')
        require(m['head']==attempt['head'] and m['kind']=='mechanism','initial mechanism identity')
        require(set(m['source_sha256'])==set(study.SOURCES) and all(sha(source/p)==h for p,h in m['source_sha256'].items()),
                'initial source binding')
        require(m['decision_sha256']==sha(candidate)==attempt['decision_sha256'],'initial declared candidate')
        require(sha(run/'decision.json')!=sha(candidate) and read(run/'decision.json')==read(candidate),'initial line-ending failure')
        rows=[json.loads(l) for l in (run/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        require([r['id'] for r in rows]==['warmup','long-code-cache'],'initial request coverage')
        records.append({'run':name,'head':m['head'],'native_requests_completed':2,
                        'declared_candidate_sha256':sha(candidate),'copied_candidate_sha256':sha(run/'decision.json')})
    try: fixed.analyze(root)
    except ValueError as exc: require(str(exc)=='decision hash','different initial audit failure')
    else: raise ValueError('initial receipt failure unexpectedly passed')
    failure=read(root/'audit-failure.json')
    require(failure['exit_code']!=0 and failure['retrospective_reproduction'] and
            failure['stderr_sha256']==sha(root/'audit-failure.stderr.log'),'initial failure reproduction')
    return {'status':'receipt-integrity failure; unscored','timed_processes_started':0,'native':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['original','initial','fixed','combined'],default='combined')
    for key in ['source','runs','initial-source','initial-runs','fixed-source','fixed-runs']: p.add_argument('--'+key,type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.part=='original': value=original(a.source.resolve(),a.runs.resolve())
    elif a.part=='initial': value=initial_fixed(a.source.resolve(),a.runs.resolve())
    elif a.part=='fixed':
        sys.path.insert(0,str(a.source.resolve()/'scripts'))
        import fixed_attention
        value=fixed_attention.analyze(a.runs.resolve())
    else:
        parts=[]
        for name,source,runs in [('original',a.source,a.runs),('initial',a.initial_source,a.initial_runs),('fixed',a.fixed_source,a.fixed_runs)]:
            target=a.output.with_name(a.output.stem+'-'+name+'.json')
            require(not target.exists(),'part output already exists')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--part',name,'--source',str(source.resolve()),
                            '--runs',str(runs.resolve()),'--output',str(target.resolve())],check=True)
            parts.append(read(target))
        first,initial,second=parts
        candidate=second['candidate']; prior=a.runs/audit_name()
        for name,digest in candidate['prior_files_sha256'].items():
            require(sha(prior/name)==digest,'fixed candidate historical receipts')
        value={'original':first,'initial_fixed':initial,'fixed':second,'source_commits':{'original':first['completed'][0]['head'],
                'initial_fixed':initial['native'][0]['head'],'fixed':second['runs'][0]['head']}}
    a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def audit_name(): return 'allocation-t2-f32-r1'


if __name__=='__main__': main()

"""Frozen fixed-32 host-FFN follow-up; no allocation search or failed-run scoring."""
import argparse
from pathlib import Path
import subprocess
import sys
import stock_placement as study
import analyze_stock_placement as audit

study.SOURCES=study.SOURCES+['scripts/fixed_attention.py','docs/fixed-attention-protocol.md','configs/fixed-attention-candidate.json']


def analyze(root):
    candidate=study.read(study.ROOT/'configs/fixed-attention-candidate.json')
    names=[audit.identity('mechanism',2,n,1) for n in [0,32]]
    mechanisms=[audit.audit(root/n) for n in names]
    ready=mechanisms[1]['graph']['all_target_attention_cuda'] and mechanisms[1]['allocation_eligible']
    runs=list(mechanisms); groups={}
    if ready:
        names+=[audit.identity('placement',2,n,r) for r,order in [(1,[0,32]),(2,[32,0])] for n in order]
        timed=[audit.audit(root/n) for n in names[2:]]; runs+=timed
        groups={str(n):audit.pooled([r for r in timed if r['configuration']['cold_ffns']==n]) for n in [0,32]}
    audit.require({p.name for p in root.iterdir() if p.is_dir()}==set(names),'fixed matrix coverage')
    audit.serial_order(runs)
    audit.require(len({r['head'] for r in runs})==1,'fixed mixed source')
    for name in names: audit.require(study.read(root/name/'decision.json')==candidate,'fixed candidate binding')
    comparisons=[]
    if ready:
        for repeat in [1,2]:
            a=next(r for r in runs if r['kind']=='placement' and r['repeat']==repeat and r['configuration']['cold_ffns']==0)
            b=next(r for r in runs if r['kind']=='placement' and r['repeat']==repeat and r['configuration']['cold_ffns']==32)
            for x,y in zip(a['rows'],b['rows']):
                comparisons.append({'repeat':repeat,'case':x['case'],'output_ids_equal':x['tokens']==y['tokens']})
    return {'candidate':candidate,'mechanism_ready':ready,'placement':groups,'cross_placement':comparisons,'runs':runs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['models','binary','output']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--worker',action='store_true'); p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--analysis-output',type=Path)
    p.add_argument('--kind',choices=['mechanism','placement']); p.add_argument('--cold-ffns',type=int,choices=[0,32])
    p.add_argument('--repeat',type=int,choices=[1,2],default=1)
    a=p.parse_args()
    for key in ['models','binary','output']: setattr(a,key,getattr(a,key).resolve())
    a.threshold=2; a.port=8105; a.decision=study.ROOT/'configs/fixed-attention-candidate.json'
    if a.worker:
        if a.kind is None or a.cold_ffns is None: raise ValueError('worker identity required')
        study.execute(a); return
    if a.analyze_only:
        study.stock.write_json(a.analysis_output or a.output/'analysis.json',analyze(a.output)); return
    a.output.mkdir(parents=True,exist_ok=False)
    def run(kind,n,repeat=1):
        name=audit.identity(kind,2,n,repeat)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--models',str(a.models),
            '--binary',str(a.binary),'--output',str(a.output/name),'--kind',kind,'--cold-ffns',str(n),
            '--repeat',str(repeat)],check=True,cwd=study.ROOT)
    for n in [0,32]: run('mechanism',n)
    fixed=audit.audit(a.output/audit.identity('mechanism',2,32,1))
    if fixed['graph']['all_target_attention_cuda'] and fixed['allocation_eligible']:
        for repeat,order in [(1,[0,32]),(2,[32,0])]:
            for n in order: run('placement',n,repeat)
    study.stock.write_json(a.output/'analysis.json',analyze(a.output))


if __name__=='__main__': main()

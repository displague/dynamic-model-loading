"""Execute the frozen stock-placement stages serially; preserve failed attempts."""
import argparse
from pathlib import Path
import subprocess
import sys
import stock_placement as study
import analyze_stock_placement as audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runs','models','binary']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['threshold','allocation','placement'],required=True)
    a=p.parse_args(); root=a.runs.resolve(); root.mkdir(parents=True,exist_ok=True)
    def run(kind,t,n=0,r=1,decision=None):
        name=audit.identity(kind,t,n,r)
        cmd=[sys.executable,str(study.ROOT/'scripts/stock_placement.py'),'--models',str(a.models.resolve()),
             '--binary',str(a.binary.resolve()),'--output',str(root/name),'--kind',kind,
             '--threshold',str(t),'--cold-ffns',str(n),'--repeat',str(r)]
        if decision: cmd+=['--decision',str(root/decision)]
        if (root/name).exists(): raise ValueError('refusing to replace existing attempt')
        print('START '+name,flush=True)
        result=subprocess.run(cmd,cwd=study.ROOT)
        if result.returncode and not (root/name/'failure.json').exists():
            raise RuntimeError('unrecorded child failure')
        return result.returncode==0
    if a.stage=='threshold':
        if any(root.iterdir()): raise ValueError('threshold requires fresh root')
        for r,order in [(1,[8,4,2]),(2,[2,4,8])]:
            for t in order:
                if not run('threshold',t,r=r): raise RuntimeError('threshold failed; retain and stop')
        study.stock.write_json(root/'threshold-decision.json',audit.threshold_decision(root))
    elif a.stage=='allocation':
        td=audit.threshold_decision(root)
        if td!=study.read(root/'threshold-decision.json'): raise ValueError('threshold decision changed')
        for n in study.ALLOCATIONS:
            if not run('allocation',td['threshold'],n,decision='threshold-decision.json'):
                raise RuntimeError('native allocation failure retained and unscored; stop without selection')
            if not audit.audit(root/audit.identity('allocation',td['threshold'],n,1))['allocation_eligible']: break
        study.stock.write_json(root/'allocation-decision.json',audit.allocation_decision(root,td['threshold']))
    else:
        td=audit.threshold_decision(root); ad=audit.allocation_decision(root,td['threshold'])
        if td!=study.read(root/'threshold-decision.json') or ad!=study.read(root/'allocation-decision.json'):
            raise ValueError('prior decisions changed')
        n=ad['cold_ffns']; t=td['threshold']
        if n is None: print('No eligible host-FFN allocation; branch ends.',flush=True); return
        for cold in [0,n]:
            if not run('mechanism',t,cold,decision='allocation-decision.json'): raise RuntimeError('mechanism failed')
        if not audit.audit(root/audit.identity('mechanism',t,n,1))['graph']['all_target_attention_cuda']:
            print('Intended attention backend absent; stop before placement timings.',flush=True); return
        for r,order in [(1,[0,n]),(2,[n,0])]:
            for cold in order:
                if not run('placement',t,cold,r,decision='allocation-decision.json'): raise RuntimeError('placement failed')
        study.stock.write_json(root/'analysis.json',audit.analyze(root))


if __name__=='__main__': main()

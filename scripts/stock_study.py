"""Run the prospectively bounded placement calibration or fixed evaluation matrix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import platform
import statistics
import subprocess
import sys

from stock_benchmark import write_json, digest, managed_process


def summarize(directory):
    path=directory/'rows.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    valid=(directory/'completion.json').exists() and all(r['resource_pass'] for r in rows)
    measured=[r for r in rows if not r['warmup']]
    startup=json.loads((directory/'startup.json').read_text(encoding='utf-8')) if (directory/'startup.json').exists() else {}
    return {'path':str(directory),'complete':valid,'rows':len(rows),
            'median_decode_ms':statistics.median(r['response']['timings']['predicted_ms'] for r in measured) if valid and measured else None,
            'sampled_gpu_peak':max([startup.get('sampled_gpu_peak') or 0]+[r['sampled_gpu_peak'] or 0 for r in rows])}


def is_resource_failure(failure, log_text, samples):
    if failure.get('type') != 'RuntimeError': return False
    errors=[s for s in samples if 'error' in s]
    if any(s.get('error_type')!='NoSuchProcess' for s in errors): return False
    message=failure.get('message','')
    if message.startswith('resource bound or telemetry failure'):
        return bool(samples) and not any('error' in s for s in samples) and any(
            s['gpu']['used'] > 15000*2**20 or s['host']['available'] < 2*2**30 for s in samples)
    # Allocation text is relevant only when the native process actually terminated at startup.
    if not message.startswith('server exited '): return False
    tail='\n'.join(log_text.lower().splitlines()[-30:])
    return any(s in tail for s in ['out of memory','failed to allocate','cannot allocate memory'])


def validate_selection(path, root):
    selection=json.loads(path.read_text(encoding='utf-8'))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if selection.get('source') != head or selection.get('evaluation_started') is not False:
        raise ValueError('calibration selection is not from the current preregistered source')
    ledger=path.parent/'driver-ledger.jsonl'
    if digest(ledger)!=selection.get('driver_ledger_sha256'):
        raise ValueError('calibration ledger digest mismatch')
    if not (path.parent/'driver-complete.json').exists():
        raise ValueError('calibration was not completed')
    recorded=[json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()]
    if set(selection['selected']) != {'target','draft05','draft15','draft32'}:
        raise ValueError('incomplete calibration selection')
    for key,chosen in selection['selected'].items():
        if not chosen['feasible']:continue
        directory=Path(chosen['calibration_result']).resolve()
        if not directory.is_relative_to(path.parent.resolve()):
            raise ValueError('selected calibration run is outside its evidence directory')
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('head') != head or manifest.get('mode') != 'calibration':
            raise ValueError('selected run has incompatible provenance')
        for field,relative in [('catalog_sha256','configs/stock-speculation-artifacts.json'),
                               ('workloads_sha256','data/stock-speculation-workloads.json')]:
            if manifest[field]!=digest(root/relative):raise ValueError('calibration substrate changed')
        matched=[r for r in recorded if Path(r['path']).resolve()==directory]
        summary=summarize(directory)
        expected_draft=None if key=='target' else key
        if (len(matched)!=1 or not matched[0]['complete'] or matched[0]['returncode']!=0
                or not summary['complete'] or summary['rows']!=3
                or summary['median_decode_ms']!=chosen['median_decode_ms']
                or any(manifest[f]!=chosen[f] for f in ['ngl','threads'])
                or manifest['draft']!=expected_draft
                or any(matched[0][f]!=manifest[f] for f in ['ngl','threads','draft','k'])
                or matched[0]['median_decode_ms']!=chosen['median_decode_ms']):
            raise ValueError('selected calibration is unsupported by retained rows')
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['calibration','evaluation'])
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    environment={'captured_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                 'processor':platform.processor(),'python':sys.version,
                 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                                '--format=csv'],text=True,timeout=15),
                 'thread_environment':{k:__import__('os').environ.get(k) for k in
                                       ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    write_json(out/'environment.json',environment)
    results=[]
    def run(name,draft,ngl,threads,k=16,mode='calibration',inputs=None,reference=False,seed=None):
        directory=out/name
        command=[sys.executable,str(root/'scripts/stock_benchmark.py'),'--models',str(args.models.resolve()),
                 '--binary',str(args.binary.resolve()),'--output',str(directory),'--ngl',str(ngl),
                 '--threads',str(threads),'--k',str(k),'--mode',mode]
        if draft:command+=['--draft',draft]
        if inputs:command+=['--inputs',str(inputs)]
        if reference:command+=['--reference']
        if seed is not None:command+=['--order-seed',str(seed)]
        print('START',name,flush=True)
        with (out/(name+'.driver.log')).open('x',encoding='utf-8') as log:
            with managed_process(command,stdout=log,stderr=subprocess.STDOUT,cwd=root) as proc:
                code=proc.wait()
        result=summarize(directory)
        result.update(name=name,draft=draft,ngl=ngl,threads=threads,k=k if draft else None,returncode=code)
        if code:result['complete']=False;result['median_decode_ms']=None
        results.append(result)
        with (out/'driver-ledger.jsonl').open('a',encoding='utf-8') as ledger:ledger.write(json.dumps(result)+'\n')
        print('END',json.dumps(result),flush=True)
        if code:
            failure=json.loads((directory/'failure.json').read_text(encoding='utf-8')) if (directory/'failure.json').exists() else {}
            log_text=(directory/'server.log').read_text(encoding='utf-8',errors='replace').lower() if (directory/'server.log').exists() else ''
            samples_path=directory/'resources.jsonl'
            samples=[json.loads(s) for s in samples_path.read_text(encoding='utf-8').splitlines()] if samples_path.exists() else []
            if not is_resource_failure(failure,log_text,samples):
                raise RuntimeError(f'Non-resource failure in {name}; stop before interpreting it as placement infeasibility')
        return result

    if args.stage=='calibration':
        selected={}
        for label,draft,start in [('target',None,48),('draft05','draft05',48),('draft15','draft15',48),('draft32','draft32',14)]:
            tested={}
            def trial(n,t=8):
                key=(n,t)
                if key not in tested:tested[key]=run(f'{label}-ngl{n}-t{t}',draft,n,t)
                return tested[key]
            first=trial(start)
            if first['complete']:
                maximum=start
                for n in range(start+1,66):
                    if not trial(n)['complete']:break
                    maximum=n
            else:
                maximum=None
                for n in range(start-1,-1,-1):
                    if trial(n)['complete']:
                        maximum=n;break
            if maximum is None:
                selected[label]={'feasible':False,'reason':'no calibrated placement within resource bounds'}
                continue
            grid=[trial(n,t) for n in sorted({maximum,max(0,maximum-4)},reverse=True) for t in [8,16,24]]
            valid=[r for r in grid if r['complete']]
            best=min(r['median_decode_ms'] for r in valid)
            tied=[r for r in valid if r['median_decode_ms'] <= best*1.01]
            chosen=min(tied,key=lambda r:(-r['ngl'],r['threads']))
            selected[label]={'feasible':True,'draft':draft,'ngl':chosen['ngl'],'threads':chosen['threads'],
                             'maximum_feasible_ngl':maximum,'calibration_result':chosen['path'],
                             'median_decode_ms':chosen['median_decode_ms']}
            write_json(out/'selection-progress.json',selected)
        selection={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                   'created_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,
                   'driver_ledger_sha256':digest(out/'driver-ledger.jsonl'),'evaluation_started':False}
        write_json(out/'selection.json',selection)
    else:
        if not args.selection:raise ValueError('evaluation requires a frozen selection receipt')
        selection=validate_selection(args.selection,root)
        write_json(out/'frozen-selection.json',selection)
        selected=selection['selected'];target=selected['target']
        if not target['feasible']:raise ValueError('no feasible target baseline')
        configs=[{'name':'target','draft':None,'ngl':target['ngl'],'threads':target['threads'],'k':16}]
        placements={(target['ngl'],target['threads'])}
        for label in ['draft05','draft15','draft32']:
            chosen=selected[label]
            if not chosen['feasible']:continue
            configs.extend({'name':f'{label}-k{k}','draft':label,'ngl':chosen['ngl'],'threads':chosen['threads'],'k':k} for k in [4,8,16])
            placement=(chosen['ngl'],chosen['threads'])
            if placement not in placements:
                configs.append({'name':f'matched-ngl{placement[0]}-t{placement[1]}','draft':None,
                                'ngl':placement[0],'threads':placement[1],'k':16})
                placements.add(placement)
        rng=random.Random(20260912)
        first=list(range(len(configs)));rng.shuffle(first)
        third=list(range(len(configs)));rng.shuffle(third)
        orders=[first,list(reversed(first)),third]
        schedule={'selection_sha256':digest(args.selection),'configs':configs,'orders':orders,
                  'prompt_seeds':[20260912,20260913,20260914]}
        write_json(out/'schedule.json',schedule)
        for mode in ['sustained','smoke']:
            r=run(f'reference-{mode}',None,target['ngl'],target['threads'],mode=mode,reference=True)
            if not r['complete']:raise RuntimeError('reference failed; evaluation stopped')
        for repeat,order in enumerate(orders):
            for index in order:
                c=configs[index]
                run(f'repeat{repeat+1}-{c["name"]}',c['draft'],c['ngl'],c['threads'],c['k'],
                    mode='sustained',inputs=out/'reference-sustained/inputs.json',seed=schedule['prompt_seeds'][repeat])
        for c in configs:
            run(f'smoke-{c["name"]}',c['draft'],c['ngl'],c['threads'],c['k'],
                mode='smoke',inputs=out/'reference-smoke/inputs.json')
    write_json(out/'driver-complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'runs':len(results)})


if __name__=='__main__':main()

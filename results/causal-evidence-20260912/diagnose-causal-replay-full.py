from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from dynamic_model_loading import causal_evidence as ce
from dynamic_model_loading.causal_evidence_study import load_inputs,controllers
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
main=Path(__file__).parent.parent;root=main/'runs/causal-evidence-worktree';run=main/'runs/causal-evidence-20260912'
cfg=json.loads((run/'config.json').read_text());parent=load_inputs(root,cfg)
rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
fitrow=next(r for r in rows if r['kind']=='fit' and r['pass_index']==2)
values=load_file(run/fitrow['tensors']['file'])
cal=[load_file(run/r['examples']['file']) for r in rows if r['kind']=='calibration' and r['pass_index']==2]
fitted=ce.flatten_models(ce.fit_models({k:torch.cat([d[k] for d in cal]) for k in cal[0]}))
print('Fits bitwise equal:',all(torch.equal(values[k],fitted[k]) for k in values),flush=True)
print('First weights:',[(k,tuple(values[k].stride()),tuple(fitted[k].stride())) for k in values if k.startswith('0.') and k.endswith('weight')],flush=True)
row=next(r for r in rows if r['kind']=='wiki' and r['mode']=='full' and r['document']==parent['development'][0])
data={k:v[:32].cuda() for k,v in load_file(run/row['trace']['file']).items()}
for name,source in [('saved_contiguous',values),('refitted_original_layout',fitted)]:
    cs=controllers(parent,ce.unflatten_models(source,28,'cuda'),'full',112,'cuda')
    mismatch=None
    for t in range(32):
        for layer,c in enumerate(cs):
            mask=c.begin(data['x'][t,layer]);observed=torch.where(data['initial'][t,layer],data['observed'][t,layer],0.)
            extra=c.acquire(observed,data['first_norm'][t,layer])
            if not torch.equal(mask,data['initial'][t,layer]) or not torch.equal(extra,data['additions'][t,layer]):
                mismatch=dict(step=t,layer=layer,initial_difference=torch.nonzero(mask!=data['initial'][t,layer]).flatten().tolist(),
                    addition_difference=torch.nonzero(extra!=data['additions'][t,layer]).flatten().tolist(),
                    detection=float(c.detection.item()),recorded_detection=float(data['detection'][t,layer].item()),
                    initial_score_max_difference=float((c.initial_scores-data['initial_scores'][t,layer]).abs().max()))
                break
            c.observe(data['observed'][t,layer],mask|extra)
        if mismatch:break
    print(name,json.dumps(mismatch),flush=True)

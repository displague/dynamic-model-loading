"""Independent artifact/resource audit and exact refit of the specialist screen."""
import json
from pathlib import Path
import math
import subprocess
import numpy as np
from safetensors.numpy import load_file
from .debt_analysis import demand, audit_cuda, read_rows
from .fault_screen import require_supervisor, ROOT
from .provenance import verify_snapshot, frozen_environment
from .specialist_screen import sha, validate_config, validate_documents
from .specialist_heads import DOMAINS, fit_bank, evaluate, summarize


def numerical(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    demand(a.shape == b.shape and a.ndim == 2 and np.isfinite(a).all() and np.isfinite(b).all(),
           'Invalid numerical-control arrays')
    rel = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30))
    same = bool(np.array_equal(a.argmax(1), b.argmax(1)))
    demand(rel <= 1e-6 and same, 'Numerical/argmax control failed')
    return dict(relative_l2=rel, argmax_identical=same)


def audit_resources(rows, receipt):
    demand(bool(rows) and all(math.isfinite(r['monotonic']) for r in rows), 'Missing resource samples')
    demand(all(a['monotonic'] <= b['monotonic'] for a,b in zip(rows,rows[1:])), 'Resource clock reversed')
    demand(all(0 < r['gpu_used'] <= 15000*2**20 and r['host_available'] >= 2048*2**20
               and r['process']['rss'] > 0 for r in rows), 'Resource cap failed')
    expected = dict(samples=len(rows), peak_gpu_used=max(r['gpu_used'] for r in rows),
        min_host_available=min(r['host_available'] for r in rows),
        peak_rss=max(r['process']['rss'] for r in rows), passed=True, error=None)
    demand(receipt == expected, 'Resource summary differs from samples')


def analyze(output):
    from .numpy_backend import set_blas_threads
    blas = set_blas_threads(4)
    output = Path(output)
    supervisor = require_supervisor(output)
    cfg = json.loads((output/'config.json').read_text())
    validate_config(cfg)
    manifest = json.loads((output/'manifest.json').read_text())
    verify_snapshot(output, manifest, 'configs/specialist-screen.json', cfg['protocol'], __file__)
    env = manifest['environment']
    demand(manifest['numpy_blas'] == blas, 'NumPy BLAS version/thread configuration changed')
    frozen_environment(env)
    demand(env['device'] == 'cuda:0' and env['cpu_threads'] == 4 and not env['tf32_matmul']
        and not env['tf32_cudnn'] and manifest['cuda_execution'] is True, 'Wrong execution settings')
    demand(manifest['models'] == cfg['models'] and manifest['documents_sha256'] == cfg['documents_sha256'],
           'Artifact bindings differ')
    inventory = json.loads((output/'files.json').read_text())
    demand(set(inventory) == {p.relative_to(output).as_posix() for p in output.rglob('*')
                             if p.is_file() and p.name != 'files.json'}, 'Raw inventory changed')
    for name, digest in inventory.items():
        demand(sha(output/name) == digest, 'Raw hash changed: '+name)
    docs = json.loads((output/'documents.json').read_text())
    validate_documents(docs)
    demand(sha(output/'documents.json') == cfg['documents_sha256'], 'Document hash differs')
    original = subprocess.check_output(['git','show',manifest['source_commit']+':'+cfg['documents']],cwd=ROOT)
    demand(original.replace(b'\r\n',b'\n') == (output/'documents.json').read_bytes().replace(b'\r\n',b'\n'),
           'Documents differ from source commit')
    tokens = json.loads((output/'tokens.json').read_text())
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    artifact = cfg['models']['draft']
    tokenizer_path = Path(snapshot_download(artifact['repo'], revision=artifact['revision'], local_files_only=True))/'tokenizer.json'
    demand(sha(tokenizer_path) == artifact['files']['tokenizer.json']['sha256'], 'Tokenizer source changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    demand(len(tokens) == 18, 'Missing token rows')
    for row, doc in zip(tokens, docs, strict=True):
        encoded = tokenizer.encode(doc['text'], add_special_tokens=False).ids
        demand(all(row[k] == doc[k] for k in ('id','domain','split')) and len(row['ids']) == 40
            and all(type(i) is int and 0 <= i < cfg['vocab'] for i in row['ids'])
            and type(row['original_tokens']) is int and row['original_tokens'] >= 40, 'Token row changed')
        demand(row['ids'] == encoded[:40] and row['original_tokens'] == len(encoded), 'Tokenization replay differs')
    expected_frames = [dict(document=i,position=p,domain=DOMAINS.index(r['domain']),fit=r['split']=='fit')
                       for i,r in enumerate(docs) for p in cfg['positions']]
    demand(json.loads((output/'frames.json').read_text()) == expected_frames, 'Frame selection changed')
    arrays = load_file(output/'inputs.safetensors')
    demand(set(arrays) == {'target','draft','hidden'}, 'Observation tensors changed')
    for name, a in arrays.items():
        demand(a.shape == (144,896 if name=='hidden' else 151936) and a.dtype == np.float32
               and np.isfinite(a).all(), 'Wrong observation shape/type')
    fit = np.array([r['fit'] for r in expected_frames], dtype=bool)
    domain = np.array([r['domain'] for r in expected_frames], dtype=np.int64)
    predictions = load_file(output/'predictions.safetensors')
    bank = fit_bank(arrays['hidden'], arrays['target'], arrays['draft'], domain, fit)
    ids = evaluate(arrays['hidden'], arrays['target'], arrays['draft'], domain, bank)
    demand(set(predictions) == {'bank','ids'} and np.array_equal(bank, predictions['bank'])
           and np.array_equal(ids, predictions['ids']), 'Refit or decision replay changed')
    report = summarize(ids, domain, fit)
    demand(report == json.loads((output/'calculation.json').read_text()), 'Gate summary changed')
    readout = load_file(output/'readout-control.safetensors')
    demand(set(readout) == {'actual','rebuilt'} and np.array_equal(readout['actual'],arrays['draft'][:8]),
           'Readout reference changed')
    controls = dict(readout=numerical(readout['actual'],readout['rebuilt']))
    repeat = load_file(output/'repeat-control.safetensors')
    demand(set(repeat) == {'target','draft','hidden'}, 'Missing repeat control')
    for name in repeat:
        controls['repeat_'+name] = numerical(arrays[name][96:104],repeat[name])
    completion = json.loads((output/'completion.json').read_text())
    demand(completion['complete'] is True and completion['new_inference'] is True and
           completion['full_suite_launched'] is False and
           0 < completion['worker_inner_seconds'] <= supervisor['worker_wall_seconds'] and
           all(math.isfinite(completion[k]) and 0 < completion[k] < completion['worker_inner_seconds']
               for k in ('fit_seconds','score_seconds')), 'Completion timing invalid')
    resources = list(read_rows(output/'resources.jsonl'))
    audit_resources(resources, completion['resources'])
    calls = json.loads((output/'calls.json').read_text())
    expected_calls = [(name,r['id'],False) for r in docs for name in ('target','draft')]
    expected_calls[26:26] = [(name,docs[12]['id'],True) for name in ('target','draft')]
    demand([(r['model'],r['document'],r['repeat']) for r in calls] == expected_calls, 'Forward order changed')
    last = resources[0]['monotonic']
    for row in calls:
        demand(row['tokens']==40 and row['scored_positions']==8 and
            last <= row['started'] < row['finished'] <= resources[-1]['monotonic'], 'Forward timing invalid')
        last = row['finished']
    allocation = json.loads((output/'allocation.json').read_text())
    for key in ('baseline_cuda','inference_cuda'):
        audit_cuda(allocation[key])
    target_bytes, draft_bytes = 6174857216, 1976131072
    demand(allocation['target_parameters_bytes'] == target_bytes and allocation['draft_parameters_bytes'] == draft_bytes
           and allocation['target_registered_buffers_bytes'] == 512
           and allocation['draft_registered_buffers_bytes'] == 256
           and allocation['model_construction_h2d_bytes'] == target_bytes+draft_bytes+768
           and allocation['baseline_cuda']['allocated_bytes'] >= target_bytes+draft_bytes+768,
           'Parameter allocation changed')
    demand(allocation['input_tensor_bytes'] == sum(a.nbytes for a in arrays.values())
        and allocation['bank_fp64_bytes'] == bank.nbytes
        and allocation['bank_fp32_equivalent_bytes'] == bank.nbytes//2
        and allocation['projection_fp64_bytes'] == 896*16*8
        and allocation['feature_fp64_bytes'] == 144*17*8, 'Tensor bytes changed')
    # 38 full forwards, 19 draft hidden readbacks; one independent readout rebuild.
    expected_copies = dict(h2d_bytes=8*8 + 38*40*8 + 8*896*4,
                          d2h_bytes=39*8*151936*4 + 19*8*896*4)
    demand(allocation['explicit_data_copies'] == expected_copies, 'Explicit transfer bytes changed')
    demand(allocation['inference_cuda']['peak_allocated_bytes'] >= target_bytes+draft_bytes
        and completion['resources']['peak_rss'] >= allocation['input_tensor_bytes']+bank.nbytes,
        'Resource peaks cannot contain owned tensors')
    return dict(**report, source_commit=manifest['source_commit'], replay_exact=True,
        numerical_controls=controls, resources=completion['resources'], allocation=allocation,
        fit_seconds=completion['fit_seconds'], score_seconds=completion['score_seconds'])

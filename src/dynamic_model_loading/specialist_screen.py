"""Minutes-scale full-vocabulary output-specialist prerequisite, never a pager."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from .fault_screen import ROOT, write, supervise

NAME = 'specialist-screen'
CONFIG = ROOT / 'configs/specialist-screen.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_documents(rows):
    from .specialist_heads import DOMAINS
    expected = [(d, 'fit', i) for d in DOMAINS for i in range(4)]
    expected += [(d, 'diagnostic', i) for d in DOMAINS for i in range(2)]
    if len(rows) != len(expected) or len({r.get('text') for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate documents')
    for row, (domain, split, index) in zip(rows, expected, strict=True):
        if (set(row) != {'id', 'domain', 'split', 'text'} or row['domain'] != domain
                or row['split'] != split or row['id'] != f'{domain}-{split}-{index}'
                or not isinstance(row['text'], str) or len(row['text']) < 160):
            raise ValueError('Document order/partition changed')


def validate_config(cfg):
    expected = dict(protocol='docs/specialist-screen-protocol.md',
        documents='configs/specialist-documents.json', positions=list(range(32, 40)),
        rank=16, ridge=32, seed=20260916, vocab=151936, draft_hidden=896,
        target_hidden=1536, worker_timeout_seconds=300, dtype='float32',
        cpu_threads=4, gpu_limit_mib=15000, host_floor_mib=2048)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen configuration changed')
    if set(cfg.get('models', {})) != {'target', 'draft'}:
        raise ValueError('Missing target or draft')
    for key, revision in [('target', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
                          ('draft', '7ae557604adf67be50417f59c2c2f167def9a775')]:
        if cfg['models'][key]['revision'] != revision:
            raise ValueError('Checkpoint revision changed')


def worker(output):
    # Heavy imports and ALL neural/fitting work are inside the supervised worker.
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .experiment import environment, cuda_memory
    from .fault_resources import Resources
    from .provenance import committed_inputs, frozen_environment
    from .specialist_heads import DOMAINS, fit_bank, evaluate, summarize
    from .specialist_analysis import numerical
    from .numpy_backend import set_blas_threads

    started = time.perf_counter()
    blas = set_blas_threads(4)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    manifest, protocol = committed_inputs(__file__, CONFIG, cfg['protocol'])
    remote = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip()
    if remote != manifest['source_commit']:
        raise ValueError('Push reviewed source before inference')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(CONFIG, output/'config.json')
    shutil.copyfile(protocol, output/'protocol.md')
    shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    docs = ROOT/cfg['documents']
    if sha(docs) != cfg['documents_sha256']:
        raise ValueError('Authored document bytes changed')
    shutil.copyfile(docs, output/'documents.json')
    rows = json.loads(docs.read_text(encoding='utf-8'))
    validate_documents(rows)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    env = environment(torch.device('cuda:0'))
    frozen_environment(env)
    paths = {}
    for name, artifact in cfg['models'].items():
        path = Path(snapshot_download(artifact['repo'], revision=artifact['revision'], local_files_only=True))
        for filename, expected in artifact['files'].items():
            if dict(bytes=(path/filename).stat().st_size, sha256=sha(path/filename)) != expected:
                raise ValueError('Model artifact changed: '+name+'/'+filename)
        paths[name] = path
    # The entire tokenizer representation, not just selected example IDs, agrees.
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
        if cfg['models']['target']['files'][filename] != cfg['models']['draft']['files'][filename]:
            raise ValueError('Tokenizer artifacts are not identical')
    tokenizers = {k:AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False)
                  for k, p in paths.items()}
    token_rows = []
    for row in rows:
        encoded = [tok.encode(row['text'], add_special_tokens=False) for tok in tokenizers.values()]
        if encoded[0] != encoded[1] or len(encoded[0]) < 40:
            raise ValueError('Incompatible or short tokenized document')
        token_rows.append(dict(id=row['id'], domain=row['domain'], split=row['split'],
                               ids=encoded[0][:40], original_tokens=len(encoded[0])))
    write(output/'tokens.json', token_rows)
    shutil.copyfile(paths['draft']/'LICENSE', output/'LICENSE-Qwen.txt')
    write(output/'NOTICE.json', dict(derived_from=cfg['models'],
        changes='Target/draft observations and fitted readout corrections; full source weights are external dependencies.',
        authored_documents='Agent-authored synthetic test fixtures, not user private files.'))
    manifest.update(environment=env, models=cfg['models'], cuda_execution=True,
                    documents_sha256=cfg['documents_sha256'], numpy_blas=blas)
    write(output/'manifest.json', manifest)
    collected = {}; calls = []; transfer = dict(h2d_bytes=0, d2h_bytes=0)
    frames = []
    with torch.inference_mode(), Resources(output/'resources.jsonl') as resources:
        models = {}
        for name in ('target', 'draft'):
            resources.boundary()
            print('Loading '+name, flush=True)
            models[name] = AutoModelForCausalLM.from_pretrained(paths[name], dtype=torch.float32,
                attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().to('cuda')
            if (models[name].config.vocab_size != cfg['vocab']
                    or models[name].config.hidden_size != cfg[name+'_hidden']
                    or {p.dtype for p in models[name].parameters()} != {torch.float32}):
                raise ValueError('Model shape/dtype changed')
        allocation = {name+'_parameters_bytes':sum(p.numel()*p.element_size() for p in model.parameters())
                      for name, model in models.items()}
        allocation.update({name+'_registered_buffers_bytes':sum(b.numel()*b.element_size() for b in model.buffers())
                           for name, model in models.items()})
        allocation['model_construction_h2d_bytes'] = sum(allocation.values())
        allocation['baseline_cuda'] = cuda_memory(torch.device('cuda:0'))
        positions = torch.tensor(cfg['positions'], device='cuda')
        transfer['h2d_bytes'] += positions.numel()*positions.element_size()
        torch.cuda.reset_peak_memory_stats()

        def forward(name, ids, document, repeat=False):
            resources.check()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            tokens = torch.tensor([ids], device='cuda')
            transfer['h2d_bytes'] += tokens.numel()*tokens.element_size()
            seen = []
            def capture(module, args):
                value = args[0][0].detach().cpu().numpy().copy()
                transfer['d2h_bytes'] += value.nbytes
                seen.append(value)
            handle = models[name].lm_head.register_forward_pre_hook(capture) if name == 'draft' else None
            try:
                result = models[name](tokens, use_cache=False, logits_to_keep=positions).logits[0]
                logits = result.detach().cpu().numpy().copy()
                transfer['d2h_bytes'] += logits.nbytes
            finally:
                if handle is not None:
                    handle.remove()
            if name == 'draft' and len(seen) != 1:
                raise ValueError('Readout capture count changed')
            torch.cuda.synchronize()
            calls.append(dict(model=name, document=document, repeat=repeat,
                started=begin, finished=time.perf_counter(), tokens=len(ids), scored_positions=8))
            return logits, seen[0] if seen else None

        for i, row in enumerate(token_rows):
            resources.boundary()
            print('Collecting '+row['id'], flush=True)
            target, _ = forward('target', row['ids'], row['id'])
            draft, hidden = forward('draft', row['ids'], row['id'])
            if i == 0:
                # A new GPU readout from the captured state; no cached logits substituted.
                h = torch.from_numpy(hidden).to('cuda')
                rebuilt = models['draft'].lm_head(h).cpu().numpy().copy()
                transfer['h2d_bytes'] += hidden.nbytes
                transfer['d2h_bytes'] += rebuilt.nbytes
                save_file({'actual':draft, 'rebuilt':rebuilt}, output/'readout-control.safetensors')
                numerical(draft, rebuilt)
            if i == 12:
                t2, _ = forward('target', row['ids'], row['id'], repeat=True)
                d2, h2 = forward('draft', row['ids'], row['id'], repeat=True)
                save_file({'target':t2, 'draft':d2, 'hidden':h2}, output/'repeat-control.safetensors')
                numerical(target, t2)
                numerical(draft, d2)
                numerical(hidden, h2)
            collected.setdefault('target', []).append(target)
            collected.setdefault('draft', []).append(draft)
            collected.setdefault('hidden', []).append(hidden)
            frames.extend([dict(document=i, position=p, domain=DOMAINS.index(row['domain']),
                                 fit=row['split']=='fit') for p in cfg['positions']])
        inputs = {k:np.concatenate(v) for k, v in collected.items()}
        domain = np.array([r['domain'] for r in frames], dtype=np.int64)
        fit = np.array([r['fit'] for r in frames], dtype=bool)
        fit_start = time.perf_counter()
        bank = fit_bank(inputs['hidden'], inputs['target'], inputs['draft'], domain, fit)
        fit_seconds = time.perf_counter()-fit_start
        score_start = time.perf_counter()
        ids = evaluate(inputs['hidden'], inputs['target'], inputs['draft'], domain, bank)
        score_seconds = time.perf_counter()-score_start
        report = summarize(ids, domain, fit)
        save_file(inputs, output/'inputs.safetensors')
        save_file({'bank':bank, 'ids':ids}, output/'predictions.safetensors')
        write(output/'frames.json', frames)
        write(output/'calls.json', calls)
        write(output/'calculation.json', report)
        allocation.update(input_tensor_bytes=sum(a.nbytes for a in inputs.values()),
            bank_fp64_bytes=bank.nbytes, bank_fp32_equivalent_bytes=bank.nbytes//2,
            projection_fp64_bytes=896*16*8, feature_fp64_bytes=144*17*8,
            explicit_data_copies=transfer, inference_cuda=cuda_memory(torch.device('cuda:0')))
        write(output/'allocation.json', allocation)
        resources.boundary()
    write(output/'completion.json', dict(complete=True, worker_inner_seconds=time.perf_counter()-started,
        fit_seconds=fit_seconds, score_seconds=score_seconds, resources=resources.receipt(),
        full_suite_launched=False, new_inference=True))
    write(output/'files.json', {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    print('Component collection and fitting complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    from .specialist_analysis import analyze
    if args.worker:
        worker(args.output)
    elif args.analyze:
        print(json.dumps(analyze(args.output), indent=2))
    else:
        report = supervise(args.output, module='dynamic_model_loading.specialist_screen', analyzer=analyze)
        print(json.dumps(report, indent=2))
        if report.get('status') in ('error', 'timeout', 'interrupted'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()

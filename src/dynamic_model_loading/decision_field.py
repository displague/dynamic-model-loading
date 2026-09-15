"""Physical, decision-sensitive interventions; no learned acquisition or target oracle."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time

from .fault_screen import ROOT, supervise, write

CONFIG = ROOT / 'configs/decision-field.json'
FROZEN = dict(protocol='docs/decision-field-protocol.md', parent='runs/fault-pager-20260914-v1',
    representation_parent='runs/refinement-screen-20260914-v1/worker',
    tokens_sha256='0ef8ea0ab583dd29f7645972623b75666c08cbf21b916e4957d186ce28727270',
    corpus_sha256='5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31',
    representation_files_sha256='f693d26dd4ca7bdcc4a66d1cfce8af4f3e165f48e2b1da8c806e91d478421dde',
    sites=[1,8,15,22], fit_documents=6, diagnostic_documents=2, prefix_tokens=4,
    positions=4, worker_timeout_seconds=300)


def validate_config(cfg):
    if json.dumps(cfg, sort_keys=True) != json.dumps(FROZEN, sort_keys=True):
        raise ValueError('Changed decision-field screen')


def branches():
    sites = FROZEN['sites']
    return [('all8', list(range(28))), *[(str(i), [i]) for i in sites],
            *[(f'{i}+{j}', [i,j]) for i,j in itertools.combinations(sites,2)], ('base', [])]


def fingerprint(cache, length):
    """Actual old FP32 KV bytes, not token counters or a placeholder digest."""
    import torch
    sha, size = hashlib.sha256(), 0
    for layer in cache.layers:
        for value in (layer.keys, layer.values):
            if value.dtype != torch.float32 or value.shape[2] < length:
                raise ValueError('Invalid KV prefix')
            data = value[:,:,:length].detach().contiguous().cpu().numpy().tobytes()
            sha.update(data)
            size += len(data)
    return sha.hexdigest(), size


def kv_storage_bytes(cache):
    """Unique underlying storage, including any suffix retained by cropped views."""
    storages = {}
    for layer in cache.layers:
        for value in (layer.keys,layer.values):
            storage = value.untyped_storage()
            storages[(str(value.device),storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def worker(output):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from .adapters import extract_ffns
    from .experiment import digest, environment, cuda_memory, load_corpus
    from .fault_generation import step, kv_bytes
    from .fault_pager_run import Ledger, move_non_ffn, unique_tensor_bytes
    from .fault_pager import metadata_bytes
    from .fault_resources import Resources
    from .metrics import relative_l2
    from .provenance import committed_inputs, frozen_environment
    from .scoped_precision import ScopedPrecision

    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance, protocol = committed_inputs(__file__, CONFIG, cfg['protocol'])
    if provenance['source_commit'] != subprocess.check_output(['git','rev-parse','origin/main'], cwd=ROOT, text=True).strip():
        raise ValueError('Push reviewed source before inference')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(CONFIG, output/'config.json')
    shutil.copyfile(protocol, output/'protocol.md')
    shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    parent, rep = ROOT/cfg['parent'], ROOT/cfg['representation_parent']
    for name,key in [('token-ids.json','tokens_sha256'), ('corpus.jsonl','corpus_sha256')]:
        if digest(parent/name) != cfg[key]:
            raise ValueError('Corpus/token drift')
        shutil.copyfile(parent/name, output/name)
    if digest(rep/'files.json') != cfg['representation_files_sha256']:
        raise ValueError('Representation inventory drift')
    inventory = json.loads((rep/'files.json').read_text(encoding='utf-8'))
    for name in ['mechanics.safetensors', *[f'constructed/layer-{i:02}.safetensors' for i in range(28)]]:
        if digest(rep/name) != inventory[name]:
            raise ValueError('Representation drift: '+name)
    shutil.copyfile(rep/'files.json', output/'representation-files.json')
    shutil.copyfile(rep/'mechanics.safetensors', output/'parent-mechanics.safetensors')
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    env = environment(torch.device('cuda:0'))
    frozen_environment(env)
    snapshot = Path(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
        revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_files_only=True))
    prior = ROOT/'runs/qwen15b-packing-pilot-20260911/manifest.json'
    if digest(prior) != '74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a':
        raise ValueError('Checkpoint manifest drift')
    checkpoint = json.loads(prior.read_text(encoding='utf-8'))['checkpoint']['files']
    for name, expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size, sha256=digest(snapshot/name)) != expected:
            raise ValueError('Checkpoint drift')
    shutil.copyfile(prior, output/'parent-manifest.json')
    write(output/'manifest.json', dict(**provenance, environment=env, checkpoint=checkpoint))
    corpus = load_corpus(output/'corpus.jsonl')
    docs = [(r['id'],'fit') for r in corpus if r['split']=='calibration'][:6]
    docs += [(r['id'],'diagnostic') for r in corpus if r['split']=='diagnostic'][:2]
    tokens = json.loads((output/'token-ids.json').read_text(encoding='utf-8'))
    events, rows = Ledger(output/'pages.jsonl.gz',True), Ledger(output/'branches.jsonl')
    phases = Ledger(output/'phases.jsonl')
    try:
        with torch.inference_mode(), Resources(output/'resources.jsonl') as resources:
            def load():
                return AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32,
                    attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval()
            start = time.perf_counter()
            target, draft = load().to('cuda'), load()
            move_non_ffn(draft,'cuda')
            tp, dp = list(target.parameters()), list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp} & {p.untyped_storage().data_ptr() for p in dp}:
                raise ValueError('Unexpected shared storage')
            known = unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            def phase(name, started, **extra):
                torch.cuda.synchronize()
                resources.boundary()
                mem = cuda_memory(torch.device('cuda'))
                row = dict(phase=name, wall_seconds=time.perf_counter()-started, cuda=mem,
                    extra_cuda_peak_bytes=mem['peak_allocated_bytes']-known, **extra)
                phases.record(row); phases.flush()
                if row['extra_cuda_peak_bytes'] > 1024*2**20:
                    raise RuntimeError('Extra CUDA cap exceeded')
            phase('model_loading', start)
            start = time.perf_counter()
            pager = ScopedPrecision(extract_ffns(draft), slots=1, group=128, check=resources.boundary)
            context = {}
            pager.cache.sink = lambda row: events.record(dict(**context, **row))
            equal = []
            for i, layer in enumerate(pager.layers):
                old = load_file(rep/f'constructed/layer-{i:02}.safetensors')
                new = {f'{j}.{name}':getattr(q,name).cpu() for j,q in enumerate(layer) for name in ('base','minimum','scale')}
                new.update({f'increment.{s}':v for s,v in enumerate(pager.host[i])})
                equal.append(set(old)==set(new) and all(torch.equal(v,old[k]) for k,v in new.items()))
                del old,new
            if not all(equal):
                raise ValueError('Constructed representation differs')
            phase('construction_and_equality', start, equal_layers=equal,
                construction_h2d_bytes=pager.construction_h2d_bytes, snapshot_d2h_bytes=pager.resident_bytes)
            write(output/'allocation.json', dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),
                charged_baseline_bytes=known, shared_bytes=0, resident_bytes=pager.resident_bytes,
                workspace_bytes=pager.workspace_bytes, cache_pool_bytes=pager.cache.pool.numel(),
                staging_bytes=pager.cache.staging.numel(), host_increments_bytes=sum(p.numel() for pair in pager.host for p in pair),
                host_metadata_bytes=metadata_bytes(vars(pager),vars(pager.cache),
                    *[vars(q) for layer in pager.layers for q in layer]),
                metadata_method='deduplicated Python containers; tensor storage separately charged; observed RSS includes other objects'))
            start = time.perf_counter()
            mech = load_file(output/'parent-mechanics.safetensors')
            checks = {}
            pager.selected = set(range(28))
            context.update(phase='mechanics', document=None, position=None, branch=None)
            for i,m in enumerate(extract_ffns(draft)):
                pager.begin_token()
                checks[f'q8.{i}'] = m(mech[f'input.{i}'].to('cuda')).cpu()
            errs = [relative_l2(mech[f'q8.{i}'],checks[f'q8.{i}']) for i in range(28)]
            save_file(checks, output/'mechanics.safetensors')
            if max(errs) > .01:
                raise RuntimeError('Numerical control failed')
            phase('mechanics',start,relative_l2=errs,correction_scalar_d2h_bytes=28*8,
                input_h2d_bytes=sum(mech[f'input.{i}'].numel()*4 for i in range(28)),
                output_d2h_bytes=sum(v.numel()*4 for v in checks.values()))
            del checks,mech
            for di,(doc,split) in enumerate(docs):
                start = time.perf_counter()
                pager.reset()
                pager.selected = set(range(28))
                context.update(phase='prefill',document=doc,position=None,branch=None)
                cache, tcache = DynamicCache(config=draft.config), DynamicCache(config=target.config)
                ids = torch.tensor([tokens[doc][:8]],device='cuda')
                for token in ids[:,:4].split(1,1):
                    step(draft,token,cache,pager)
                    step(target,token,tcache)
                phase('prefill',start,document=doc,kv_bytes=kv_bytes(cache)+kv_bytes(tcache),
                    correction_scalar_d2h_bytes=4*28*8)
                for pos in range(4):
                    start = time.perf_counter()
                    folder = output/f'frame-{di:02}-{pos}'
                    folder.mkdir()
                    length = cache.get_seq_length()
                    before, size = fingerprint(cache,length)
                    vectors = {}
                    target_start = time.perf_counter()
                    vectors['target'] = step(target,ids[:,4+pos:5+pos],tcache)[0,-1].cpu()
                    torch.cuda.synchronize()
                    target_seconds = time.perf_counter()-target_start
                    for name,sites in branches():
                        branch_start = time.perf_counter()
                        pager.selected = set(sites)
                        context.update(phase='intervention',document=doc,position=pos,branch=name)
                        traffic = pager.cache.stats['h2d_bytes']
                        logits = step(draft,ids[:,4+pos:5+pos],cache,pager)[0,-1].cpu()
                        vectors[name] = logits
                        if name != 'base':
                            cache.crop(length)
                        after, readback = fingerprint(cache,length)
                        if after != before:
                            raise RuntimeError('Old KV mutated by intervention/crop')
                        events.flush()
                        row = dict(document=doc,split=split,position=pos,branch=name,sites=sites,
                            consumed_token=tokens[doc][4+pos],base_length=length,
                            end_length=cache.get_seq_length(),prefix_before=before,prefix_after=after,
                            kv_fingerprint_d2h_bytes=readback,logit_d2h_bytes=logits.numel()*4,
                            correction_scalar_d2h_bytes=len(sites)*8,
                            draft_kv_bytes=kv_bytes(cache),target_kv_bytes=kv_bytes(tcache),
                            draft_kv_storage_bytes=kv_storage_bytes(cache),
                            target_kv_storage_bytes=kv_storage_bytes(tcache),
                            h2d_bytes=pager.cache.stats['h2d_bytes']-traffic,
                            wall_seconds=time.perf_counter()-branch_start)
                        rows.record(row); rows.flush()
                    save_file(vectors,folder/'logits.safetensors')
                    phase('frame',start,document=doc,position=pos,split=split,
                        target_seconds=target_seconds,initial_fingerprint_d2h_bytes=size,
                        target_logit_d2h_bytes=vectors['target'].numel()*4)
                    del vectors
                    print(f'Frame {di}:{pos} {split} complete',flush=True)
                del cache,tcache,ids
            write(output/'completion.json',dict(resources=resources.receipt()))
        # __exit__ makes one final resource sample; retain that final receipt.
        write(output/'final-resources.json',resources.receipt())
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        events.close(); rows.close(); phases.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args = parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    from .decision_field_analysis import analyze
    if args.worker:
        worker(args.output.resolve())
    else:
        result = analyze(args.output) if args.analyze else supervise(args.output,
            module='dynamic_model_loading.decision_field',analyzer=analyze)
        print(json.dumps(result,indent=2))
        if result.get('status') in ('error','timeout'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()

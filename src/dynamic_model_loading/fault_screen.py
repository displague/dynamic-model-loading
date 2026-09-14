"""Bounded subset of the physical-pager experiment; never launches a full matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/fault-screen.json'
TIMEOUT = 300


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def supervise(output):
    """Only the directly owned inference worker is terminated on timeout."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (output/'worker.log').open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([sys.executable, '-u', '-m',
                'dynamic_model_loading.fault_screen', '--worker', '--output', str(output/'worker')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            receipt = dict(status='error',reason='launch_error',error=repr(exc),
                           worker_wall_seconds=time.perf_counter()-started,full_suite_launched=False)
            write(output/'supervisor.json',receipt)
            write(output/'decision.json',dict(decision='inconclusive',reason='launch_error',full_suite_launched=False))
            return receipt
        try:
            code = process.wait(timeout=TIMEOUT)
            status = 'complete' if code == 0 else 'error'
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            status = 'timeout'
        except BaseException:
            process.kill()
            process.wait()
            write(output/'supervisor.json', dict(status='interrupted', timeout_seconds=TIMEOUT,
                  worker_wall_seconds=time.perf_counter()-started, full_suite_launched=False))
            write(output/'decision.json',dict(decision='inconclusive',reason='interrupted',full_suite_launched=False))
            raise
    receipt = dict(status=status, worker_exit_code=code, timeout_seconds=TIMEOUT,
                   worker_wall_seconds=time.perf_counter()-started, full_suite_launched=False)
    write(output/'supervisor.json', receipt)
    if status != 'complete':
        write(output/'decision.json', dict(decision='inconclusive', reason=status,
                                          full_suite_launched=False))
        return receipt
    analysis_started = time.perf_counter()
    try:
        report = analyze(output/'worker')
        report['worker_wall_seconds'] = receipt['worker_wall_seconds']
        report['analysis_wall_seconds'] = time.perf_counter()-analysis_started
        write(output/'summary.json', report)
        write(output/'decision.json', dict(decision=report['decision'], full_suite_launched=False))
        return report
    except BaseException as exc:
        write(output/'decision.json', dict(decision='inconclusive', reason='analysis_error',
              error=repr(exc), full_suite_launched=False))
        raise


def validate_config(cfg):
    expected = dict(protocol='docs/fault-screen-protocol.md', parent='runs/fault-pager-20260914-v1',
        index_sha256='9c0db26e82225b835aaf442ede02a76c0ec23e1762c2057a4ffa8a1afcc4754c',
        tokens_sha256='0ef8ea0ab583dd29f7645972623b75666c08cbf21b916e4957d186ce28727270',
        corpus_sha256='5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31',
        prefix_tokens=32, generation_tokens=8, diagnostic_indices=[0, 1], repetitions=1,
        budget_mib=512, conditions_by_document=[['eager','prefetch'],['prefetch','eager']],
        worker_timeout_seconds=TIMEOUT)
    if json.dumps(cfg, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('Screen differs from frozen subset')


def screen_decision(conditions):
    p, e = conditions['prefetch'], conditions['eager']
    if min(p['attempted'], e['attempted'], e['h2d_bytes']) <= 0:
        raise ValueError('Missing proposal or transfer denominator')
    checks = dict(acceptance_at_least_half=2*p['accepted'] >= p['attempted'],
        byte_saving_at_least_tenth=10*p['h2d_bytes']*e['attempted'] <=
                                   9*e['h2d_bytes']*p['attempted'])
    return dict(decision='eligible_for_expanded_protocol' if all(checks.values()) else 'stop',
        checks=checks, full_suite_launched=False, native_admission_evaluated=False,
        h2d_per_proposal_saving=1-(p['h2d_bytes']/p['attempted'])/(e['h2d_bytes']/e['attempted']))


def require_supervisor(run):
    receipt = json.loads((Path(run).parent/'supervisor.json').read_text(encoding='utf-8'))
    if (receipt.get('status') != 'complete' or receipt.get('worker_exit_code') != 0 or
            receipt.get('timeout_seconds') != TIMEOUT or receipt.get('full_suite_launched') is not False or
            not 0 < receipt.get('worker_wall_seconds',0) <= TIMEOUT+1):
        raise ValueError('Missing successful bounded supervisor receipt')
    return receipt


def full_logit_metrics(reference,candidate):
    """Numerical reconstruction, including the final position (not corpus NLL)."""
    import torch
    from .metrics import relative_l2
    if reference.ndim != 3 or reference.shape[:2] != (1,2):
        raise ValueError('Expected both full-model smoke-test positions')
    error = relative_l2(reference,candidate)  # Also rejects nonfinite values at either position.
    a,b = torch.log_softmax(reference.float(),-1),torch.log_softmax(candidate.float(),-1)
    kl = (a.exp()*(a-b)).sum(-1).clamp_min(0).double().mean().item()
    return dict(checked_logit_positions=2,logit_relative_l2=error,mean_kl_dense_to_candidate=kl)


def audit_allocation(allocation,episodes):
    expected = dict(target_parameters_bytes=6174857216,draft_cuda_parameters_bytes=1550637056,
                    draft_host_parameters_bytes=4624220160,controller_cuda_bytes=2877952,
                    controller_host_bytes=2877952,shared_bytes=0,catalogue_aliases_host=True)
    if any(allocation.get(k)!=v for k,v in expected.items()):
        raise ValueError('Physical allocation differs from frozen representation')
    for ep in episodes:
        if ep.get('staging_bytes')!=4718592:
            raise ValueError('Wrong pinned staging charge')
        max_positions = (4+4+4) if ep['warmup'] else (32+8+4)
        if not all(0<ep[k]<=57344*max_positions for k in ('target_kv_peak_bytes','draft_kv_peak_bytes')):
            raise ValueError('Missing or invalid KV charge')
        if not (0<ep['cuda']['peak_allocated_bytes']<=ep['cuda']['peak_reserved_bytes']):
            raise ValueError('Missing CUDA allocator charge')


def full_page_hooks(mlps,cache):
    return [mlp.source.register_forward_hook(lambda m,a,o,i=i: cache.record(
        dict(outcome='layer',request='execution',layer=i,selected_pages=[list(range(35))],bytes=0)))
        for i,mlp in enumerate(mlps)]


def worker(output):
    # Imports and all setup below are inside the supervisor's wall-time allowance.
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    from .adapters import extract_ffns
    from .experiment import digest, environment, cuda_memory, load_corpus
    from .fault_generation import generate
    from .fault_pager import PagedDraft, SideIndex
    from .fault_pager_run import Ledger, move_non_ffn, unique_tensor_bytes
    from .fault_resources import Resources
    from .ffn import grouped_forward, dimensions
    from .metrics import relative_l2
    from .provenance import committed_inputs, frozen_environment

    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance, protocol = committed_inputs(__file__, CONFIG, cfg['protocol'])
    remote = subprocess.check_output(['git','rev-parse','origin/main'], cwd=ROOT, text=True).strip()
    if provenance['source_commit'] != remote:
        raise ValueError('Push the reviewed source before inference')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(CONFIG, output/'config.json')
    shutil.copyfile(protocol, output/'protocol.md')
    shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    parent = ROOT/cfg['parent']
    for source, dest, key in [('calibration/index.safetensors','index.safetensors','index_sha256'),
                             ('token-ids.json','token-ids.json','tokens_sha256'),
                             ('corpus.jsonl','corpus.jsonl','corpus_sha256')]:
        if digest(parent/source) != cfg[key]:
            raise ValueError('Frozen parent artifact differs: '+source)
        shutil.copyfile(parent/source, output/dest)
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
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
        raise ValueError('Checkpoint manifest differs')
    checkpoint = json.loads(prior.read_text(encoding='utf-8'))['checkpoint']['files']
    for name, expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size, sha256=digest(snapshot/name)) != expected:
            raise ValueError('Checkpoint bytes differ: '+name)
    shutil.copyfile(prior, output/'parent-manifest.json')
    write(output/'manifest.json', dict(**provenance, environment=env, checkpoint=checkpoint))
    rows = load_corpus(output/'corpus.jsonl')
    diagnostics = [r['id'] for r in rows if r['split']=='diagnostic'][:2]
    calibration = next(r['id'] for r in rows if r['split']=='calibration')
    token_ids = json.loads((output/'token-ids.json').read_text(encoding='utf-8'))
    index_data = load_file(output/'index.safetensors')
    indexes = [SideIndex(index_data[f'centroids.{i}'], index_data[f'rankings.{i}']) for i in range(28)]
    def load_model():
        return AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32,
            attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval()

    raw = Ledger(output/'episodes.jsonl')
    try:
        with torch.inference_mode(), Resources(output/'resources.jsonl') as resources:
            print('Loading dense CUDA reference', flush=True)
            target = load_model().to('cuda')
            eos = tuple(target.generation_config.eos_token_id)
            if eos != (151645,151643) or [dimensions(m) for m in extract_ffns(target)] != [(1536,8960)]*28:
                raise ValueError('Unexpected model architecture')
            ids = {d:torch.tensor([token_ids[d]],device='cuda') for d in [calibration,*diagnostics]}
            refs = {}
            for d in [calibration,*diagnostics]:
                warm = d == calibration
                resources.boundary()
                result = generate(target, ids[d][:,:4 if warm else 32], eos,
                                  cap=4 if warm else 8, check=resources.check)
                refs[d] = dict(ids=result['ids'], stop_reason=result['stop_reason'])
                raw.record(dict(mode='target',document=d,warmup=warm,**result))
                raw.flush()
            mechanics = {}
            handles = []
            for i, mlp in enumerate(extract_ffns(target)):
                handles.append(mlp.register_forward_pre_hook(
                    lambda m,a,i=i: mechanics.update({f'input.{i}':a[0].clone()})))
            try:
                mechanics['reference.logits'] = target(ids[diagnostics[0]][:,:2],use_cache=False).logits.cpu()
            finally:
                for h in handles:
                    h.remove()
            for i, mlp in enumerate(extract_ffns(target)):
                mechanics[f'reference.{i}'] = grouped_forward(mlp,mechanics[f'input.{i}'],256).cpu()
            print('Loading paged draft; checking reduced numerical subset', flush=True)
            draft = load_model()
            pager = PagedDraft(extract_ffns(draft),indexes,page_width=256,selected_pages=27,
                               capacity_bytes=512*2**20,device='cuda:0',mode='lru',sink=lambda e:None)
            move_non_ffn(draft,'cuda')
            tp, dp = list(target.parameters()), list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp} & {p.untyped_storage().data_ptr() for p in dp}:
                raise RuntimeError('Unintended shared weights')
            for mlp,catalog in zip(extract_ffns(draft),pager.catalogs,strict=True):
                for a,b in ((mlp.gate_proj.weight,catalog._gate),(mlp.up_proj.weight,catalog._up),
                            (mlp.down_proj.weight,catalog._down)):
                    if a.is_cuda or a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr():
                        raise RuntimeError('Catalogue does not alias evacuated host weights')
            write(output/'allocation.json', dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),
                controller_cuda_bytes=pager.controller_bytes,controller_host_bytes=sum(i.bytes for i in indexes),
                shared_bytes=0,catalogue_aliases_host=True,host_metadata_bytes=pager.host_metadata_bytes,
                metadata_method='deduplicated sys.getsizeof excluding tensor storage',
                cuda=cuda_memory(torch.device('cuda'))))
            mechanics_pages = Ledger(output/'mechanics-pages.jsonl.gz',True)
            pager.cache.sink = mechanics_pages.record
            pager.full = True
            full_handles = full_page_hooks(extract_ffns(draft),pager.cache)
            errors = []
            for i,mlp in enumerate(extract_ffns(draft)):
                resources.boundary()
                alt = mlp(mechanics[f'input.{i}']).cpu()
                mechanics[f'candidate.{i}'] = alt
                errors.extend(relative_l2(mechanics[f'reference.{i}'][:,j],alt[:,j]) for j in range(2))
            mechanics['candidate.logits'] = draft(ids[diagnostics[0]][:,:2],use_cache=False).logits.cpu()
            for handle in full_handles:
                handle.remove()
            metrics = full_logit_metrics(mechanics['reference.logits'],mechanics['candidate.logits'])
            save_file({k:v.cpu().contiguous() for k,v in mechanics.items()},output/'mechanics.safetensors')
            ok = max(errors)<=.01 and metrics['logit_relative_l2']<=.01 and metrics['mean_kl_dense_to_candidate']<=.001
            write(output/'mechanics.json',dict(passed=ok,ffn_relative_l2=errors,**metrics))
            del mechanics
            pager.full = False
            pager.reset()
            mechanics_pages.close()
            if not ok:
                raise RuntimeError('Reduced numerical checks failed')
            pairs = [(calibration,'eager'),(calibration,'prefetch')]
            pairs += [(d,m) for d,order in zip(diagnostics,cfg['conditions_by_document'],strict=True) for m in order]
            for number,(d,mode) in enumerate(pairs):
                warm = d == calibration
                folder = output/f'episode-{number}'
                folder.mkdir()
                pages, rounds = Ledger(folder/'pages.jsonl.gz',True), Ledger(folder/'rounds.jsonl')
                pager.cache.sink = pages.record
                pager.cache.mode = mode
                pager.reset()
                resources.boundary()
                torch.cuda.reset_peak_memory_stats()
                print(f'Start {number}: {mode}, warmup={warm}',flush=True)
                try:
                    result = generate(target,ids[d][:,:4 if warm else 32],eos,cap=4 if warm else 8,
                                      draft=draft,pager=pager,record=rounds.record,check=resources.check)
                    cleanup = time.perf_counter()
                    pager.cache.clear()
                    pages.flush()
                    rounds.flush()
                    elapsed = time.perf_counter()-cleanup
                    result['wall_seconds'] += elapsed
                    result['decode_seconds'] += elapsed
                    resources.boundary()
                    same = result['ids']==refs[d]['ids'] and result['stop_reason']==refs[d]['stop_reason']
                    row = dict(episode=folder.name,mode=mode,document=d,warmup=warm,
                        reference_match=same,**result,cache=dict(pager.cache.stats),
                        staging_bytes=pager.cache.staging.numel()*pager.cache.staging.element_size(),
                        cuda=cuda_memory(torch.device('cuda')),resources=resources.receipt())
                    raw.record(row)
                    raw.flush()
                    write(folder/'episode.json',row)
                    print(f'End {number}: {result["wall_seconds"]:.3f}s; accepted {result["accepted"]}/{result["attempted"]}; match={same}',flush=True)
                    if not same:
                        raise RuntimeError('Committed IDs or stop reason differ from scalar reference')
                finally:
                    pages.close()
                    rounds.close()
        write(output/'completion.json',dict(resources=resources.receipt()))
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        raw.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p)
          for p in output.rglob('*') if p.is_file()})


def analyze(run):
    import torch
    from safetensors.torch import load_file
    from .experiment import digest, load_corpus
    from .fault_pager_analysis import audit_pages, audit_rounds, demand, read_rows
    from .metrics import relative_l2
    from .provenance import verify_snapshot, frozen_environment

    torch.set_num_threads(4)
    run = Path(run)
    require_supervisor(run)
    demand((run/'completion.json').is_file() and not (run/'failure.json').exists(),'incomplete screen')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run,manifest,'configs/fault-screen.json','docs/fault-screen-protocol.md',__file__)
    frozen_environment(manifest['environment'])
    env = manifest['environment']
    demand(env['device']=='cuda:0' and env['cpu_threads']==4 and
           not env['tf32_matmul'] and not env['tf32_cudnn'] and 'gpu' in env,'execution environment drift')
    demand(digest(run/'parent-manifest.json')=='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a',
           'checkpoint parent manifest drift')
    demand(manifest['checkpoint']==json.loads((run/'parent-manifest.json').read_text(encoding='utf-8'))['checkpoint']['files'],
           'checkpoint inventory drift')
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    inventory = json.loads((run/'files.json').read_text(encoding='utf-8'))
    demand(set(inventory)=={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p!=run/'files.json'},'file inventory differs')
    for name, expected in inventory.items():
        demand(digest(run/name)==expected,'raw file hash differs: '+name)
    for name,key in [('index.safetensors','index_sha256'),('token-ids.json','tokens_sha256'),('corpus.jsonl','corpus_sha256')]:
        demand(digest(run/name)==cfg[key],'frozen parent artifact hash')
    corpus = load_corpus(run/'corpus.jsonl')
    ds = [r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    cal = next(r['id'] for r in corpus if r['split']=='calibration')
    rows = list(read_rows(run/'episodes.jsonl'))
    expected = [('target',cal,True),('target',ds[0],False),('target',ds[1],False),
                ('eager',cal,True),('prefetch',cal,True),('eager',ds[0],False),
                ('prefetch',ds[0],False),('prefetch',ds[1],False),('eager',ds[1],False)]
    demand([(r['mode'],r['document'],r['warmup']) for r in rows]==expected,'screen matrix/order differs')
    refs = {r['document']:r for r in rows[:3]}
    allocation = json.loads((run/'allocation.json').read_text(encoding='utf-8'))
    audit_allocation(allocation,rows[3:])
    paths = {}
    metadata_peak = 0
    for number, row in enumerate(rows[3:]):
        folder = run/f'episode-{number}'
        demand(row==json.loads((folder/'episode.json').read_text(encoding='utf-8')),'episode receipt drift')
        ref = refs[row['document']]
        demand(row['reference_match'] and row['ids']==ref['ids'] and row['stop_reason']==ref['stop_reason'],'reference mismatch')
        layers = audit_rounds(folder/'rounds.jsonl',row,prefix_tokens=4 if row['warmup'] else 32,
                              generation_tokens=4 if row['warmup'] else 8)
        stats = audit_pages(folder/'pages.jsonl.gz',512*2**20,row['mode'])
        metadata_peak = max(metadata_peak,stats['peak_host_metadata_bytes'])
        demand(stats['layer']==layers,'page/verification work does not reconcile')
        for key in ('h2d_bytes','demand_bytes','prefetch_bytes','demand_hits','prefetch_hits','load','hit',
                    'release','cancelled_prefetch','layer','peak_payload_bytes','transfer_wall_ms','transfer_cuda_ms'):
            demand(stats.get(key,0)==row['cache'].get(key,0),'cache counter drift: '+key)
        path = [(r['proposed'],r['target_predictions'],r['committed']) for r in read_rows(folder/'rounds.jsonl')]
        if row['document'] in paths:
            demand(paths[row['document']]==path,'selection-matched paths differ')
        paths[row['document']] = path
    tensors = load_file(run/'mechanics.safetensors')
    mechanical_pages = audit_pages(run/'mechanics-pages.jsonl.gz',512*2**20,'lru',selected_pages=35)
    demand(mechanical_pages['layer']==56,'missing numerical page executions')
    demand(set(tensors)=={f'{kind}.{i}' for i in range(28) for kind in ('input','reference','candidate')} |
           {'reference.logits','candidate.logits'},'numerical tensor inventory differs')
    demand(all(tensors[f'{kind}.{i}'].shape==(1,2,1536) for i in range(28)
               for kind in ('input','reference','candidate')),'numerical FFN shape differs')
    errors = [relative_l2(tensors[f'reference.{i}'][:,j],tensors[f'candidate.{i}'][:,j])
              for i in range(28) for j in range(2)]
    metrics = full_logit_metrics(tensors['reference.logits'],tensors['candidate.logits'])
    demand(max(errors)<=.01 and metrics['logit_relative_l2']<=.01 and metrics['mean_kl_dense_to_candidate']<=.001,'numerical screen failed')
    numerical = json.loads((run/'mechanics.json').read_text(encoding='utf-8'))
    demand(numerical==dict(passed=True,ffn_relative_l2=errors,**metrics),'numerical receipt drift')
    samples = list(read_rows(run/'resources.jsonl'))
    demand(bool(samples),'missing resources')
    demand(all(r['gpu_used']<=15000*2**20 and r['host_available']>=2048*2**20 for r in samples),'resource limits failed')
    resource = dict(samples=len(samples),peak_gpu_used=max(r['gpu_used'] for r in samples),
        min_host_available=min(r['host_available'] for r in samples),peak_rss=max(r['process']['rss'] for r in samples),passed=True,error=None)
    demand(json.loads((run/'completion.json').read_text(encoding='utf-8'))['resources']==resource,'final resources differ')
    conditions = {}
    for mode in ('target','eager','prefetch'):
        group = [r for r in rows if not r['warmup'] and r['mode']==mode]
        attempted, accepted = sum(r['attempted'] for r in group), sum(r['accepted'] for r in group)
        h2d = sum(r.get('cache',{}).get('h2d_bytes',0) for r in group)
        count, wall = sum(len(r['ids']) for r in group), sum(r['wall_seconds'] for r in group)
        conditions[mode] = dict(episodes=len(group),tokens=count,wall_seconds=wall,
            committed_tokens_per_second=count/wall,attempted=attempted,accepted=accepted,
            acceptance=accepted/attempted if attempted else None,h2d_bytes=h2d,
            h2d_per_proposal=h2d/attempted if attempted else None,h2d_per_committed=h2d/count,
            target_seconds=sum(r['target_seconds'] for r in group))
    return dict(source_commit=manifest['source_commit'],conditions=conditions,resources=resource,
        allocation=allocation,peak_host_metadata_bytes=metadata_peak,
        pinned_staging_bytes=max(r['staging_bytes'] for r in rows[3:]),
        target_kv_peak_bytes=max(r['target_kv_peak_bytes'] for r in rows),
        draft_kv_peak_bytes=max(r['draft_kv_peak_bytes'] for r in rows),
        peak_page_payload_bytes=max(r['cache']['peak_payload_bytes'] for r in rows[3:]),
        max_ffn_relative_l2=max(errors),logit_relative_l2=metrics['logit_relative_l2'],
        logit_kl=metrics['mean_kl_dense_to_candidate'],all_candidate_ids_match=True,
        warmup_wall_seconds=sum(r['wall_seconds'] for r in rows if r['warmup']),
        note='Known-negative workflow validation on reused diagnostics; not novel policy or native admission.',
        **screen_decision(conditions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--analyze',action='store_true',help='Print replayed summary for an existing worker directory')
    args = parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    if args.worker:
        worker(args.output.resolve())
    else:
        report = analyze(args.output) if args.analyze else supervise(args.output)
        print(json.dumps(report,indent=2))
        if report.get('status') in ('timeout','error'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()

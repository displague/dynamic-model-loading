"""Frozen five-minute screen for resident-base correction-debt acquisition."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .fault_screen import ROOT,supervise,write


CONFIG = ROOT/'configs/debt-screen-v2.json'
FROZEN = dict(protocol='docs/debt-screen-protocol-v2.md',parent='runs/fault-pager-20260914-v1',
    index_sha256='9c0db26e82225b835aaf442ede02a76c0ec23e1762c2057a4ffa8a1afcc4754c',
    tokens_sha256='0ef8ea0ab583dd29f7645972623b75666c08cbf21b916e4957d186ce28727270',
    corpus_sha256='5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31',
    bits=2,quant_group=128,sketch_rank=16,page_width=256,max_corrections=4,extra_cuda_mib=768,
    prefix_tokens=8,generation_tokens=8,repetitions=1,worker_timeout_seconds=300,
    conditions=['dense_stream','base','fixed','debt'])


def validate_config(cfg):
    if json.dumps(cfg,sort_keys=True)!=json.dumps(FROZEN,sort_keys=True):
        raise ValueError('Changed frozen debt screen')


def charged_generate(call,cleanup=lambda:None):
    started=time.perf_counter()
    result=call()
    cleanup_started=time.perf_counter()
    cleanup()
    finished=time.perf_counter()
    generation=result['wall_seconds']
    result.update(charged_started_monotonic=started,charged_finished_monotonic=finished,
        cleanup_started_monotonic=cleanup_started,cleanup_seconds=finished-cleanup_started,
        generation_wall_seconds=generation,wall_seconds=finished-started,
        wrapper_seconds=finished-started-generation-(finished-cleanup_started))
    return result


def worker(output):
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file,save_file
    from transformers import AutoModelForCausalLM
    from .adapters import extract_ffns
    from .experiment import digest,environment,cuda_memory,load_corpus
    from .fault_generation import generate
    from .fault_pager import metadata_bytes
    from .fault_pager_run import Ledger,move_non_ffn,unique_tensor_bytes
    from .fault_resources import Resources
    from .fault_screen import full_logit_metrics
    from .ffn import grouped_forward,dimensions
    from .metrics import relative_l2
    from .provenance import committed_inputs,frozen_environment
    from .residual_debt import ResidualDraft
    from .debt_analysis import audit_allocation

    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance,protocol = committed_inputs(__file__,CONFIG,cfg['protocol'])
    if provenance['source_commit']!=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip():
        raise ValueError('Push reviewed source before inference')
    output.mkdir(parents=True,exist_ok=False)
    for source,dest in ((CONFIG,'config.json'),(protocol,'protocol.md')):
        shutil.copyfile(source,output/dest)
    shutil.copytree(Path(__file__).parent,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    parent = ROOT/cfg['parent']
    for source,dest,key in [('calibration/index.safetensors','index.safetensors','index_sha256'),
                            ('token-ids.json','token-ids.json','tokens_sha256'),('corpus.jsonl','corpus.jsonl','corpus_sha256')]:
        if digest(parent/source)!=cfg[key]:
            raise ValueError('Parent artifact differs: '+source)
        shutil.copyfile(parent/source,output/dest)
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    env = environment(torch.device('cuda:0'))
    frozen_environment(env)
    snapshot = Path(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
        revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=True))
    prior = ROOT/'runs/qwen15b-packing-pilot-20260911/manifest.json'
    if digest(prior)!='74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a':
        raise ValueError('Checkpoint manifest differs')
    checkpoint = json.loads(prior.read_text(encoding='utf-8'))['checkpoint']['files']
    for name,expected in checkpoint.items():
        if dict(bytes=(snapshot/name).stat().st_size,sha256=digest(snapshot/name))!=expected:
            raise ValueError('Checkpoint bytes differ: '+name)
    shutil.copyfile(prior,output/'parent-manifest.json')
    write(output/'manifest.json',dict(**provenance,environment=env,checkpoint=checkpoint))
    corpus = load_corpus(output/'corpus.jsonl')
    ds = [r['id'] for r in corpus if r['split']=='diagnostic'][:2]
    cal = next(r['id'] for r in corpus if r['split']=='calibration')
    tokens = json.loads((output/'token-ids.json').read_text(encoding='utf-8'))
    index = load_file(output/'index.safetensors')
    def load_model():
        return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).eval()
    raw = Ledger(output/'episodes.jsonl')
    try:
        with torch.inference_mode(),Resources(output/'resources.jsonl') as resources:
            print('Loading dense reference',flush=True)
            target = load_model().to('cuda')
            eos = tuple(target.generation_config.eos_token_id)
            if eos!=(151645,151643) or [dimensions(m) for m in extract_ffns(target)]!=[(1536,8960)]*28:
                raise ValueError('Unexpected target architecture')
            ids = {d:torch.tensor([tokens[d]],device='cuda') for d in [cal,*ds]}
            references = {}
            for d in [cal]:
                warm = d==cal
                resources.boundary()
                result = charged_generate(lambda:generate(target,ids[d][:,:2 if warm else 8],eos,
                    cap=4 if warm else 8,check=resources.check))
                references[d] = dict(ids=result['ids'],stop_reason=result['stop_reason'])
                raw.record(dict(mode='target',document=d,warmup=warm,**result))
                raw.flush()
            mechanics = {}
            handles = [m.register_forward_pre_hook(lambda m,a,i=i:mechanics.update({f'input.{i}':a[0].detach().clone()}))
                       for i,m in enumerate(extract_ffns(target))]
            try:
                mechanics['reference.logits'] = target(ids[ds[0]][:,:2],use_cache=False).logits.cpu()
            finally:
                for h in handles:
                    h.remove()
            for i,m in enumerate(extract_ffns(target)):
                mechanics[f'reference.{i}'] = grouped_forward(m,mechanics[f'input.{i}'],256).cpu()
                mechanics[f'input.{i}'] = mechanics[f'input.{i}'].cpu()
            print('Constructing resident two-bit base and error sketches',flush=True)
            draft = load_model()
            move_non_ffn(draft,'cuda')
            tp,dp = list(target.parameters()),list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in tp}&{p.untyped_storage().data_ptr() for p in dp}:
                raise RuntimeError('Unintended shared parameters')
            non_ffn_baseline = torch.cuda.memory_allocated()
            charged_baseline=unique_tensor_bytes(tp)+unique_tensor_bytes([p for p in dp if p.is_cuda])
            baseline_cuda=cuda_memory(torch.device('cuda'))
            torch.cuda.reset_peak_memory_stats()
            construction_start = time.perf_counter()
            pager = ResidualDraft(extract_ffns(draft),[index[f'centroids.{i}'] for i in range(28)],
                device='cuda:0',page_sink=lambda e:None,check=resources.boundary)
            constructed = output/'constructed'
            constructed.mkdir()
            save_file({'projection':pager.projection.cpu()},constructed/'projection.safetensors')
            for i,layer in enumerate(pager.layers):
                save_file({k:v.cpu().contiguous() for k,v in layer.tensors().items()},constructed/f'layer-{i:02}.safetensors')
            constructed_hashes = {p.name:digest(p) for p in constructed.iterdir()}
            account = dict(target_parameters_bytes=unique_tensor_bytes(tp),
                draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in dp if p.is_cuda]),
                draft_host_parameters_bytes=unique_tensor_bytes([p for p in dp if not p.is_cuda]),
                shared_bytes=0,catalogue_aliases_host=True,resident_bytes=pager.resident_bytes,
                workspace_bytes=pager.workspace_bytes,non_ffn_baseline_allocated=non_ffn_baseline,
                baseline_cuda=baseline_cuda,
                charged_baseline_bytes=charged_baseline,
                construction_h2d_bytes=pager.construction_h2d_bytes,
                construction_d2h_bytes=pager.resident_bytes,
                construction_wall_seconds=time.perf_counter()-construction_start,
                host_calibration_index_bytes=sum(t.numel()*t.element_size() for t in index.values()),
                construction_cuda=cuda_memory(torch.device('cuda')),constructed_sha256=constructed_hashes)
            write(output/'allocation.json',account)
            audit_allocation(account)
            resources.boundary()
            # Freeze the constructed candidate before even scalar-reference scoring.
            for d in ds:
                result=charged_generate(lambda:generate(target,ids[d][:,:8],eos,cap=8,check=resources.check))
                references[d]=dict(ids=result['ids'],stop_reason=result['stop_reason'])
                raw.record(dict(mode='target',document=d,warmup=False,**result))
                raw.flush()
                resources.boundary()
            print('Checking all-page reconstruction and local policy errors',flush=True)
            pages = Ledger(output/'mechanics-pages.jsonl.gz',True)
            policy = Ledger(output/'mechanics-policy.jsonl.gz',True)
            pager.cache.sink,pager.policy_sink = pages.record,policy.record
            errors = {}
            try:
                for mode in ('complete','base','fixed','debt'):
                    pager.mode=mode
                    errors[mode]=[]
                    for i,m in enumerate(extract_ffns(draft)):
                        resources.boundary()
                        alt = m(mechanics[f'input.{i}'].to('cuda')).cpu()
                        mechanics[f'{mode}.{i}']=alt
                        errors[mode].extend(relative_l2(mechanics[f'reference.{i}'][:,j],alt[:,j]) for j in range(2))
                pager.mode='complete'
                mechanics['complete.logits']=draft(ids[ds[0]][:,:2],use_cache=False).logits.cpu()
                metrics=full_logit_metrics(mechanics['reference.logits'],mechanics['complete.logits'])
                save_file({k:v.contiguous() for k,v in mechanics.items()},output/'mechanics.safetensors')
                passed=(max(errors['complete'])<=.01 and metrics['logit_relative_l2']<=.01 and
                        metrics['mean_kl_dense_to_candidate']<=.001)
                write(output/'mechanics.json',dict(passed=passed,ffn_relative_l2=errors,**metrics))
                pager.reset()
            finally:
                pages.close()
                policy.close()
            del mechanics
            if not passed:
                raise RuntimeError('All-page correction reconstruction failed')
            pairs = [(cal,m) for m in cfg['conditions']]
            pairs += [(d,m) for i,d in enumerate(ds) for m in (cfg['conditions'] if i==0 else cfg['conditions'][::-1])]
            for number,(d,mode) in enumerate(pairs):
                warm=d==cal
                folder=output/f'episode-{number:02}'
                folder.mkdir()
                pages=Ledger(folder/'pages.jsonl.gz',True)
                policy=Ledger(folder/'policy.jsonl.gz',True)
                rounds=Ledger(folder/'rounds.jsonl')
                pager.reset()
                pager.mode=mode
                pager.cache.sink,pager.policy_sink=pages.record,policy.record
                resources.boundary()
                torch.cuda.reset_peak_memory_stats()
                print(f'Start {number}: {mode}, warmup={warm}',flush=True)
                try:
                    def cleanup():
                        pager.cache.clear()
                        pages.flush()
                        policy.flush()
                        rounds.flush()
                    result=charged_generate(lambda:generate(target,ids[d][:,:2 if warm else 8],eos,
                        cap=4 if warm else 8,draft=draft,pager=pager,record=rounds.record,
                        check=resources.check),cleanup)
                    resources.boundary()
                    cuda=cuda_memory(torch.device('cuda'))
                    extra=cuda['peak_allocated_bytes']-charged_baseline
                    match=result['ids']==references[d]['ids'] and result['stop_reason']==references[d]['stop_reason']
                    row=dict(episode=folder.name,mode=mode,document=d,warmup=warm,reference_match=match,
                        **result,cache=dict(pager.cache.stats),cuda=cuda,extra_cuda_peak_bytes=extra,
                        staging_bytes=pager.cache.staging.numel()*4,resources=resources.receipt(),
                        host_catalogue_metadata_bytes=metadata_bytes(pager.catalogs,pager.cache._entries))
                    raw.record(row)
                    raw.flush()
                    write(folder/'episode.json',row)
                    print(f'End {number}: {result["wall_seconds"]:.3f}s; {result["accepted"]}/{result["attempted"]} accepted; match={match}',flush=True)
                    if not match or extra>cfg['extra_cuda_mib']*2**20:
                        raise RuntimeError('Reference mismatch or extra-CUDA allowance exceeded')
                finally:
                    pages.close()
                    policy.close()
                    rounds.close()
        write(output/'completion.json',dict(resources=resources.receipt()))
    except BaseException as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        raw.close()
    write(output/'files.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.worker and args.analyze:
        parser.error('Conflicting modes')
    if args.worker:
        worker(args.output.resolve())
    else:
        from .debt_analysis import analyze
        result=(analyze(args.output) if args.analyze else supervise(args.output,
                module='dynamic_model_loading.debt_screen',analyzer=analyze))
        print(json.dumps(result,indent=2))
        if result.get('status') in ('timeout','error'):
            raise SystemExit(1)


if __name__=='__main__':
    main()

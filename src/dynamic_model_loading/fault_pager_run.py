"""Run the frozen physical-pager experiment on the pinned CUDA checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import extract_ffns
from .experiment import digest, write_json, environment, cuda_memory, load_corpus
from .fault_generation import generate
from .fault_pager import PagedDraft, SideIndex
from .fault_pager_study import validate_config, shuffled_conditions
from .fault_resources import Resources
from .ffn import dimensions, grouped_forward
from .metrics import compare_logits, relative_l2
from .provenance import committed_inputs, frozen_environment


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = 'docs/fault-pager-amendment-1.md'


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


class Ledger:
    def __init__(self, path, compressed=False):
        self.out = (gzip.open(path, 'xt', encoding='utf-8', compresslevel=1) if compressed
                    else Path(path).open('x', encoding='utf-8'))

    def record(self, value):
        self.out.write(json.dumps(value, default=json_default, allow_nan=False, separators=(',', ':'))+'\n')

    def flush(self):
        self.out.flush()

    def close(self):
        self.out.close()


def unique_tensor_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def move_non_ffn(model, device):
    for name, parameter in model.named_parameters():
        if '.mlp.' not in name:
            parameter.data = parameter.data.to(device)
    for name, buffer in model.named_buffers():
        if '.mlp.' not in name:
            buffer.data = buffer.data.to(device)


@torch.inference_mode()
def calibrate(target, rows, ids, output, raw, resources):
    mlps = extract_ffns(target)
    vectors, scores = [[] for _ in mlps], [[] for _ in mlps]
    handles = []
    for i, mlp in enumerate(mlps):
        norms = mlp.down_proj.weight.float().norm(dim=0)
        def capture_x(module, args, i=i):
            vectors[i].append(args[0][0, :128].float().cpu())
        def capture_z(module, args, i=i, norms=norms):
            z = args[0][0, :128]
            score = (z.abs() * norms).reshape(128, 35, 256).sum(-1)
            scores[i].append(score.cpu())
        handles.append(mlp.register_forward_pre_hook(capture_x))
        handles.append(mlp.down_proj.register_forward_pre_hook(capture_z))
    try:
        for r, row in enumerate(rows):
            resources.boundary()
            target(ids[row['id']][:, :129], use_cache=False)
            raw.record(dict(kind='calibration_document', document=row['id'], positions=128))
            raw.flush()
            if (r+1) % 10 == 0:
                print(f'Calibration {r+1}/80', flush=True)
    finally:
        for h in handles:
            h.remove()
    output.mkdir()
    indexes, index_tensors, positions = [], {}, {}
    for layer in range(len(mlps)):
        x, score = torch.cat(vectors[layer]), torch.cat(scores[layer])
        vectors[layer].clear()
        scores[layer].clear()
        save_file({'inputs': x, 'scores': score}, output/f'layer-{layer:02}.safetensors')
        index = SideIndex.build(x, score, 16)
        indexes.append(index)
        index_tensors[f'centroids.{layer}'] = index.centroids.contiguous()
        index_tensors[f'rankings.{layer}'] = index.rankings.contiguous()
        positions[str(layer)] = [dict(document=rows[i//128]['id'], position=i%128)
                                  for i in index.source_positions]
        del x, score
    save_file(index_tensors, output/'index.safetensors')
    write_json(output/'index.json', dict(sha256=digest(output/'index.safetensors'), positions=positions,
                observations_per_layer=10240, layers=28, medoids=16,
                calibration_files={p.name:digest(p) for p in output.glob('layer-*.safetensors')}))
    raw.record(dict(kind='index_frozen', sha256=digest(output/'index.safetensors'), positions=positions))
    raw.flush()
    return indexes


@torch.inference_mode()
def run(output, config_path):
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(cfg)
    provenance, protocol = committed_inputs(__file__, config_path, cfg['protocol'])
    commit = provenance['source_commit']
    remote = subprocess.check_output(['git','rev-parse','origin/main'], cwd=ROOT, text=True).strip()
    if commit != remote:
        raise ValueError('Push the reviewed source before inference')
    corpus_path = ROOT/cfg['corpus']
    if digest(corpus_path) != cfg['corpus_sha256']:
        raise ValueError('Corpus hash mismatch')
    rows = load_corpus(corpus_path)
    calibration = [r for r in rows if r['split']=='calibration']
    diagnostics = [r for r in rows if r['split']=='diagnostic']
    if (len(calibration), len(diagnostics)) != (80,16):
        raise ValueError('Wrong corpus inventory')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output/'config.json')
    shutil.copyfile(protocol, output/'protocol.md')
    shutil.copyfile(ROOT/AMENDMENT, output/'amendment.md')
    shutil.copyfile(corpus_path, output/'corpus.jsonl')
    shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    raw = Ledger(output/'results.jsonl')
    write_json(output/'started.json', dict(utc=datetime.now(timezone.utc).isoformat(), **provenance))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        torch.set_num_threads(4)
        torch.manual_seed(20260914)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        env = environment(torch.device('cuda:0'))
        frozen_environment(env)
        snapshot = Path(snapshot_download(cfg['model'], revision=cfg['revision'], local_files_only=True))
        checkpoint = {p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
                      for p in snapshot.iterdir() if p.is_file()}
        parent_path = ROOT/'runs/qwen15b-packing-pilot-20260911/manifest.json'
        if digest(parent_path) != '74349fb6fdbed4f0591add5f6d901d59787b09a9f4a776c79f316b6743de160a':
            raise ValueError('Prior checkpoint manifest hash mismatch')
        parent = json.loads(parent_path.read_text(encoding='utf-8'))
        shutil.copyfile(parent_path,output/'parent-manifest.json')
        for name, expected in parent['checkpoint']['files'].items():
            if checkpoint.get(name) != expected:
                raise ValueError('Checkpoint differs from prior dense reference: '+name)
        if checkpoint['model.safetensors']['sha256'] != 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee':
            raise ValueError('Unexpected weight artifact')
        tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        cpu_ids = {r['id']:tokenizer(r['text'],add_special_tokens=False,return_tensors='pt')['input_ids'][:, :129]
                   for r in rows}
        if any(v.shape[1] != 129 for v in cpu_ids.values()):
            raise ValueError('Document too short for frozen input')
        write_json(output/'token-ids.json', {k:v.tolist()[0] for k,v in cpu_ids.items()})
        manifest = dict(**provenance, environment=env, checkpoint=checkpoint,
                        config_sha256=digest(output/'config.json'), corpus_sha256=digest(output/'corpus.jsonl'),
                        protocol_sha256=digest(output/'protocol.md'), amendment_sha256=digest(output/'amendment.md'),
                        tokens_sha256=digest(output/'token-ids.json'),
                        source_sha256={p.name:digest(p) for p in (output/'source').glob('*.py')})
        write_json(output/'manifest.json', manifest)
        def load_model():
            return AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,
                 attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
        with Resources(output/'resources.jsonl') as resources:
            print('Loading dense CUDA reference', flush=True)
            target = load_model().to('cuda')
            resources.boundary()
            eos = tuple(target.generation_config.eos_token_id)
            if eos != (151645,151643) or [dimensions(m) for m in extract_ffns(target)] != [(1536,8960)]*28:
                raise ValueError('Unexpected EOS or FFN architecture')
            ids = {k:v.to('cuda') for k,v in cpu_ids.items()}
            indexes = calibrate(target,calibration,ids,output/'calibration',raw,resources)
            gc.collect()
            reference = {}
            for rep in range(-1,3):
                for row in (calibration[:1] if rep==-1 else diagnostics[:4]):
                    torch.cuda.reset_peak_memory_stats()
                    resources.boundary()
                    result = generate(target,ids[row['id']][:,:32],eos,check=resources.check)
                    resources.boundary()
                    raw.record(dict(kind='episode',mode='target',budget_mib=0,repeat=rep,
                                    document=row['id'],warmup=rep==-1,**result,resources=resources.receipt(),
                                    cuda=cuda_memory(torch.device('cuda'))))
                    raw.flush()
                    if rep==0:
                        reference[row['id']] = dict(ids=result['ids'],stop_reason=result['stop_reason'])
                    elif rep>0 and (result['ids']!=reference[row['id']]['ids'] or result['stop_reason']!=reference[row['id']]['stop_reason']):
                        raise RuntimeError('Dense reference repeats disagree')
                    print(f'Dense reference r{rep} {row["id"]}: {result["wall_seconds"]:.3f}s',flush=True)
            write_json(output/'reference.json',reference)
            # Untimed numerical reference; release each full-vocabulary tensor to disk.
            mechanics_dir = output/'mechanics'
            mechanics_dir.mkdir()
            inputs, handles = [], []
            for mlp in extract_ffns(target):
                handles.append(mlp.register_forward_pre_hook(lambda m,a: inputs.append(a[0][:,:2].clone())))
            target(ids[diagnostics[0]['id']][:,:128],use_cache=False)
            for h in handles:
                h.remove()
            save_file({str(i):x.cpu() for i,x in enumerate(inputs)},mechanics_dir/'ffn-inputs.safetensors')
            for i,(mlp,x) in enumerate(zip(extract_ffns(target),inputs,strict=True)):
                save_file({'grouped':grouped_forward(mlp,x,256).cpu()},mechanics_dir/f'ffn-{i:02}.safetensors')
            del inputs
            for i,row in enumerate(diagnostics):
                logits = target(ids[row['id']][:,:128],use_cache=False).logits.cpu()
                save_file({'logits':logits},mechanics_dir/f'reference-{i:02}.safetensors')
                del logits
            print('Loading CPU-backed draft and placing its non-FFN weights on CUDA',flush=True)
            draft = load_model()
            pager = PagedDraft(extract_ffns(draft),indexes,page_width=256,selected_pages=27,
                               capacity_bytes=512*2**20,device='cuda:0',sink=lambda e:None)
            move_non_ffn(draft,'cuda')
            resources.boundary()
            target_params = list(target.parameters())
            draft_params = list(draft.parameters())
            if {p.untyped_storage().data_ptr() for p in target_params} & {p.untyped_storage().data_ptr() for p in draft_params}:
                raise RuntimeError('Unintended target/draft storage alias')
            account = dict(target_parameters_bytes=unique_tensor_bytes(target_params),
                 draft_cuda_parameters_bytes=unique_tensor_bytes([p for p in draft_params if p.is_cuda]),
                 draft_host_parameters_bytes=unique_tensor_bytes([p for p in draft_params if not p.is_cuda]),
                 controller_bytes=pager.controller_bytes, target_draft_shared_bytes=0,
                 controller_host_bytes=sum(index.bytes for index in indexes),
                 host_metadata_bytes=pager.host_metadata_bytes,
                 metadata_method='deduplicated sys.getsizeof Python objects, excluding tensor storage',
                 catalogue_aliases_host=True,
                 cuda=cuda_memory(torch.device('cuda')))
            for mlp,catalog in zip(extract_ffns(draft),pager.catalogs,strict=True):
                for a,b in ((mlp.gate_proj.weight,catalog._gate),(mlp.up_proj.weight,catalog._up),(mlp.down_proj.weight,catalog._down)):
                    if a.is_cuda or a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr():
                        raise RuntimeError('Catalogue did not share evacuated host weights')
            write_json(output/'allocation.json',account)
            page_log = Ledger(mechanics_dir/'pages.jsonl.gz',True)
            pager.cache.sink = page_log.record
            pager.full = True
            passed = True
            xfiles = load_file(mechanics_dir/'ffn-inputs.safetensors')
            for i,mlp in enumerate(extract_ffns(draft)):
                alt = mlp(xfiles[str(i)].to('cuda')).cpu()
                expected = load_file(mechanics_dir/f'ffn-{i:02}.safetensors')['grouped']
                errors = [relative_l2(expected[:,j],alt[:,j]) for j in range(2)]
                error = max(errors)
                save_file({'candidate':alt},mechanics_dir/f'ffn-candidate-{i:02}.safetensors')
                passed &= error <= .01
                raw.record(dict(kind='mechanics_ffn',layer=i,relative_l2=error,
                                per_input_relative_l2=errors,passed=error<=.01))
            for i,row in enumerate(diagnostics):
                resources.boundary()
                alt = draft(ids[row['id']][:,:128],use_cache=False).logits.cpu()
                save_file({'logits':alt},mechanics_dir/f'candidate-{i:02}.safetensors')
                expected = load_file(mechanics_dir/f'reference-{i:02}.safetensors')['logits']
                metrics = compare_logits(expected,alt,cpu_ids[row['id']][:,:128])
                ok = metrics['logit_relative_l2']<=.01 and metrics['mean_kl_dense_to_candidate']<=.001
                passed &= ok
                raw.record(dict(kind='mechanics_document',document=row['id'],passed=ok,**metrics))
                raw.flush()
                del alt,expected
            pager.cache.clear()
            page_log.close()
            pager.full = False
            if not passed:
                raise RuntimeError('Dense-completion mechanics failed; scored draft generation blocked')
            print('All dense-completion numerical checks passed',flush=True)
            episodes_dir = output/'episodes'
            episodes_dir.mkdir()
            warmed = set()
            order = {str(r):shuffled_conditions(cfg['order_seed'],r) for r in range(3)}
            write_json(output/'condition-order.json',order)
            for rep in range(3):
                for mode,budget in order[str(rep)]:
                    for row in ((calibration[:1] if (mode,budget) not in warmed else [])+diagnostics[:4]):
                        warmup = row['split']=='calibration'
                        episode = f'{mode}-{budget}-r{rep}-'+('warmup' if warmup else f'd{diagnostics.index(row)}')
                        folder = episodes_dir/episode
                        folder.mkdir()
                        pages = Ledger(folder/'pages.jsonl.gz',True)
                        rounds = Ledger(folder/'rounds.jsonl')
                        pager.cache.sink = pages.record
                        pager.cache.mode = mode
                        pager.cache.capacity_bytes = budget*2**20
                        pager.reset()
                        torch.cuda.reset_peak_memory_stats()
                        resources.boundary()
                        print(f'Start {episode}',flush=True)
                        try:
                            result = generate(target,ids[row['id']][:,:32],eos,draft=draft,pager=pager,
                                              record=rounds.record,check=resources.check)
                            finishing = time.perf_counter()
                            pager.cache.clear()
                            pages.flush()
                            rounds.flush()
                            finishing_seconds=time.perf_counter()-finishing
                            result['wall_seconds'] += finishing_seconds
                            result['decode_seconds'] += finishing_seconds
                            resources.boundary()
                            stats = dict(pager.cache.stats)
                            same = (not warmup and result['ids']==reference[row['id']]['ids'] and
                                    result['stop_reason']==reference[row['id']]['stop_reason'])
                            item = dict(kind='episode',episode=episode,mode=mode,budget_mib=budget,repeat=rep,
                                document=row['id'],warmup=warmup,reference_match=same,**result,cache=stats,
                                staging_bytes=pager.cache.staging.numel()*4,
                                cuda=cuda_memory(torch.device('cuda')),resources=resources.receipt())
                            raw.record(item)
                            raw.flush()
                            write_json(folder/'episode.json',item)
                            print(f'End {episode}: {result["wall_seconds"]:.3f}s, accepted {result["accepted"]}/{result["attempted"]}, match={same}',flush=True)
                        finally:
                            pages.close()
                            rounds.close()
                        warmed.add((mode,budget))
        # Exiting the sampler joins it and checks one final boundary before success.
        raw.record(dict(kind='complete',resources=resources.receipt()))
        raw.flush()
        write_json(output/'completion.json',dict(status='complete',resources=resources.receipt()))
    except BaseException as exc:
        raw.record(dict(kind='failure',type=type(exc).__name__,message=str(exc)))
        raw.flush()
        write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/fault-pager.json')
    args = parser.parse_args()
    run(args.output.resolve(),args.config.resolve())


if __name__=='__main__':
    main()

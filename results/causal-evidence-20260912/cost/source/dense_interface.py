"""Prospective native/chat/incremental dense interface qualification."""

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .adapters import extract_ffns
from .balanced_control import PREFIX
from .causal_study import packed
from .experiment import digest, environment, write_json
from .metrics import relative_l2
from .provenance import committed_inputs, frozen_environment, verify_snapshot

SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
DOMAINS = ('code', 'extraction', 'arithmetic', 'copying', 'topic_changes')
SYNTAX = {'integer': r'-?\d+', 'choice': r'[ABC]', 'identifier': r'[A-Za-z0-9_|-]+'}


def score(text, task):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else ''
    correct = (text.strip() if task['domain'] == 'copying' else first) == task['answer']
    compliant = len(lines) == 1 and re.fullmatch(SYNTAX[task['syntax']], first) is not None
    return dict(first_line=first, answer_correct=correct, format_compliant=compliant,
                success=correct and compliant)


def prepare(tokenizer, task, interface):
    if interface == 'plain':
        return tokenizer(PREFIX + task['task'] + '\nAnswer:', add_special_tokens=False,
                         return_tensors='pt')['input_ids']
    if interface != 'chat':
        raise ValueError('Unknown interface')
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': PREFIX + task['task']}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors='pt', return_dict=True)
    return encoded['input_ids']


def decoding(checkpoint):
    source = json.loads((Path(checkpoint) / 'generation_config.json').read_text(encoding='utf-8'))
    eos = source['eos_token_id']
    # In Transformers 5.13, None means inherit from the checkpoint. Fill global
    # defaults explicitly, especially repetition_penalty=1.0 (Qwen ships 1.1).
    values = GenerationConfig._get_default_generation_params()
    values.update(max_new_tokens=64, eos_token_id=eos, pad_token_id=source['pad_token_id'],
                  bos_token_id=source.get('bos_token_id'), output_logits=True,
                  return_dict_in_generate=True)
    return GenerationConfig(**values)


@torch.inference_mode()
def incremental_generate(model, prompt, config, forced=None):
    """Replay native tokens when forced, otherwise decode on this path's own prefix."""
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError('Nonempty batch-one prompt required')
    eos = config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    count = config.max_new_tokens if forced is None else len(forced)
    if count < 1 or count > config.max_new_tokens:
        raise ValueError('Invalid continuation length')
    past = None
    for index in range(prompt.shape[1]):
        result = model(input_ids=prompt[:, index:index+1], past_key_values=past, use_cache=True)
        past = result.past_key_values
    ids, values = [], []
    for index in range(count):
        logits = result.logits[:, -1].float()
        values.append(logits.detach().cpu())
        token = int(logits.argmax(-1).item()) if forced is None else int(forced[index])
        ids.append(token)
        if token in eos or index + 1 == count:
            break
        result = model(input_ids=torch.tensor([[token]], device=prompt.device),
                       past_key_values=past, use_cache=True)
        past = result.past_key_values
    if forced is not None and ids != list(forced):
        raise ValueError('Forced continuation includes post-EOS tokens')
    return ids, torch.cat(values, 0)


def aligned_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] < 1:
        raise ValueError('Aligned decision shape mismatch')
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise ValueError('FP32 decision logits required')
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return None
    logp, logq = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
    error = relative_l2(reference, candidate)
    kl = float((logp.exp() * (logp-logq)).sum(-1).mean().clamp_min(0))
    if not math.isfinite(error) or not math.isfinite(kl):
        return None
    return {'relative_l2': error, 'mean_kl': kl,
            'argmax_agreement': float((reference.argmax(-1) == candidate.argmax(-1)).float().mean())}


@contextmanager
def full_retention(mlps):
    """Exercise a group-mask hook while retaining and counting every neuron."""
    counts = [0] * len(mlps)
    handles = []
    try:
        for index, mlp in enumerate(mlps):
            neurons = mlp.down_proj.weight.shape[1]
            if neurons % 8:
                raise ValueError('Width-eight groups required')
            mask = torch.ones(neurons // 8, device=mlp.down_proj.weight.device)
            def observe(module, args, i=index, group_mask=mask):
                if args[0].numel() != group_mask.numel()*8:
                    raise ValueError('Incremental FFN visit required')
                counts[i] += group_mask.numel()
                return (args[0] * group_mask.repeat_interleave(8), *args[1:])
            handles.append(mlp.down_proj.register_forward_pre_hook(observe))
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def tasks_from(cfg, root):
    tasks = []
    for split, expected, per_domain in [('debug', 10, 2), ('fresh', 20, 4)]:
        spec = cfg['corpora'][split]
        path = root / spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError('Frozen corpus mismatch')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows) != expected or Counter(t['domain'] for t in rows) != {d: per_domain for d in DOMAINS}:
            raise ValueError('Task grid mismatch')
        for task in rows:
            if task['syntax'] not in SYNTAX or not task['task'] or not re.fullmatch(SYNTAX[task['syntax']], task['answer']):
                raise ValueError('Invalid task or answer syntax')
            tasks.append({**task, 'split': split})
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task ID')
    return tasks


def summarize(rows):
    result = {}
    for split in ('debug', 'fresh'):
        for interface in ('plain', 'chat') if split == 'debug' else ('chat',):
            for path in ('native', 'incremental', 'packed'):
                selected = [r for r in rows if (r['split'], r['interface'], r['path']) == (split, interface, path)]
                result[f'{split}-{interface}-{path}'] = {
                    'tasks': len(selected),
                    **{key: sum(r[key] for r in selected) for key in ('answer_correct', 'format_compliant', 'success')},
                    'domain_success': {d: sum(r['success'] for r in selected if r['domain'] == d) for d in DOMAINS}}
    debug, fresh = result['debug-chat-native'], result['fresh-chat-native']
    utility = debug['success'] >= 7 and fresh['success'] >= 14 and min(fresh['domain_success'].values()) >= 2
    fidelity = all(r['finite'] and (r['path'] == 'native' or (
        r['generated_equal'] and r['aligned'] is not None and r['aligned']['relative_l2'] <= .01
        and r['aligned']['mean_kl'] <= .001)) for r in rows)
    return {'conditions': result, 'utility_qualified': utility, 'reference_fidelity': fidelity,
            'gate_a': utility and fidelity, 'runtime_nomination': None}


@torch.inference_mode()
def run(config_path, output_path):
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    source, protocol = committed_inputs(__file__, config_path, cfg['protocol'])
    root = config_path.parent.parent
    tasks = tasks_from(cfg, root)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    torch.set_num_threads(4); torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    env = environment(torch.device('cuda')); frozen_environment(env)
    output.mkdir(parents=True, exist_ok=False); (output/'tensors').mkdir(); (output/'data').mkdir()
    shutil.copyfile(config_path, output/'config.json'); shutil.copyfile(protocol, output/'protocol.md')
    shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    for spec in cfg['corpora'].values():
        shutil.copyfile(root/spec['path'], output/spec['path'])
    write_json(output/'manifest.json', {**source, 'environment': env, 'started_utc': datetime.now(timezone.utc).isoformat()})
    stream = (output/'results.jsonl').open('x', encoding='utf-8')
    rows = []
    def record(row):
        rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        spec = cfg['model']; parent = root/spec['parent']
        for name, sha in spec['parent_hashes'].items():
            if digest(parent/name) != sha: raise ValueError('Parent identity mismatch')
        checkpoint = Path(snapshot_download(spec['model'], revision=spec['revision'], local_files_only=True))
        catalog = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
        if catalog != json.loads((parent/'manifest.json').read_text(encoding='utf-8'))['checkpoint_files']:
            raise ValueError('Checkpoint catalog mismatch')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        config = decoding(checkpoint)
        write_json(output/'model.json', {'checkpoint_files': catalog, 'generation_config': config.to_dict(),
            'chat_template': tokenizer.chat_template, 'model': spec['model'], 'revision': spec['revision']})
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation='sdpa',
            local_files_only=True, trust_remote_code=False).cuda().eval()
        mlps = extract_ffns(model)
        orders = json.loads(gzip.decompress((parent/'layouts.json.gz').read_bytes()))['popularity']
        inputs = [(task, interface, prepare(tokenizer, task, interface)) for task in tasks
                  for interface in (('plain', 'chat') if task['split'] == 'debug' else ('chat',))]
        if any(p.shape[1] < 1 or p.shape[1] + 64 > 512 for _, _, p in inputs):
            raise ValueError('Token bound exceeded')
        native = {}
        for path in ('native', 'incremental', 'packed'):
            # Native remains wholly uninstrumented except a passive finite-logit observer.
            # This observer does not modify arguments, weights, outputs or decoding.
            layout = packed(mlps, orders) if path == 'packed' else nullcontext()
            with layout:
                for task, interface, prompt in inputs:
                    label = f"{task['id']}-{interface}-{path}"
                    failures, finite_calls = [], []
                    def finite_observer(module, args, result):
                        good = bool(torch.isfinite(result.logits).all()); finite_calls.append(good)
                        if not good:
                            file = output/'tensors'/f'{label}-nonfinite-{len(finite_calls)}.safetensors'
                            save_file({'logits': result.logits.detach().cpu()}, file)
                            failures.append({'file': file.relative_to(output).as_posix(), 'sha256': digest(file)})
                    handle = model.register_forward_hook(finite_observer)
                    retain = full_retention(mlps) if path == 'packed' else nullcontext()
                    try:
                        with retain as counts:
                            if path == 'native':
                                value = model.generate(prompt.cuda(), attention_mask=torch.ones_like(prompt).cuda(),
                                                       generation_config=config)
                                ids = value.sequences[0, prompt.shape[1]:].tolist()
                                logits = torch.cat([v.float().cpu() for v in value.logits], 0)
                                native[task['id'], interface] = (ids, logits)
                                aligned = logits
                            else:
                                ids, logits = incremental_generate(model, prompt.cuda(), config)
                                reference_ids, _ = native[task['id'], interface]
                                _, aligned = incremental_generate(model, prompt.cuda(), config, forced=reference_ids)
                    finally:
                        handle.remove()
                    reference_ids, reference = native[task['id'], interface]
                    file = output/'tensors'/f'{label}.safetensors'
                    save_file({'own_logits': logits.contiguous(), 'aligned_logits': aligned.clone().contiguous()}, file)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    row = {k: task[k] for k in ('id', 'domain', 'split')}
                    row.update(interface=interface, path=path, prompt_ids=prompt[0].tolist(), token_ids=ids,
                        text=text, finite=all(finite_calls), finite_calls=finite_calls, nonfinite=failures,
                        selected_groups_by_layer=counts, generated_equal=ids == reference_ids,
                        aligned=aligned_metrics(reference, aligned), tensors={'file': file.relative_to(output).as_posix(), 'sha256': digest(file)},
                        **score(text, task))
                    record(row)
                print('Completed dense interface path', path, flush=True)
        summary = summarize(rows); write_json(output/'summary.json', summary); return summary
    except (Exception, KeyboardInterrupt) as error:
        write_json(output/'failure.json', {'type': type(error).__name__, 'message': str(error)}); raise
    finally:
        stream.close()


def analyze(run_path, output_path):
    """Recompute scoring and aligned-prefix fidelity from all saved decision logits."""
    run = Path(run_path)
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(run, manifest, 'configs/dense-interface.json', cfg['protocol'], __file__)
    frozen_environment(manifest['environment'])
    tasks = {t['id']: t for t in tasks_from(cfg, run)}
    checkpoint = Path(snapshot_download(cfg['model']['model'], revision=cfg['model']['revision'], local_files_only=True))
    model_receipt = json.loads((run/'model.json').read_text(encoding='utf-8'))
    catalog = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
    from .provenance import owning_repository
    repository = owning_repository(__file__)
    parent = repository/cfg['model']['parent']
    for name, sha in cfg['model']['parent_hashes'].items():
        if digest(parent/name) != sha: raise ValueError('Parent identity mismatch')
    if catalog != model_receipt['checkpoint_files'] or catalog != json.loads((parent/'manifest.json').read_text(encoding='utf-8'))['checkpoint_files']:
        raise ValueError('Checkpoint catalog mismatch')
    config = decoding(checkpoint)
    if config.to_dict() != model_receipt['generation_config']: raise ValueError('Decoding configuration mismatch')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if tokenizer.chat_template != model_receipt['chat_template']: raise ValueError('Chat template mismatch')
    rows = [json.loads(line) for line in (run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(t['id'], i, p) for t in tasks.values() for i in (('plain','chat') if t['split']=='debug' else ('chat',))
                for p in ('native','incremental','packed')}
    if len(rows) != len(expected) or {(r['id'],r['interface'],r['path']) for r in rows} != expected:
        raise ValueError('Incomplete comparison grid')
    native = {}
    eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    for row in rows:
        task = tasks[row['id']]; key = row['id'], row['interface']
        if (row['split'], row['domain']) != (task['split'], task['domain']): raise ValueError('Task identity mismatch')
        if row['prompt_ids'] != prepare(tokenizer, task, row['interface'])[0].tolist(): raise ValueError('Prompt mismatch')
        path = run/row['tensors']['file']
        if digest(path) != row['tensors']['sha256']: raise ValueError('Tensor checksum mismatch')
        values = load_file(path); own = values['own_logits']; ids = row['token_ids']
        if own.ndim != 2 or own.shape[0] != len(ids) or own.shape[1] != json.loads((checkpoint/'config.json').read_text())['vocab_size']:
            raise ValueError('Generation tensor dimensions mismatch')
        if len(ids) < 1 or len(ids) > 64 or any(i in eos for i in ids[:-1]) or (len(ids) < 64 and ids[-1] not in eos):
            raise ValueError('EOS or length mismatch')
        if ids != own.argmax(-1).tolist(): raise ValueError('Generation was not greedy')
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if row['text'] != text or any(row[k] != v for k,v in score(text,task).items()): raise ValueError('Scoring mismatch')
        if row['path'] == 'native': native[key] = ids, own
        reference_ids, reference = native[key]
        if row['generated_equal'] != (ids == reference_ids) or row['aligned'] != aligned_metrics(reference, values['aligned_logits']):
            raise ValueError('Aligned comparison mismatch')
        calls = len(ids) if row['path']=='native' else 2*len(row['prompt_ids'])+len(ids)+len(reference_ids)-2
        if len(row['finite_calls']) != calls or row['finite'] != all(row['finite_calls']): raise ValueError('Finite observation grid mismatch')
        if row['finite'] and (row['nonfinite'] or not torch.isfinite(own).all() or not torch.isfinite(values['aligned_logits']).all()):
            raise ValueError('False finite receipt')
        if len(row['nonfinite']) != sum(not v for v in row['finite_calls']): raise ValueError('Incomplete nonfinite receipt')
        for receipt in row['nonfinite']:
            bad = run/receipt['file']
            if digest(bad) != receipt['sha256'] or torch.isfinite(load_file(bad)['logits']).all(): raise ValueError('Invalid nonfinite tensor')
        expected_counts = [calls*1120]*28 if row['path']=='packed' else None
        if row['selected_groups_by_layer'] != expected_counts: raise ValueError('Full retention accounting mismatch')
    summary = summarize(rows)
    if summary != json.loads((run/'summary.json').read_text(encoding='utf-8')): raise ValueError('Summary mismatch')
    output = Path(output_path); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'summary.json', summary)
    write_json(output/'verification.json', {'rows': len(rows), 'status': 'verified', 'source_commit': manifest['source_commit']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--run'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if bool(args.config) == bool(args.run): parser.error('Specify exactly one of --config or --run')
    print(json.dumps(run(args.config, args.output) if args.config else analyze(args.run, args.output), indent=2))


if __name__ == '__main__':
    main()

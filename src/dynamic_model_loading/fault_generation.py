"""Greedy target reference and explicit independent target/draft KV rollback."""
import time

import torch
from transformers.cache_utils import DynamicCache

from .fault_pager_study import verification_commit


def sync(model):
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.synchronize()


def kv_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in cache.layers
               for t in (layer.keys, layer.values) if t is not None)


@torch.inference_mode()
def step(model, tokens, cache, pager=None):
    if pager is not None:
        if tokens.shape[1] != 1:
            raise ValueError('Draft must execute one token at a time')
        pager.begin_token()
    result = model(tokens, past_key_values=cache, use_cache=True).logits
    if pager is not None and hasattr(pager, 'after_cache_step'):
        pager.after_cache_step(cache)
    return result


@torch.inference_mode()
def generate(target, prefix, eos, cap=64, *, draft=None, pager=None, record=lambda r: None,
             check=lambda: None, copy_receipt=False):
    """Both caches end at the last consumed token; saved logits predict the next.

    Batched target logits are a candidate numerical path, not a substituted reference.
    The caller compares the complete output against the retained scalar reference.
    """
    sync(target)
    started = time.perf_counter()
    copies={'h2d_bytes':0,'d2h_bytes':0,'operations':{}}
    def charge(direction,size,operation):
        if copy_receipt and prefix.is_cuda:
            copies[direction+'_bytes']+=size
            copies['operations'][operation]=copies['operations'].get(operation,0)+size
    target_cache = DynamicCache(config=target.config)
    draft_cache = DynamicCache(config=draft.config) if draft is not None else None
    t0 = time.perf_counter()
    target_next = step(target, prefix, target_cache)[:, -1, :]
    sync(target)
    target_seconds = time.perf_counter() - t0
    if draft is not None:
        for token in prefix.split(1, dim=1):
            draft_next = step(draft, token, draft_cache, pager)[:, -1, :]
    sync(target)
    prefill_seconds = time.perf_counter() - started
    output, attempted, accepted, rounds = [], 0, 0, 0
    peak_tkv, peak_dkv = kv_bytes(target_cache), kv_bytes(draft_cache) if draft_cache else 0
    while len(output) < cap:
        check()
        if draft is None:
            token = int(target_next.argmax(-1).item())
            charge('d2h',8,'target_argmax')
            output.append(token)
            if token in eos or len(output) == cap:
                break
            t0 = time.perf_counter()
            charge('h2d',8,'target_next_token')
            target_next = step(target, prefix.new_tensor([[token]]), target_cache)[:, -1, :]
            sync(target)
            target_seconds += time.perf_counter() - t0
            peak_tkv = max(peak_tkv, kv_bytes(target_cache))
            continue
        base = target_cache.get_seq_length()
        if draft_cache.get_seq_length() != base:
            raise RuntimeError('Target/draft cache boundary mismatch')
        proposed = []
        for i in range(4):
            token = int(draft_next.argmax(-1).item())
            charge('d2h',8,'draft_argmax')
            proposed.append(token)
            # Consume even the fourth: its KV can be cropped on rejection.
            charge('h2d',8,'draft_next_token')
            draft_next = step(draft, prefix.new_tensor([[token]]), draft_cache, pager)[:, -1, :]
        sync(target)
        t0 = time.perf_counter()
        charge('h2d',32,'target_proposal_batch')
        logits = step(target, prefix.new_tensor([proposed]), target_cache)
        predictions = torch.cat((target_next.argmax(-1), logits[0, :3].argmax(-1)))
        sync(target)
        target_seconds += time.perf_counter() - t0
        peak_tkv = max(peak_tkv, kv_bytes(target_cache))
        peak_dkv = max(peak_dkv, kv_bytes(draft_cache))
        charge('h2d',32,'verifier_proposals')
        verdict = verification_commit(prefix.new_tensor(proposed), predictions, tuple(eos))
        charge('d2h',64,'verifier_inputs')
        charge('h2d',8*verdict['committed'].numel(),'verifier_committed_tensor')
        charge('d2h',8*verdict['committed'].numel(),'verifier_committed_readback')
        committed = verdict['committed'].tolist()[:cap-len(output)]
        emitted_accepted = min(verdict['accepted'], len(committed))
        attempted += 4
        accepted += emitted_accepted
        rounds += 1
        done = committed[-1] in eos or len(output) + len(committed) == cap
        a = verdict['accepted']
        if verdict['fallback'] is not None:
            target_cache.crop(base + a)
            draft_cache.crop(base + a)
            pager.pending.clear()
            if hasattr(pager, 'after_cache_crop'):
                pager.after_cache_crop(draft_cache)
            if not done:
                charge('h2d',8,'fallback_token')
                fallback = prefix.new_tensor([[verdict['fallback']]])
                t0 = time.perf_counter()
                target_next = step(target, fallback, target_cache)[:, -1, :]
                sync(target)
                target_seconds += time.perf_counter() - t0
                draft_next = step(draft, fallback, draft_cache, pager)[:, -1, :]
        else:
            target_next = logits[:, -1, :]
        charge('d2h',32,'ledger_target_predictions')
        record(dict(kind='verification', round=rounds, base=base, proposed=proposed,
                    target_predictions=predictions.tolist(), accepted=a, emitted_accepted=emitted_accepted,
                    fallback=verdict['fallback'], committed=committed,
                    target_cache=target_cache.get_seq_length(), draft_cache=draft_cache.get_seq_length(),
                    stop_reason='eos' if committed[-1] in eos else ('length' if done else verdict['stop_reason'])))
        output.extend(committed)
        del logits, predictions, verdict
        if done:
            break
    sync(target)
    wall = time.perf_counter() - started
    return dict(ids=output, stop_reason='eos' if output[-1] in eos else 'length',
                started_monotonic=started,finished_monotonic=time.perf_counter(),
                wall_seconds=wall, prefill_seconds=prefill_seconds, decode_seconds=wall-prefill_seconds,
                target_seconds=target_seconds, attempted=attempted, accepted=accepted, rounds=rounds,
                target_kv_peak_bytes=peak_tkv, draft_kv_peak_bytes=peak_dkv,
                **({'generator_copies':copies} if copy_receipt else {}))

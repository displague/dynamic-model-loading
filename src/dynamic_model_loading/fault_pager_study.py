"""Protocol-bound verifier bookkeeping for the physical fault-pager study."""

from __future__ import annotations

import random
import json

import torch


FROZEN_CONFIG = {
    "protocol": "docs/fault-pager-protocol.md", "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "dtype": "float32", "device": "cuda",
    "corpus": "runs/qwen15b-packing-pilot-20260911/corpus.jsonl",
    "corpus_sha256": "5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31",
    "page_width": 256, "selected_pages": 27, "medoids": 16, "cache_budgets_mib": [128, 512],
    "proposal_tokens": 4, "generation_tokens": 64, "calibration_tokens": 128,
    "mechanics_tokens": 128, "diagnostic_documents": 4, "repetitions": 3, "order_seed": 20260914,
    "gpu_used_max_mib": 15000, "host_available_min_mib": 2048,
    "reconstruction_relative_l2_max": 0.01, "logit_relative_l2_max": 0.01,
    "logit_mean_kl_max": 0.001,
}


def validate_config(cfg: dict) -> None:
    if json.dumps(cfg, sort_keys=True, allow_nan=False) != json.dumps(FROZEN_CONFIG, sort_keys=True):
        raise ValueError("Fault-pager configuration differs from the frozen protocol inputs")


def verification_commit(proposed: torch.Tensor, target_predictions: torch.Tensor, eos_token_id: int | tuple) -> dict:
    """Commit only a target-verified draft prefix plus a target fallback on mismatch.

    ``target_predictions[i]`` is the target greedy token conditional on the common
    prefix and the preceding proposed tokens.  This is sufficient because all such
    preceding tokens are equal whenever position ``i`` is examined.
    """
    if (proposed.ndim != target_predictions.ndim or proposed.ndim != 1
            or proposed.numel() != FROZEN_CONFIG["proposal_tokens"]
            or not (type(eos_token_id) is int or isinstance(eos_token_id, tuple))):
        raise ValueError("Expected a frozen-length matching one-dimensional proposal and target tensors")
    if proposed.shape != target_predictions.shape or proposed.dtype != torch.long or target_predictions.dtype != torch.long:
        raise ValueError("Proposal and target predictions must be matching torch.long token IDs")
    eos = (eos_token_id,) if type(eos_token_id) is int else eos_token_id
    if not eos or any(type(v) is not int or v < 0 for v in eos):
        raise ValueError("Invalid EOS IDs")
    accepted = 0
    committed = []
    fallback = None
    stop_reason = "proposal_exhausted"
    for proposal, target in zip(proposed.tolist(), target_predictions.tolist(), strict=True):
        if proposal != target:
            fallback = target
            committed.append(target)
            stop_reason = "eos" if target in eos else "rejected"
            break
        accepted += 1
        committed.append(proposal)
        if target in eos:
            stop_reason = "eos"
            break
    committed_tensor = torch.tensor(committed, dtype=torch.long, device=proposed.device)
    return {"proposed": proposed.clone(), "target_predictions": target_predictions.clone(), "accepted": accepted,
            "attempted": proposed.numel(), "accepted_prefix": proposed[:accepted].clone(), "fallback": fallback,
            "committed": committed_tensor, "stop_reason": stop_reason}


def shuffled_conditions(seed: int, repetition: int) -> list[tuple[str, int]]:
    """The exact six-condition ordering prescribed by the protocol."""
    if type(seed) is not int or type(repetition) is not int or repetition not in (0, 1, 2):
        raise ValueError("Invalid frozen condition-order seed or repetition")
    conditions = [(mode, budget) for mode in ("eager", "lru", "prefetch") for budget in (128, 512)]
    random.Random(seed + repetition).shuffle(conditions)
    return conditions


def mib_to_bytes(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("MiB value must be a positive integer")
    return value * 1024 * 1024

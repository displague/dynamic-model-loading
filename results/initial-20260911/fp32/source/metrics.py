"""Token-weighted diagnostics with an explicit next-token denominator."""

import math

import torch
from torch.nn import functional as F


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("Shape mismatch")
    a, b = reference.double(), candidate.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite values")
    denominator = a.norm().item()
    error = (a - b).norm().item()
    return error / denominator if denominator else (0.0 if error == 0 else math.inf)


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, ids: torch.Tensor) -> dict:
    """Compare positions 0..T-2, which predict the observed tokens 1..T-1.

    Corpus text NLL includes all tokens, not assistant-only or task-answer scoring.
    Vocabulary computations are chunked to bound temporary FP32 storage.
    """
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[0] != 1:
        raise ValueError("Expected matching batch-one [1, tokens, vocabulary] logits")
    if ids.shape != reference.shape[:2] or ids.shape[1] < 2:
        raise ValueError("Expected at least two matching token IDs")
    a, b = reference[0, :-1], candidate[0, :-1]
    labels = ids[0, 1:].to(a.device)
    kl_sum = dense_nll = candidate_nll = error_sq = norm_sq = 0.0
    agreement = 0
    max_abs = 0.0
    for start in range(0, len(a), 16):
        ref = a[start:start + 16].float()
        alt = b[start:start + 16].to(ref.device).float()
        if not torch.isfinite(ref).all() or not torch.isfinite(alt).all():
            raise ValueError("Non-finite logits")
        target = labels[start:start + 16, None]
        ref_logp, alt_logp = F.log_softmax(ref, -1), F.log_softmax(alt, -1)
        # Tiny negative values from floating-point summation are clipped per token.
        kl_sum += (ref_logp.exp() * (ref_logp - alt_logp)).sum(-1).clamp_min(0).double().sum().item()
        dense_nll -= ref_logp.gather(-1, target).double().sum().item()
        candidate_nll -= alt_logp.gather(-1, target).double().sum().item()
        agreement += (ref.argmax(-1) == alt.argmax(-1)).sum().item()
        diff = ref - alt
        error_sq += diff.double().square().sum().item()
        norm_sq += ref.double().square().sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
    n = len(a)
    return {
        "predicted_tokens": n,
        "mean_kl_dense_to_candidate": kl_sum / n,
        "top1_agreement": agreement / n,
        "dense_nll": dense_nll / n,
        "candidate_nll": candidate_nll / n,
        "relative_perplexity": math.exp((candidate_nll - dense_nll) / n),
        "logit_relative_l2": math.sqrt(error_sq / norm_sq) if norm_sq else (0 if not error_sq else math.inf),
        "logit_max_abs_error": max_abs,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate empty measurements")
    total = sum(r["predicted_tokens"] for r in rows)
    if total <= 0:
        raise ValueError("No predicted tokens")
    keys = ("mean_kl_dense_to_candidate", "top1_agreement", "dense_nll", "candidate_nll")
    result = {key: sum(r[key] * r["predicted_tokens"] for r in rows) / total for key in keys}
    result["predicted_tokens"] = total
    result["relative_perplexity"] = math.exp(result["candidate_nll"] - result["dense_nll"])
    return result

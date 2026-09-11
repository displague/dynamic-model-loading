"""Fixed-mask, privileged additive repair; never a causal acquisition policy."""

from contextlib import contextmanager
import math

import numpy as np
import torch
from torch.nn import functional as F

from .causal import group_scores
from .ffn import dimensions


SCHEDULES = ('hindsight', 'resident_first')
LIMITS = (0, 8, 28, 56, 84, 112)
ONE_SHOT = (.9, .925, .95, .975, 1.)
MODES = ('static', 'recency', 'ema', 'learned')


def additions(initial, hot, scores, schedule, limit):
    """Rank eligible omitted groups only, with deterministic group-index ties."""
    if schedule not in SCHEDULES or type(limit) is not int or limit < 0:
        raise ValueError('Invalid repair schedule or limit')
    if initial.dtype != torch.bool or hot.dtype != torch.bool:
        raise ValueError('Boolean masks required')
    if initial.ndim != 1 or initial.shape != hot.shape or initial.shape != scores.shape:
        raise ValueError('Repair vector shape mismatch')
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Finite nonnegative privileged scores required')
    repair = (~initial & hot) if schedule == 'resident_first' else torch.zeros_like(initial)
    candidates = torch.nonzero(~(initial | repair), as_tuple=False).flatten()
    ranked = candidates[scores[candidates].argsort(descending=True, stable=True)]
    repair[ranked[:limit]] = True
    return repair


def expected_additions(initial, hot, omitted_scores, schedule, limit):
    """Independent NumPy receipt reconstruction, in ascending omitted-index order."""
    omitted = np.flatnonzero(~initial)
    if omitted_scores.shape != (len(omitted),) or not np.isfinite(omitted_scores).all() or (omitted_scores < 0).any():
        raise ValueError('Invalid omitted-score receipt')
    score = dict(zip(omitted.tolist(), omitted_scores.tolist(), strict=True))
    result = ~initial & hot if schedule == 'resident_first' else np.zeros_like(initial)
    candidates = [int(i) for i in omitted if not result[i]]
    ranked = sorted(candidates, key=lambda i: (-score[i], i))
    result[ranked[:limit]] = True
    return result


class RepairProbe:
    """Add disjoint contributions before the FFN output reaches its residual consumer.

    The ordinary dense FFN is the privileged audit. Its normalized input and z are
    retained through this visit; the initial and corrective down products are each
    evaluated once, separately. Frozen initial masks do not adapt to repaired inputs.
    """
    def __init__(self, mlp, initial_masks, hot, schedule, limit, width=8):
        self.mlp = mlp
        self.hidden, self.neurons = dimensions(mlp)
        self.width, self.schedule, self.limit = width, schedule, limit
        groups = math.ceil(self.neurons / width)
        if initial_masks.ndim != 2 or initial_masks.shape[1] != groups or initial_masks.dtype != torch.bool:
            raise ValueError('Invalid frozen masks')
        if hot.shape != (groups,) or hot.dtype != torch.bool:
            raise ValueError('Invalid resident map')
        if not torch.all(initial_masks.sum(-1) == initial_masks[0].sum()):
            raise ValueError('Fixed initial cardinality required')
        self.initial_masks, self.hot = initial_masks, hot
        self.norms = mlp.down_proj.weight.detach().float().norm(dim=0)
        self.position = 0
        self.x = self.z = None
        self.repairs, self.scores, self.audits = [], [], []

    def before(self, module, args):
        if self.x is not None or self.z is not None or self.position >= len(self.initial_masks):
            raise ValueError('Unexpected or unconsumed repair visit')
        if args[0].numel() != self.hidden:
            raise ValueError('Repair requires batch-one incremental input')
        self.x = args[0]

    def capture(self, module, args):
        if self.x is None or self.z is not None:
            raise ValueError('Repair activation order mismatch')
        self.z = args[0]

    def after(self, module, args, full):
        if self.x is None or self.z is None:
            raise ValueError('Incomplete repair input')
        initial = self.initial_masks[self.position]
        scores = group_scores(self.z, self.norms, self.width).reshape(-1)
        repair = additions(initial, self.hot, scores, self.schedule, self.limit)
        first = F.linear(self.z * initial.repeat_interleave(self.width)[:self.neurons], self.mlp.down_proj.weight)
        extra = F.linear(self.z * repair.repeat_interleave(self.width)[:self.neurons], self.mlp.down_proj.weight)
        corrected = first + extra
        denominator = full.float().norm()
        def error(value):
            numerator = (value.float() - full.float()).norm()
            return torch.where(denominator > 0, numerator / denominator,
                               torch.where(numerator == 0, 0., float('inf')))
        audit = torch.stack((error(first), error(corrected), denominator,
                             first.float().norm(), extra.float().norm()))
        if not torch.isfinite(audit).all():
            raise ValueError('Non-finite repair audit')
        self.repairs.append(repair.detach().cpu())
        self.scores.append(scores[~initial].detach().cpu())
        self.audits.append(audit.detach().cpu())
        self.position += 1
        self.x = self.z = None
        return corrected

    def take(self):
        if self.position != len(self.initial_masks) or self.x is not None or self.z is not None:
            raise ValueError('Incomplete frozen-mask episode')
        return tuple(torch.stack(values).numpy() for values in (self.repairs, self.scores, self.audits))


@contextmanager
def repair_probes(mlps, observers):
    handles = []
    try:
        for mlp, observer in zip(mlps, observers, strict=True):
            source = getattr(mlp, 'source', mlp)
            handles.append(source.register_forward_pre_hook(observer.before))
            handles.append(mlp.down_proj.register_forward_pre_hook(observer.capture))
            handles.append(source.register_forward_hook(observer.after))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for observer in observers:
            observer.x = observer.z = None


def save_receipt(path, initial, repairs, scores, audits):
    final = initial | repairs
    np.savez_compressed(path, initial_bits=np.packbits(initial, axis=-1, bitorder='little'),
        repair_bits=np.packbits(repairs, axis=-1, bitorder='little'),
        final_bits=np.packbits(final, axis=-1, bitorder='little'), shape=np.asarray(initial.shape),
        omitted_scores=scores, audits=audits)


def trace_metrics(initial, repair, hot, sizes):
    cold = repair & ~hot
    return {'added_groups': int(repair.sum()), 'resident_added_groups': int((repair & hot).sum()),
            'cold_added_groups': int(cold.sum()), 'added_cold_bytes': int((cold.sum(0) * sizes).sum()),
            'initial_selected_bytes': int((initial.sum(0) * sizes).sum()),
            'added_selected_bytes': int((repair.sum(0) * sizes).sum()),
            'final_selected_bytes': int(((initial | repair).sum(0) * sizes).sum())}

"""Slot-bounded analytical cache replay; no device transfers or timing claims."""

from collections import OrderedDict
import math

import numpy as np


POLICIES = ("no_cache", "lru", "static_global", "static_equal_layer")


def cache_shape(group_sizes, budget):
    sizes = np.asarray(group_sizes)
    if sizes.ndim != 2 or sizes.dtype.kind not in "iu" or (sizes <= 0).any():
        raise ValueError("Positive integer group sizes required")
    slot = int(sizes.max())
    if type(budget) is not int or budget < slot:
        raise ValueError("Budget must fit the incoming group slot")
    slots = min(budget // slot - 1, sizes.size)
    return {"budget_bytes": budget, "slot_bytes": slot, "retained_slots": slots,
            "incoming_reserve_bytes": slot,
            "unallocated_bytes": budget - (slots + 1) * slot}


def hot_mask(importance, sizes, slots, policy):
    if importance.shape != sizes.shape or not np.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("Invalid calibration group importance")
    value = importance / sizes
    mask = np.zeros(sizes.shape, dtype=bool)
    if policy == "static_global":
        indices = np.argsort(-value.reshape(-1), kind="stable")[:slots]
        mask.reshape(-1)[indices] = True
    elif policy == "static_equal_layer":
        quotient, remainder = divmod(slots, sizes.shape[0])
        for layer in range(sizes.shape[0]):
            indices = np.argsort(-value[layer], kind="stable")[:quotient + (layer < remainder)]
            mask[layer, indices] = True
    else:
        raise ValueError("Unknown static policy")
    return mask


def zero_lru_bound(trace):
    """Distinct other-layer requests between consecutive visits, including wraparound.

    Each layer has a disjoint page namespace. The minimum count observed at every
    other layer is a conservative lower bound; repeated tokens cannot reduce it.
    """
    minima = trace.sum(axis=2).min(axis=0)
    return int(minima.sum() - minima.max())


def explicit_lru(trace, sizes, slots):
    """Reference replay: touch all hits first, then ascending misses per layer."""
    cache = OrderedDict()
    result = []
    groups = sizes.shape[1]
    for state in ("cold", "warm"):
        hits = misses = volume = 0
        for token in trace:
            for layer, requested in enumerate(token):
                missing = []
                for group in np.flatnonzero(requested):
                    page = layer * groups + int(group)
                    if page in cache:
                        hits += 1
                        cache.move_to_end(page)
                    else:
                        missing.append((page, int(sizes[layer, group])))
                for page, size in missing:
                    misses += 1
                    volume += size
                    if slots:
                        cache[page] = None
                        if len(cache) > slots:
                            cache.popitem(last=False)
        result.append(dict(state=state, demand_hits=hits, demand_misses=misses,
                           demand_bytes=volume, preload_bytes=0, total_bytes=volume))
    return result


def replay(trace, sizes, importance, budget, policy, force_explicit=False):
    if trace.dtype != bool or trace.ndim != 3 or trace.shape[1:] != sizes.shape or not trace.shape[0]:
        raise ValueError("Invalid token/layer/group trace")
    if policy not in POLICIES:
        raise ValueError("Unknown cache policy")
    shape = cache_shape(sizes, budget)
    slots = shape["retained_slots"]
    requests = int(trace.sum())
    volume = int((trace.sum(axis=0) * sizes).sum())
    bound = zero_lru_bound(trace)
    method = "direct_count"
    if policy == "lru" and (force_explicit or (slots and bound <= slots)):
        result = explicit_lru(trace, sizes, slots)
        method = "explicit_lru"
    elif policy in ("lru", "no_cache"):
        result = [dict(state=state, demand_hits=0, demand_misses=requests,
                       demand_bytes=volume, preload_bytes=0, total_bytes=volume)
                  for state in ("cold", "warm")]
        method = "distinct_other_layer_bound" if policy == "lru" else "direct_count"
    else:
        hot = hot_mask(importance, sizes, slots, policy)
        hits = int((trace & hot).sum())
        demand = volume - int(((trace & hot).sum(axis=0) * sizes).sum())
        preload = int(sizes[hot].sum())
        result = [dict(state=state, demand_hits=hits, demand_misses=requests-hits,
                       demand_bytes=demand, preload_bytes=preload if state == "cold" else 0,
                       total_bytes=demand + (preload if state == "cold" else 0))
                  for state in ("cold", "warm")]
    return [{**shape, **row, "policy": policy, "method": method,
             "distinct_other_layer_lower_bound": bound,
             "selected_request_bytes": volume} for row in result]


def group_arrays(means, orders, width, bytes_per_neuron):
    """Charge a short final group by its real weight bytes; slots use maximum size."""
    layers, neurons = means.shape
    if orders.shape != means.shape or any(not np.array_equal(np.sort(p), np.arange(neurons)) for p in orders):
        raise ValueError("Invalid physical permutation")
    groups = math.ceil(neurons / width)
    sizes = np.tile(np.minimum(width, neurons - np.arange(groups) * width) * bytes_per_neuron, (layers, 1))
    values = np.take_along_axis(means, orders, axis=1)
    values = np.pad(values, ((0, 0), (0, groups * width-neurons))).reshape(layers, groups, width).sum(axis=2)
    return sizes.astype(np.int64), values


def dense_static_frontier(means, layouts, widths, budgets, bytes_per_neuron, document_tokens):
    """Strongest dense baseline can choose any declared layout/width, not the sparse one's."""
    candidates = []
    for name, orders in layouts.items():
        for width in widths:
            sizes, importance = group_arrays(means, np.asarray(orders), width, bytes_per_neuron)
            for budget in budgets:
                shape = cache_shape(sizes, budget)
                for policy in ("static_global", "static_equal_layer"):
                    hot = hot_mask(importance, sizes, shape["retained_slots"], policy)
                    retained_bytes = int(sizes[hot].sum())
                    demand = sum(document_tokens) * (int(sizes.sum()) - retained_bytes)
                    for state in ("cold", "warm"):
                        candidates.append({"layout": name, "group_width": width, "policy": policy,
                            **shape, "state": state, "retained_weight_bytes": retained_bytes,
                            "total_bytes": demand + (len(document_tokens)*retained_bytes if state == "cold" else 0)})
    return { (budget, state): min((r for r in candidates if r["budget_bytes"] == budget and r["state"] == state),
                    key=lambda r: (r["total_bytes"], r["layout"], r["group_width"], r["policy"]))
             for budget in budgets for state in ("cold", "warm") }

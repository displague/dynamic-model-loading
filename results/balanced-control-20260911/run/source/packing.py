"""Calibration-only co-activation signatures and capacity-constrained grouping."""

import math

import torch

from .ffn import ImportanceCollector, validate_order


class CoactivationCollector(ImportanceCollector):
    """Uniform token reservoir using random priorities, independent of activations."""

    def __init__(self, mlp, capacity=512, seed=1729):
        super().__init__(mlp)
        if capacity <= 0:
            raise ValueError("Reservoir capacity must be positive")
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self.keys = torch.empty(0, dtype=torch.float64)
        self.samples = torch.empty((0, self.norms.numel()), dtype=torch.float32)
        self.ordinals = torch.empty(0, dtype=torch.long)

    def __call__(self, module, args):
        previous_tokens = self.tokens
        super().__call__(module, args)
        z = args[0].detach().reshape(-1, self.norms.numel())
        keys = torch.rand(z.shape[0], generator=self.generator, dtype=torch.float64)
        combined = torch.cat((self.keys, keys))
        selected = combined.argsort(descending=True, stable=True)[:self.capacity]
        old_count = self.keys.numel()
        # Copy only newly admitted vectors from GPU, not every observed activation.
        new_indices = selected[selected >= old_count] - old_count
        new_samples = (z.index_select(0, new_indices.to(z.device)).float().abs() * self.norms).cpu()
        old_indices = selected[selected < old_count]
        self.samples = torch.cat((self.samples[old_indices], new_samples))
        self.keys = torch.cat((self.keys[old_indices], keys[new_indices]))
        self.ordinals = torch.cat((self.ordinals[old_indices], new_indices + previous_tokens))

    def coactivation_order(self, group_width=32, sketch_dim=64, iterations=5, seed=1729):
        if not self.samples.shape[0]:
            raise ValueError("No calibration samples")
        return balanced_coactivation_order(self.samples, group_width, sketch_dim, iterations, seed)


def balanced_coactivation_order(samples, group_width=32, sketch_dim=64, iterations=5, seed=1729):
    """Recursive approximate principal-axis bisection with exact leaf capacities.

    samples[token, neuron] contains nonnegative contribution importance. Normalizing
    each neuron emphasizes its pattern over tokens, rather than its global popularity.
    """
    if samples.ndim != 2 or min(samples.shape) <= 0 or min(group_width, sketch_dim, iterations) <= 0:
        raise ValueError("Invalid signature dimensions or grouping settings")
    if not torch.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Expected finite nonnegative signatures")
    generator = torch.Generator().manual_seed(seed)
    profiles = samples.float().T.contiguous()
    profiles = profiles / profiles.norm(dim=1, keepdim=True).clamp_min(1e-12)
    signs = torch.randint(0, 2, (profiles.shape[1], sketch_dim), generator=generator).float() * 2 - 1
    features = profiles @ signs / math.sqrt(sketch_dim)

    def partition(indices):
        if len(indices) <= group_width:
            return indices
        block = features[indices]
        block = block - block.mean(0, keepdim=True)
        axis = torch.randn(sketch_dim, generator=generator)
        for _ in range(iterations):
            axis = block.T @ (block @ axis)
            norm = axis.norm()
            if norm <= 1e-12:
                break
            axis = axis / norm
        if axis.norm() <= 1e-12:
            ranked = indices
        else:
            ranked = indices[(block @ axis).argsort(stable=True)]
        groups = math.ceil(len(indices) / group_width)
        cut = (groups // 2) * group_width
        return torch.cat((partition(ranked[:cut]), partition(ranked[cut:])))

    order = partition(torch.arange(samples.shape[1]))
    validate_order(order, samples.shape[1])
    return order

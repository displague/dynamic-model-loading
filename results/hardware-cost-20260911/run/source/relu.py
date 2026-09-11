"""OPT's two biased ReLU projections, without altering its enclosing decoder layer."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ffn import select_groups, validate_order


def dimensions(layer):
    if not all(isinstance(getattr(layer, name, None), nn.Linear) for name in ("fc1", "fc2")):
        raise ValueError("Expected OPT fc1/fc2 linear projections")
    neurons, hidden = layer.fc1.weight.shape
    if layer.fc2.weight.shape != (hidden, neurons):
        raise ValueError("Mismatched OPT FFN dimensions")
    if layer.fc1.bias is None or layer.fc2.bias is None:
        raise ValueError("This OPT control requires both projection biases")
    return hidden, neurons


def extract(model):
    if model.config.model_type != "opt" or model.config.activation_function != "relu":
        raise ValueError("Only the declared OPT ReLU architecture is supported")
    layers = list(model.model.decoder.layers)
    for layer in layers:
        dimensions(layer)
    return layers


@torch.no_grad()
def repack_(layer, order):
    _, neurons = dimensions(layer)
    validate_order(order, neurons)
    order = order.to(layer.fc1.weight.device)
    targets = (layer.fc1.weight, layer.fc1.bias, layer.fc2.weight)
    originals = [target.detach().clone() for target in targets]
    replacements = [value.index_select(axis, order) for value, axis in zip(originals, (0, 0, 1), strict=True)]
    try:
        for target, value in zip(targets, replacements, strict=True):
            target.copy_(value)
    except (Exception, KeyboardInterrupt):
        for target, original in zip(targets, originals, strict=True):
            target.copy_(original)
        raise


@contextmanager
def layout(layers, orders):
    restored = []
    try:
        for layer, order in zip(layers, orders, strict=True):
            targets = (layer.fc1.weight, layer.fc1.bias, layer.fc2.weight)
            originals = [target.detach().clone() for target in targets]
            # Register pre-layout bytes before any mutation, including failed entry.
            restored.append((targets, originals))
            repack_(layer, order)
        yield
    finally:
        restore_originals(restored)


@torch.no_grad()
def restore_originals(restored):
    """Finish restoring all layers before propagating a recoverable cleanup interruption.

    Retry an interrupted/failed copy once from immutable original bytes. Persistent
    device failures cannot guarantee restoration; still attempt every remaining tensor
    and report the failure instead of silently accepting the model's state.
    """
    deferred, failed = None, []
    for targets, originals in reversed(restored):
        for target, original in zip(targets, originals, strict=True):
            for attempt in range(2):
                try:
                    target.copy_(original)
                    break
                except (Exception, KeyboardInterrupt) as error:
                    if deferred is None:
                        deferred = error
                    if attempt == 1:
                        failed.append(error)
    if failed:
        raise RuntimeError(f"Failed to restore {len(failed)} FFN tensors after retry") from failed[0]
    if deferred is not None:
        raise deferred


@contextmanager
def hooks(layers, observers):
    handles = []
    try:
        for layer, observer in zip(layers, observers, strict=True):
            handles.append(layer.fc2.register_forward_pre_hook(observer))
        yield
    finally:
        for handle in handles:
            handle.remove()


def dense_forward(layer, x):
    return layer.fc2(F.relu(layer.fc1(x)))


@torch.inference_mode()
def grouped_forward(layer, x, width):
    hidden, neurons = dimensions(layer)
    if width <= 0:
        raise ValueError("Positive group width required")
    output = torch.zeros((*x.shape[:-1], hidden), dtype=torch.float32, device=x.device)
    for start in range(0, neurons, width):
        end = min(start + width, neurons)
        z = F.relu(F.linear(x, layer.fc1.weight[start:end], layer.fc1.bias[start:end]))
        output.add_(F.linear(z, layer.fc2.weight[:, start:end]).float())
    return (output + layer.fc2.bias.float()).to(x.dtype)


class ImportanceCollector:
    def __init__(self, layer):
        _, neurons = dimensions(layer)
        self.norms = layer.fc2.weight.detach().float().norm(dim=0)
        self.total = torch.zeros(neurons, dtype=torch.float64, device=self.norms.device)
        self.tokens = 0

    def __call__(self, module, args):
        z = args[0].detach().reshape(-1, self.norms.numel())
        self.total += (z.float().abs() * self.norms).double().sum(0)
        self.tokens += len(z)

    def order(self):
        if not self.tokens:
            raise ValueError("No calibration observations")
        return self.total.argsort(descending=True, stable=True).cpu()


class Probe:
    def __init__(self, layer, width, mode):
        self.hidden, self.neurons = dimensions(layer)
        if width <= 0 or mode not in ("exact_zero", "approximate_75"):
            raise ValueError("Invalid ReLU probe")
        self.width, self.mode = width, mode
        self.groups = math.ceil(self.neurons / width)
        self.norms = layer.fc2.weight.detach().float().norm(dim=0)
        self.neuron_bytes = (2*self.hidden + 1)*layer.fc1.weight.element_size()
        self.output_bias_bytes = layer.fc2.bias.numel()*layer.fc2.bias.element_size()
        self.pending = None

    def __call__(self, module, args):
        if self.pending is not None:
            raise ValueError("Consume each document's accounting before another forward")
        z = args[0]
        if (z < 0).any() or not torch.isfinite(z).all():
            raise ValueError("Expected finite nonnegative ReLU activations")
        active = F.pad(z.ne(0), (0, self.groups*self.width-self.neurons)).reshape(-1, self.groups, self.width).any(-1)
        if self.mode == "exact_zero":
            mask = active.repeat_interleave(self.width, -1)[:, :self.neurons].reshape_as(z)
        else:
            mask = select_groups(z.float().abs()*self.norms, self.width, .75)
        tokens = z.numel() // self.neurons
        selected_neurons = int(mask.sum().item())
        self.pending = {"token_observations": tokens, "zero_neurons": int(z.eq(0).sum().item()),
            "neuron_observations": z.numel(), "zero_groups": int((~active).sum().item()),
            "group_observations": tokens*self.groups,
            "selected_neurons": selected_neurons,
            "selected_group_weight_bytes": selected_neurons*self.neuron_bytes,
            "fixed_output_bias_bytes": tokens*self.output_bias_bytes,
            "hypothetical_selected_ffn_bytes": selected_neurons*self.neuron_bytes + tokens*self.output_bias_bytes,
            "full_ffn_bytes": tokens*(self.neurons*self.neuron_bytes+self.output_bias_bytes)}
        return (z*mask, *args[1:])

    def take(self):
        if self.pending is None:
            raise ValueError("Missing ReLU accounting")
        result, self.pending = self.pending, None
        return result

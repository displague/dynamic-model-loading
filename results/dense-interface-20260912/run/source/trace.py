"""Applied hindsight masks, with separate individual-neuron cover diagnostics."""

import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F


class TraceRecorder:
    def __init__(self, group_width, keep):
        self.group_width, self.keep = group_width, keep
        self.pending = None

    def __call__(self, scores, mask):
        if self.pending is not None:
            raise ValueError("Trace must be consumed after each document forward")
        scores = scores.detach().reshape(-1, scores.shape[-1])
        mask = mask.detach().reshape_as(scores)
        if not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Trace importance must be finite and nonnegative")
        width, stride = scores.shape[-1], self.group_width
        groups = math.ceil(width / stride)
        individual_count = math.ceil(width * self.keep)
        top = scores.argsort(dim=-1, descending=True, stable=True)[:, :individual_count]
        individual = torch.zeros_like(mask).scatter_(-1, top, True)
        covered = F.pad(individual, (0, groups * stride - width)).reshape(-1, groups, stride).any(-1)
        covered_neurons = covered.repeat_interleave(stride, dim=-1)[:, :width].sum().item()
        self.pending = (
            mask[:, ::stride].cpu().numpy(),
            {"top_individual_neurons": individual.numel() // width * individual_count,
             "top_individual_neurons_in_applied_mask": (individual & mask).sum().item(),
             "neurons_in_individual_cover_groups": covered_neurons,
             "applied_neurons": mask.sum().item()},
        )

    def take(self):
        if self.pending is None:
            raise ValueError("Missing applied trace")
        result, self.pending = self.pending, None
        return result


def save_trace(path, recorders, metadata):
    values = [recorder.take() for recorder in recorders]
    # The declared Qwen study has homogeneous FFN dimensions. Fail closed otherwise.
    masks = np.stack([value[0] for value in values], axis=1)
    metadata = {**metadata, "shape": list(masks.shape), "bitorder": "little",
                "geometry_per_layer": [value[1] for value in values]}
    with path.open("xb") as stream:
        np.savez_compressed(stream, bits=np.packbits(masks.reshape(-1), bitorder="little"),
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return {"path": "traces/" + path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metadata}


def load_trace(path, receipt):
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("Trace hash mismatch")
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata != {k: v for k, v in receipt.items() if k not in ("path", "sha256")}:
            raise ValueError("Trace metadata mismatch")
        shape = metadata["shape"]
        if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Invalid trace shape")
        count = math.prod(shape)
        bits = stored["bits"]
        if bits.dtype != np.uint8 or bits.shape != (math.ceil(count / 8),):
            raise ValueError("Invalid packed trace")
        unpacked = np.unpackbits(bits, bitorder="little")
        if unpacked[count:].any():
            raise ValueError("Nonzero trace padding")
        return unpacked[:count].reshape(shape).astype(bool)

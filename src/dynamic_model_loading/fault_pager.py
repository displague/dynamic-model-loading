"""Physical FFN pages and a causal side index for the fault-pager protocol.

These components deliberately expose every cache decision.  They make no claim that
the approximate draft is numerically equivalent to its dense target; target
verification belongs to the experiment runner.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
import sys
from collections import Counter
from types import MethodType

import torch
from torch.nn import functional as F

from .ffn import dimensions


def metadata_bytes(*objects):
    """Python object bytes, deduplicated; tensor storage is accounted separately."""
    seen = set()
    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            total += sum(size(k)+size(v) for k,v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            total += sum(size(v) for v in obj)
        elif isinstance(obj, (PageKey, PagePayload, PageCatalog)):
            total += size(vars(obj))
        return total
    return sum(size(obj) for obj in objects)


@dataclass(frozen=True)
class PageKey:
    layer: int
    page: int


@dataclass(frozen=True)
class PagePayload:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.gate, self.up, self.down))


class PageCatalog:
    """CPU-backed paired SwiGLU pages, with no implicit layout permutation."""

    def __init__(self, mlp, layer: int, width: int):
        hidden, neurons = dimensions(mlp)
        if type(layer) is not int or layer < 0 or type(width) is not int or width <= 0:
            raise ValueError("Layer must be nonnegative and page width must be positive")
        self.layer, self.hidden, self.neurons, self.width = layer, hidden, neurons, width
        self._gate = mlp.gate_proj.weight.detach().cpu()
        self._up = mlp.up_proj.weight.detach().cpu()
        self._down = mlp.down_proj.weight.detach().cpu()

    @property
    def pages(self) -> int:
        return math.ceil(self.neurons / self.width)

    def payload(self, page: int) -> PagePayload:
        if type(page) is not int or not 0 <= page < self.pages:
            raise ValueError("Unknown page")
        start, stop = page * self.width, min((page + 1) * self.width, self.neurons)
        return PagePayload(self._gate[start:stop], self._up[start:stop], self._down[:, start:stop])


class PageCache:
    """Byte-bounded LRU cache whose event records form the transfer ledger."""

    def __init__(self, capacity_bytes: int, device: torch.device | str, *, mode="lru", sink=None):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Cache capacity must be a positive integer")
        self.capacity_bytes, self.device = capacity_bytes, torch.device(device)
        if mode not in ("lru", "eager", "prefetch"):
            raise ValueError("Invalid cache mode")
        self.mode, self.sink = mode, sink
        self._entries: OrderedDict[PageKey, PagePayload] = OrderedDict()
        self.used_bytes = 0
        self.events: list[dict] = []
        self.stats = Counter()
        self.prefetched = set()
        self.staging = None
        self.clock = ((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      if self.device.type == "cuda" else None)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def record(self, event):
        event.setdefault("used_bytes", self.used_bytes)
        self.stats[event["outcome"]] += 1
        if event["outcome"] == "load":
            self.stats[event["request"] + "_bytes"] += event["bytes"]
            self.stats["h2d_bytes"] += event["h2d_bytes"]
            self.stats["transfer_wall_ms"] += event["wall_ms"]
            self.stats["transfer_cuda_ms"] += event["cuda_ms"]
        if event["outcome"] == "hit":
            self.stats[event["request"] + "_hits"] += 1
        self.stats["peak_payload_bytes"] = max(self.stats["peak_payload_bytes"], self.used_bytes)
        if self.sink is None:
            self.events.append(event)
        else:
            self.sink(event)

    def clear(self):
        for key in list(self._entries):
            self.release(key, "episode_end")

    def release(self, key, reason="eager_release"):
        value = self._entries.pop(key)
        size = value.bytes
        self.used_bytes -= size
        del value
        if key in self.prefetched:
            self.prefetched.remove(key)
            self.record({"key": key, "request": "prefetch", "outcome": "cancelled_prefetch", "bytes": 0})
        self.record({"key": key, "request": reason, "outcome": "release", "bytes": size})

    def get(self, key: PageKey, source: PagePayload, *, request: str) -> PagePayload:
        if request not in ("demand", "prefetch"):
            raise ValueError("Unknown page request kind")
        if key in self._entries:
            value = self._entries.pop(key)
            self._entries[key] = value
            if request == "demand":
                self.prefetched.discard(key)
            self.record({"key": key, "request": request, "outcome": "hit", "bytes": 0,
                                "used_bytes": self.used_bytes})
            return value
        size = source.bytes
        if size > self.capacity_bytes:
            raise ValueError("One page exceeds the cache capacity")
        evicted = []
        while self.used_bytes + size > self.capacity_bytes:
            old_key = next(iter(self._entries))
            self.release(old_key, "eviction")
            evicted.append(old_key)
        self.sync()
        started = time.perf_counter()
        # One contiguous, pinned page; transposed down columns preserve neuron pairing.
        n, d = source.gate.shape
        if self.staging is None or self.staging.shape != (3, n, d):
            self.staging = torch.empty((3, n, d), dtype=source.gate.dtype,
                                       pin_memory=self.device.type == "cuda")
        self.staging[0].copy_(source.gate)
        self.staging[1].copy_(source.up)
        self.staging[2].copy_(source.down.T)
        if self.clock:
            self.clock[0].record()
        packed = self.staging.to(self.device, copy=True, non_blocking=True)
        if self.clock:
            self.clock[1].record()
            self.clock[1].synchronize()
        value = PagePayload(packed[0], packed[1], packed[2].T)
        cuda_ms = self.clock[0].elapsed_time(self.clock[1]) if self.clock else 0.0
        wall_ms = (time.perf_counter() - started) * 1000
        self._entries[key] = value
        self.used_bytes += value.bytes
        if request == "prefetch":
            self.prefetched.add(key)
        self.record({"key": key, "request": request, "outcome": "load", "bytes": value.bytes,
                     "h2d_bytes": size if self.device.type == "cuda" else 0,
                     "cuda_ms": cuda_ms, "wall_ms": wall_ms,
                     "evicted": evicted, "used_bytes": self.used_bytes})
        return value


class SideIndex:
    """Deterministic FFN-input medoids with a precomputed page ordering."""

    def __init__(self, centroids: torch.Tensor, rankings: torch.Tensor):
        if centroids.ndim != 2 or rankings.ndim != 2 or centroids.shape[0] != rankings.shape[0]:
            raise ValueError("Centroids and rankings must have matching two-dimensional medoid axes")
        if centroids.numel() == 0 or rankings.shape[1] == 0 or rankings.dtype != torch.long:
            raise ValueError("Empty or malformed side index")
        if not torch.isfinite(centroids).all() or centroids.device != rankings.device:
            raise ValueError("Nonfinite or mixed-device side index")
        expected = torch.arange(rankings.shape[1], device=rankings.device).expand_as(rankings)
        if not torch.equal(rankings.sort(dim=1).values, expected):
            raise ValueError("Every ranking must be a page permutation")
        self.centroids, self.rankings = centroids.float(), rankings

    @classmethod
    def build(cls, inputs: torch.Tensor, page_scores: torch.Tensor, medoids: int) -> "SideIndex":
        if inputs.ndim != 2 or page_scores.ndim != 2 or inputs.shape[0] != page_scores.shape[0]:
            raise ValueError("Inputs and page scores must have matching rows")
        if type(medoids) is not int or not 0 < medoids <= inputs.shape[0]:
            raise ValueError("Invalid medoid count")
        if not torch.isfinite(inputs).all() or not torch.isfinite(page_scores).all():
            raise ValueError("Side-index inputs must be finite")
        vectors = inputs.detach().float().cpu()
        chosen = [0]
        nearest = (vectors - vectors[0]).square().sum(dim=1)
        for _ in range(1, medoids):
            # torch.argmax deterministically chooses the lowest index for equal maxima.
            nearest[chosen] = -1
            next_index = int(nearest.argmax().item())
            chosen.append(next_index)
            nearest = torch.minimum(nearest, (vectors - vectors[next_index]).square().sum(dim=1))
        selected_scores = page_scores.detach().float().cpu()[chosen]
        ranking = selected_scores.argsort(dim=1, descending=True, stable=True).long()
        result = cls(vectors[chosen], ranking)
        result.source_positions = chosen
        return result

    def select(self, value: torch.Tensor, pages: int) -> torch.Tensor:
        if value.shape[-1] != self.centroids.shape[1] or type(pages) is not int or not 0 < pages <= self.rankings.shape[1]:
            raise ValueError("Invalid side-index lookup")
        flat = value.detach().float().reshape(-1, value.shape[-1])
        if flat.device != self.centroids.device:
            raise ValueError("Move the side index to the FFN input device before lookup")
        nearest = (flat[:, None, :] - self.centroids[None, :, :]).square().sum(dim=-1).argmin(dim=1)
        return self.rankings.index_select(0, nearest)[:, :pages]

    def to(self, device: torch.device | str) -> "SideIndex":
        """Return an explicit device-resident controller copy for charged accounting."""
        return SideIndex(self.centroids.to(device), self.rankings.to(device))

    @property
    def bytes(self) -> int:
        return self.centroids.numel() * self.centroids.element_size() + self.rankings.numel() * self.rankings.element_size()


@torch.inference_mode()
def paged_swiglu(x: torch.Tensor, catalog: PageCatalog, cache: PageCache, pages: torch.Tensor,
                 activation, *, request: str = "demand") -> torch.Tensor:
    """Execute only selected pages, retaining a ledger-visible cache transaction."""
    if x.ndim < 2 or x.shape[-1] != catalog.hidden or pages.ndim != 2 or pages.shape[0] != x.numel() // catalog.hidden:
        raise ValueError("Paged SwiGLU inputs do not match the catalogue")
    result = torch.zeros((pages.shape[0], catalog.hidden), device=x.device, dtype=torch.float32)
    flat = x.reshape(pages.shape[0], catalog.hidden)
    selected_pages = pages.to("cpu")
    cache.record({"request": "controller", "outcome": "selection_copy",
                         "bytes": selected_pages.numel() * selected_pages.element_size(),
                         "d2h_bytes": pages.numel() * pages.element_size() if pages.device.type == "cuda" else 0,
                         "used_bytes": cache.used_bytes})
    for row, selected in enumerate(selected_pages.tolist()):
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate selected page")
        for page in sorted(selected):
            payload = cache.get(PageKey(catalog.layer, page), catalog.payload(page), request=request)
            z = activation(F.linear(flat[row:row + 1], payload.gate)) * F.linear(flat[row:row + 1], payload.up)
            result[row:row + 1].add_(F.linear(z, payload.down).float())
            # No outstanding view may keep an evicted page alive beyond the budget.
            del payload, z
            if cache.mode == "eager":
                cache.release(PageKey(catalog.layer, page))
    return result.reshape(*x.shape[:-1], catalog.hidden).to(x.dtype)


class PagedDraft:
    """Replace compatible FFN forwards with side-indexed physical-page execution.

    The caller owns the model and must call :meth:`restore` before using that model
    as a dense target.  The wrapper intentionally does not implement verification.
    """

    def __init__(self, mlps: list, indexes: list[SideIndex], *, page_width: int,
                 selected_pages: int, capacity_bytes: int, device: torch.device | str,
                 mode="lru", sink=None):
        if len(mlps) != len(indexes) or not mlps:
            raise ValueError("Every FFN needs exactly one side index")
        if type(selected_pages) is not int or selected_pages <= 0:
            raise ValueError("Invalid selected page count")
        # Validate all layers before mutation; catalogues are made after evacuation
        # so that CPU parameters and catalogue views share storage, not duplicate it.
        dims = [dimensions(mlp) for mlp in mlps]
        if any(index.rankings.shape[1] != math.ceil(n / page_width) or
               index.centroids.shape[1] != h or selected_pages > math.ceil(n / page_width)
               for index, (h, n) in zip(indexes, dims, strict=True)):
            raise ValueError("Side-index pages do not match its FFN catalogue")
        self.cache = PageCache(capacity_bytes, device, mode=mode, sink=sink)
        self.catalogs = []
        self.pending = {}
        self.full = False
        self.indexes = [index.to(device) for index in indexes]
        self._originals = []
        try:
            for layer, (mlp, index) in enumerate(zip(mlps, self.indexes, strict=True)):
                original = mlp.source.forward if hasattr(mlp, "source") else mlp.forward
                module = mlp.source if hasattr(mlp, "source") else mlp
                original_device = next(module.parameters()).device
                self._originals.append((module, original, original_device))
                module.to("cpu")
                catalog = PageCatalog(mlp, layer, page_width)
                self.catalogs.append(catalog)

                def forward(module_self, x, *, _catalog=catalog, _index=index, _activation=mlp.act_fn):
                    self.cache.sync()
                    start = time.perf_counter()
                    if self.full:
                        output = torch.zeros_like(x)
                        for page in range(_catalog.pages):
                            payload = self.cache.get(PageKey(_catalog.layer, page), _catalog.payload(page), request="demand")
                            z = _activation(F.linear(x, payload.gate)) * F.linear(x, payload.up)
                            output.add_(F.linear(z, payload.down))
                            del payload, z
                        self.cache.sync()
                        return output
                    selection_gpu = _index.select(x, selected_pages)
                    selection = selection_gpu.cpu()
                    self.cache.record({"outcome": "lookup", "request": "controller", "bytes": 0,
                        "d2h_bytes": selection.numel()*selection.element_size() if selection_gpu.is_cuda else 0,
                        "wall_ms": (time.perf_counter()-start)*1000})
                    output = paged_swiglu(x, _catalog, self.cache, selection, _activation)
                    if self.cache.mode == "prefetch":
                        self.pending[_catalog.layer] = selection[-1, :2].tolist()
                    self.cache.sync()
                    self.cache.record({"outcome": "layer", "request": "execution", "layer": _catalog.layer,
                                       "selected_pages": selection.tolist(), "bytes": 0,
                                       "host_metadata_bytes": self.host_metadata_bytes,
                                       "wall_ms": (time.perf_counter() - start) * 1000})
                    return output

                module.forward = MethodType(forward, module)
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        self.cache.clear()
        while self._originals:
            module, forward, device = self._originals.pop()
            module.forward = forward
            module.to(device)

    @property
    def controller_bytes(self) -> int:
        return sum(index.bytes for index in self.indexes)

    @property
    def host_metadata_bytes(self):
        return metadata_bytes(self.catalogs, self.cache._entries, self.cache.prefetched, self.pending)

    def begin_token(self):
        pending, self.pending = self.pending, {}
        for layer, pages in sorted(pending.items()):
            for page in pages:
                self.cache.get(PageKey(layer, page), self.catalogs[layer].payload(page), request="prefetch")

    def reset(self):
        self.cache.clear()
        self.pending.clear()
        self.cache.stats.clear()

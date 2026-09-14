"""Physical FFN pages and a causal side index for the fault-pager protocol.

These components deliberately expose every cache decision.  They make no claim that
the approximate draft is numerically equivalent to its dense target; target
verification belongs to the experiment runner.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from types import MethodType

import torch
from torch.nn import functional as F

from .ffn import dimensions


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

    def __init__(self, capacity_bytes: int, device: torch.device | str):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Cache capacity must be a positive integer")
        self.capacity_bytes, self.device = capacity_bytes, torch.device(device)
        self._entries: OrderedDict[PageKey, PagePayload] = OrderedDict()
        self.used_bytes = 0
        self.events: list[dict] = []

    def get(self, key: PageKey, source: PagePayload, *, request: str) -> PagePayload:
        if request not in ("demand", "prefetch"):
            raise ValueError("Unknown page request kind")
        if key in self._entries:
            value = self._entries.pop(key)
            self._entries[key] = value
            self.events.append({"key": key, "request": request, "outcome": "hit", "bytes": 0,
                                "used_bytes": self.used_bytes})
            return value
        size = source.bytes
        if size > self.capacity_bytes:
            raise ValueError("One page exceeds the cache capacity")
        evicted = []
        while self.used_bytes + size > self.capacity_bytes:
            old_key, old = self._entries.popitem(last=False)
            self.used_bytes -= old.bytes
            evicted.append(old_key)
        value = PagePayload(*(tensor.to(self.device, non_blocking=False) for tensor in
                              (source.gate, source.up, source.down)))
        self._entries[key] = value
        self.used_bytes += value.bytes
        self.events.append({"key": key, "request": request, "outcome": "load", "bytes": value.bytes,
                            "evicted": evicted, "used_bytes": self.used_bytes})
        return value


class SideIndex:
    """Deterministic FFN-input medoids with a precomputed page ordering."""

    def __init__(self, centroids: torch.Tensor, rankings: torch.Tensor):
        if centroids.ndim != 2 or rankings.ndim != 2 or centroids.shape[0] != rankings.shape[0]:
            raise ValueError("Centroids and rankings must have matching two-dimensional medoid axes")
        if centroids.numel() == 0 or rankings.shape[1] == 0 or rankings.dtype != torch.long:
            raise ValueError("Empty or malformed side index")
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
            next_index = int(nearest.argmax().item())
            chosen.append(next_index)
            nearest = torch.minimum(nearest, (vectors - vectors[next_index]).square().sum(dim=1))
        selected_scores = page_scores.detach().float().cpu()[chosen]
        ranking = selected_scores.argsort(dim=1, descending=True, stable=True).long()
        return cls(vectors[chosen], ranking)

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
    cache.events.append({"request": "controller", "outcome": "selection_copy",
                         "bytes": selected_pages.numel() * selected_pages.element_size(),
                         "used_bytes": cache.used_bytes})
    for row, selected in enumerate(selected_pages.tolist()):
        for page in selected:
            payload = cache.get(PageKey(catalog.layer, page), catalog.payload(page), request=request)
            z = activation(F.linear(flat[row:row + 1], payload.gate)) * F.linear(flat[row:row + 1], payload.up)
            result[row:row + 1].add_(F.linear(z, payload.down).float())
    return result.reshape(*x.shape[:-1], catalog.hidden).to(x.dtype)


class PagedDraft:
    """Replace compatible FFN forwards with side-indexed physical-page execution.

    The caller owns the model and must call :meth:`restore` before using that model
    as a dense target.  The wrapper intentionally does not implement verification.
    """

    def __init__(self, mlps: list, indexes: list[SideIndex], *, page_width: int,
                 selected_pages: int, capacity_bytes: int, device: torch.device | str):
        if len(mlps) != len(indexes) or not mlps:
            raise ValueError("Every FFN needs exactly one side index")
        self.cache = PageCache(capacity_bytes, device)
        self.catalogs = [PageCatalog(mlp, layer, page_width) for layer, mlp in enumerate(mlps)]
        self.indexes = [index.to(device) for index in indexes]
        self._originals = []
        if any(index.rankings.shape[1] != catalog.pages or selected_pages > catalog.pages
               for index, catalog in zip(self.indexes, self.catalogs, strict=True)):
            raise ValueError("Side-index pages do not match its FFN catalogue")
        try:
            for layer, (mlp, index, catalog) in enumerate(zip(mlps, self.indexes, self.catalogs, strict=True)):
                original = mlp.source.forward if hasattr(mlp, "source") else mlp.forward
                module = mlp.source if hasattr(mlp, "source") else mlp
                original_device = next(module.parameters()).device
                module.to("cpu")

                def forward(module_self, x, *, _catalog=catalog, _index=index, _activation=mlp.act_fn):
                    selection = _index.select(x, selected_pages)
                    return paged_swiglu(x, _catalog, self.cache, selection, _activation)

                module.forward = MethodType(forward, module)
                self._originals.append((module, original, original_device))
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        while self._originals:
            module, forward, device = self._originals.pop()
            module.forward = forward
            module.to(device)

    @property
    def controller_bytes(self) -> int:
        return sum(index.bytes for index in self.indexes)

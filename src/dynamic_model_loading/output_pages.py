"""Final-FFN physical sensors with embedded two-bit base and three refinements.

Only the final FFN has this representation. Its output follows the last KV write,
so re-evaluating its readout cannot change already computed K/V. A pairwise margin
is not a full-vocabulary argmax certificate; the dense verifier remains separate.
"""
import torch
from torch.nn import functional as F

from .progressive_precision import ProgressiveDraft, IncrementCache, pack_codes, unpack_codes
from .ffn import dimensions


def split_four_base(base):
    """Split packed upper-four-bit codes into upper two and next two planes."""
    if base.dtype!=torch.uint8 or base.ndim!=2 or base.shape[1]%2:
        raise ValueError('Invalid packed four-bit base')
    codes = torch.stack((base&15,base>>4),-1).reshape(base.shape[0],-1)
    return pack_codes(codes>>2,2),pack_codes(codes&3,2)


def projected_margin(basis,hidden,delta,eps):
    """Predicted two-token logit difference, including the new common RMS scale."""
    h = hidden+delta
    return (basis*h).sum()/torch.sqrt(h.square().mean()+eps)


class OutputPageDraft(ProgressiveDraft):
    @torch.inference_mode()
    def __init__(self,mlps,*,page_width=256,**kwargs):
        h,n = dimensions(mlps[-1])
        if type(page_width) is not int or page_width<1 or n%page_width:
            raise ValueError('Page width must partition the final FFN')
        super().__init__(mlps,**kwargs)
        self.last = len(self.layers)-1
        self.width,self.count = page_width,n//page_width
        self.low = []
        first = []
        for q in self.layers[-1]:
            self.construction_d2h_bytes += q.base.numel() if q.base.is_cuda else 0
            q.base = q.base.cpu()  # Not secretly resident alongside the new base.
            lo,hi = split_four_base(q.base)
            self.low.append(lo.to(self.device))
            self.construction_h2d_bytes += lo.numel() if self.device.type=='cuda' else 0
            first.append(hi)
        self.page_host = [torch.stack(first),*self.host[-1]]
        self.page_cache = IncrementCache((3,page_width,h//4),1,self.device)
        self.prefill = False
        self.force_full = False
        self.deltas = {}
        self.x = self.base_y = None

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for key,t in self.tensors()
            if not (key.startswith(f'{len(self.layers)-1}.') and key.endswith('.base'))) + sum(t.numel() for t in getattr(self,'low',[]))

    def reset(self):
        super().reset()
        if hasattr(self,'page_cache'):
            self.page_cache.reset()
        self.deltas = {}

    def begin_token(self):
        self.cache.begin_token(False)
        self.page_cache.begin_token(False)
        self.deltas = {}

    def _base(self):
        for j,q in enumerate(self.layers[-1]):
            out = self.workspace[j]
            unpack_codes(self.low[j],2,out,self.byte_scratch,self.float_scratch)
            out.mul_(64).add_(31.5)
            out.view(out.shape[0],-1,q.group).mul_(q.scale[...,None]).add_(q.minimum[...,None])

    def _compute(self,lo=0,hi=None):
        w = self.workspace[:,lo:hi]
        return F.linear(self.activation(F.linear(self.x,w[0]))*F.linear(self.x,w[1]),w[2].T)

    @torch.inference_mode()
    def acquire(self,page):
        if type(page) is not int or not 0<=page<self.count or self.x is None:
            raise ValueError('Invalid or premature output-page observation')
        if page in self.deltas:
            raise ValueError('Each page may be observed only once per draft call')
        lo,hi = page*self.width,(page+1)*self.width
        base = self._compute(lo,hi)
        for stage,(factor,offset) in enumerate(((16,24),(4,6),(1,1.5))):
            payload = self.page_cache.get((page,stage),self.page_host[stage][:,lo:hi])
            for j,q in enumerate(self.layers[-1]):
                delta = self.delta[:self.width]
                unpack_codes(payload[j],2,delta,self.byte_scratch[:self.width],self.float_scratch[:self.width])
                delta.mul_(factor).sub_(offset)
                delta.view(self.width,-1,q.group).mul_(q.scale[lo:hi,:,None])
                self.workspace[j,lo:hi].add_(delta)
        value = (self._compute(lo,hi)-base).clone()
        self.deltas[page] = value
        return value

    @torch.inference_mode()
    def partial(self,pages):
        pages = list(pages)
        if len(set(pages))!=len(pages):
            raise ValueError('Duplicate partial pages')
        self._base()
        self.deltas = {}
        for page in pages:
            self.acquire(page)
        return self._compute().clone()

    @torch.inference_mode()
    def forward_one(self,x,layer,activation):
        if layer!=self.last:
            self.mode = 'q8' if self.prefill else 'q6'
            return super().forward_one(x,layer,activation)
        self.x,self.activation = x.clone(),activation
        self._base()
        self.base_y = self._compute().clone()
        if self.prefill or self.force_full:
            for page in range(self.count):
                self.acquire(page)
            return self._compute()
        return self.base_y

"""Resident low-bit FFNs with causal, feedback-directed physical correction pages."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MethodType

import torch
from torch.nn import functional as F

from .fault_pager import PageCache, PageCatalog, PageKey
from .ffn import dimensions


@dataclass
class Packed2:
    codes: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    group: int

    @classmethod
    def pack(cls, weight, group=128):
        if (weight.ndim!=2 or weight.dtype!=torch.float32 or type(group) is not int or
                group<4 or group%4 or weight.shape[1]%group or not torch.isfinite(weight).all()):
            raise ValueError('Invalid two-bit input or group')
        w = weight.detach().reshape(weight.shape[0],-1,group)
        lo, hi = w.amin(-1), w.amax(-1)
        scale = (hi-lo)/3
        if not torch.isfinite(scale).all():
            raise ValueError('Nonfinite quantization scale')
        safe = torch.where(scale>0,scale,torch.ones_like(scale))
        q = ((w-lo[...,None])/safe[...,None]).round().clamp(0,3).to(torch.uint8)
        q = q.reshape(weight.shape[0],-1,4)
        packed = q[:,:,0] | (q[:,:,1]<<2) | (q[:,:,2]<<4) | (q[:,:,3]<<6)
        return cls(packed.contiguous(),lo.contiguous(),scale.contiguous(),group)

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.codes,self.minimum,self.scale))

    def unpack_into(self, out, byte_scratch, float_scratch):
        n,c = self.codes.shape
        if (out.shape!=(n,c*4) or out.dtype!=torch.float32 or
                byte_scratch.shape!=(n,c) or byte_scratch.dtype!=torch.uint8 or
                float_scratch.shape!=(n,c) or float_scratch.dtype!=torch.float32 or
                any(t.device!=out.device for t in (self.codes,self.minimum,self.scale,byte_scratch,float_scratch))):
            raise ValueError('Invalid unpack workspace')
        view = out.view(n,c,4)
        for i in range(4):
            torch.bitwise_right_shift(self.codes,2*i,out=byte_scratch)
            byte_scratch.bitwise_and_(3)
            float_scratch.copy_(byte_scratch)
            view[:,:,i].copy_(float_scratch)
        out.view(n,-1,self.group).mul_(self.scale[...,None]).add_(self.minimum[...,None])
        return out


def squared_norm(vector):
    return math.fsum(v*v for v in vector)


def choose_page(sketches,debt,used):
    """Python-double arithmetic is logged and independently reproducible."""
    if not sketches or not debt or any(len(v)!=len(debt) for v in sketches):
        raise ValueError('Invalid correction sketch shape')
    if not all(math.isfinite(v) for row in [*sketches,debt] for v in row):
        raise ValueError('Nonfinite correction sketch')
    candidates = [(2*math.fsum(a*b for a,b in zip(debt,v,strict=True))-squared_norm(v),-p)
                  for p,v in enumerate(sketches) if p not in used]
    if not candidates:
        return None,0.0
    gain,neg_page = max(candidates)
    return (-neg_page if gain>0 else None),gain


@dataclass
class ResidualLayer:
    base: list[Packed2]
    basis: torch.Tensor
    gate_error: torch.Tensor
    up_error: torch.Tensor
    down_full: torch.Tensor
    down_base: torch.Tensor

    def tensors(self):
        result = dict(basis=self.basis,gate_error=self.gate_error,up_error=self.up_error,
                      down_full=self.down_full,down_base=self.down_base)
        for name,q in zip(('gate','up','down'),self.base,strict=True):
            result.update({name+'.codes':q.codes,name+'.minimum':q.minimum,name+'.scale':q.scale})
        return result

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors().values())


class ResidualDraft:
    MODES = ('dense_stream','base','fixed','debt','complete')

    @torch.inference_mode()
    def __init__(self,mlps,centroids,*,page_width=256,group=128,rank=16,max_pages=4,
                 device='cuda:0',seed=20260914,page_sink=None,policy_sink=None,check=lambda:None):
        if not mlps or len(mlps)!=len(centroids):
            raise ValueError('Missing FFN inputs')
        dims = [dimensions(m) for m in mlps]
        h,n = dims[0]
        if (any(d!=(h,n) for d in dims) or type(page_width) is not int or page_width<=0 or
                n%page_width or type(rank) is not int or not 0<rank<=h or
                any(c.shape!=(rank,h) for c in centroids) or type(max_pages) is not int or
                not 0<max_pages<=n//page_width):
            raise ValueError('Invalid fixed representation dimensions')
        self.device = torch.device(device)
        self.width,self.max_pages = page_width,max_pages
        self.workspace = torch.empty(3,n,h,device=self.device,dtype=torch.float32)
        self.byte_scratch = torch.empty(n,h//4,device=self.device,dtype=torch.uint8)
        self.float_scratch = torch.empty(n,h//4,device=self.device,dtype=torch.float32)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        projection = (torch.randint(0,2,(h,rank),generator=generator).float()*2-1)/math.sqrt(rank)
        self.projection = projection.to(self.device)
        self.construction_h2d_bytes = projection.numel()*4 if self.device.type=='cuda' else 0
        self.cache = PageCache(3*page_width*h*4,self.device,mode='eager',sink=page_sink)
        self.layers,self.catalogs,self.originals = [],[],[]
        self.pending = {}
        self.mode = 'base'
        self.policy_sink = policy_sink or (lambda row:None)
        try:
            for i,(mlp,c) in enumerate(zip(mlps,centroids,strict=True)):
                check()
                module = mlp.source
                self.originals.append((module,module.forward,next(module.parameters()).device))
                module.to('cpu')
                catalog = PageCatalog(mlp,i,page_width)
                self.catalogs.append(catalog)
                for param,view in ((mlp.gate_proj.weight,catalog._gate),(mlp.up_proj.weight,catalog._up),
                                   (mlp.down_proj.weight,catalog._down)):
                    if param.is_cuda or param.untyped_storage().data_ptr()!=view.untyped_storage().data_ptr():
                        raise RuntimeError('Catalogue did not alias host source')
                basis = torch.linalg.qr(c.T.to(self.device),mode='reduced').Q.contiguous()
                self.construction_h2d_bytes += c.numel()*4 if self.device.type=='cuda' else 0
                base,sketch = [],{}
                for j,host in enumerate((catalog._gate,catalog._up,catalog._down.T)):
                    weight = host.to(self.device).contiguous()
                    self.construction_h2d_bytes += weight.numel()*4 if self.device.type=='cuda' else 0
                    packed = Packed2.pack(weight,group)
                    q = packed.unpack_into(self.workspace[j],self.byte_scratch,self.float_scratch)
                    if j<2:
                        sketch[j] = (weight-q)@basis
                    else:
                        sketch[2],sketch[3] = weight@self.projection,q@self.projection
                    base.append(packed)
                    del weight,q
                self.layers.append(ResidualLayer(base,basis,sketch[0],sketch[1],sketch[2],sketch[3]))
                def forward(module_self,x,*,layer=i,activation=mlp.act_fn):
                    flat = x.reshape(-1,h)
                    return torch.cat([self.forward_one(flat[k:k+1],layer,activation)
                                      for k in range(len(flat))],0).reshape_as(x)
                module.forward = MethodType(forward,module)
        except BaseException:
            self.restore()
            raise

    @property
    def resident_bytes(self):
        return sum(layer.bytes for layer in self.layers)+self.projection.numel()*4

    @property
    def workspace_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.workspace,self.byte_scratch,self.float_scratch))

    def reset(self):
        self.cache.clear()
        self.cache.stats.clear()
        self.pending.clear()

    def begin_token(self):
        pass  # No prior-token state, prefetch, target feedback or hidden acquisition.

    def restore(self):
        self.cache.clear()
        while self.originals:
            module,forward,device = self.originals.pop()
            module.forward = forward
            module.to(device)

    @torch.inference_mode()
    def forward_one(self,x,layer,activation):
        if self.mode not in self.MODES:
            raise ValueError('Unknown correction mode')
        self.cache.sync()
        started = time.perf_counter()
        catalog = self.catalogs[layer]
        data = self.layers[layer]
        sketches,choices,used = [],[],set()
        d2h = 0
        stop_gain = None
        if self.mode=='dense_stream':
            output = torch.zeros_like(x)
            order = list(range(catalog.pages))
        else:
            for j,q in enumerate(data.base):
                q.unpack_into(self.workspace[j],self.byte_scratch,self.float_scratch)
            g,u = F.linear(x,self.workspace[0]),F.linear(x,self.workspace[1])
            z = activation(g)*u
            output = F.linear(z,self.workspace[2].T)
            if self.mode in ('fixed','debt'):
                projected = x@data.basis
                estimated = activation(g+F.linear(projected,data.gate_error))*(u+F.linear(projected,data.up_error))
                # Summation is confined to each physical FFN page; down sketches
                # are full vs quantized columns, not an activation-mass selector.
                vector = ((estimated.reshape(catalog.pages,self.width,1)*data.down_full.reshape(catalog.pages,self.width,-1))-
                          (z.reshape(catalog.pages,self.width,1)*data.down_base.reshape(catalog.pages,self.width,-1))).sum(1)
                sketches = vector.cpu().tolist()
                d2h += vector.numel()*4 if x.is_cuda else 0
                debt = [math.fsum(v[j] for v in sketches) for j in range(len(sketches[0]))]
                order = sorted(range(catalog.pages),key=lambda p:(-squared_norm(sketches[p]),p))[:self.max_pages]
                del vector,estimated,projected
            else:
                order = list(range(catalog.pages)) if self.mode=='complete' else []
        limit = self.max_pages if self.mode=='debt' else len(order)
        for step in range(limit):
            if self.mode=='debt':
                page,gain = choose_page(sketches,debt,used)
                if page is None:
                    stop_gain = gain
                    break
            else:
                page,gain = order[step],None
            key = PageKey(layer,page)
            payload = self.cache.get(key,catalog.payload(page),request='demand')
            exact_z = activation(F.linear(x,payload.gate))*F.linear(x,payload.up)
            exact = F.linear(exact_z,payload.down)
            if self.mode=='dense_stream':
                delta = exact
            else:
                a,b = page*self.width,(page+1)*self.width
                delta = exact-F.linear(z[:,a:b],self.workspace[2,a:b].T)
            output.add_(delta)
            observed = []
            if self.mode in ('fixed','debt'):
                observed = (delta@self.projection).flatten().cpu().tolist()
                d2h += len(observed)*4 if x.is_cuda else 0
                if self.mode=='debt':
                    debt = [a-b for a,b in zip(debt,observed,strict=True)]
            choices.append(dict(page=page,predicted_gain=gain,observed=observed))
            used.add(page)
            del payload,exact_z,exact,delta
            self.cache.release(key)
        self.cache.sync()
        self.policy_sink(dict(kind='layer',layer=layer,mode=self.mode,sketches=sketches,
            choices=choices,stop_gain=stop_gain,d2h_bytes=d2h,
            wall_ms=(time.perf_counter()-started)*1000))
        return output

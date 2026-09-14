"""Embedded precision with physical increment acquisition and causal cache admission.

The error proxy is deliberately heuristic, not a certificate. Persistent state
changes placement only: even state from rejected draft work is safe to retain.
"""
from dataclasses import dataclass
import math
from types import MethodType

import torch
from torch.nn import functional as F

from .ffn import dimensions


def pack_codes(codes, bits):
    if bits not in (2,4) or codes.dtype!=torch.uint8 or codes.ndim!=2 or codes.shape[1]%(8//bits):
        raise ValueError('Invalid packed codes')
    if torch.any(codes>2**bits-1):
        raise ValueError('Code out of range')
    fields=codes.reshape(codes.shape[0],-1,8//bits)
    out=torch.zeros_like(fields[:,:,0])
    for i in range(8//bits):
        out.bitwise_or_(fields[:,:,i]<<(i*bits))
    return out.contiguous()


def unpack_codes(codes,bits,out,byte_scratch,float_scratch):
    n,c=codes.shape
    fields=8//bits
    if out.shape!=(n,c*fields) or out.dtype!=torch.float32:
        raise ValueError('Invalid unpack output')
    b,f=byte_scratch[:,:c],float_scratch[:,:c]
    for i in range(fields):
        torch.bitwise_right_shift(codes,i*bits,out=b)
        b.bitwise_and_(2**bits-1)
        f.copy_(b)
        out.view(n,c,fields)[:,:,i].copy_(f)
    return out


@dataclass
class EmbeddedWeight:
    base: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    increments: tuple
    group: int

    @classmethod
    def encode(cls,weight,group=128):
        if (weight.ndim!=2 or weight.dtype!=torch.float32 or type(group) is not int or
                group<4 or group%4 or weight.shape[1]%group or not torch.isfinite(weight).all()):
            raise ValueError('Invalid embedded weight')
        w=weight.detach().reshape(weight.shape[0],-1,group)
        lo,hi=w.amin(-1),w.amax(-1)
        scale=(hi-lo)/255
        if not torch.isfinite(scale).all():
            raise ValueError('Nonfinite scale')
        safe=torch.where(scale>0,scale,torch.ones_like(scale))
        q=((w-lo[...,None])/safe[...,None]).round().clamp(0,255).to(torch.uint8).reshape_as(weight)
        return cls(pack_codes(q>>4,4),lo.contiguous(),scale.contiguous(),
                   (pack_codes((q>>2)&3,2),pack_codes(q&3,2)),group)

    def base_into(self,out,byte_scratch,float_scratch):
        unpack_codes(self.base,4,out,byte_scratch,float_scratch)
        out.mul_(16).add_(7.5)
        out.view(out.shape[0],-1,self.group).mul_(self.scale[...,None]).add_(self.minimum[...,None])

    def add_into(self,out,stage,codes,delta,byte_scratch,float_scratch):
        if stage not in (0,1):
            raise ValueError('Invalid refinement stage')
        unpack_codes(codes,2,delta,byte_scratch,float_scratch)
        delta.mul_(4 if stage==0 else 1).sub_(6 if stage==0 else 1.5)
        delta.view(delta.shape[0],-1,self.group).mul_(self.scale[...,None])
        out.add_(delta)


class IncrementCache:
    """Fixed allocations; unadmitted demand loads bypass persistent slots."""
    def __init__(self,shape,slots,device,sink=lambda row:None):
        if type(slots) is not int or slots<1:
            raise ValueError('Positive cache slots required')
        self.device=torch.device(device)
        self.pool=torch.empty((slots+1,*shape),dtype=torch.uint8,device=device)
        self.staging=torch.empty(shape,dtype=torch.uint8,pin_memory=self.device.type=='cuda')
        self.slots=slots
        self.sink=sink
        self.reset()

    def reset(self):
        self.entries={}
        self.scores={}
        self.admitted=set()
        self.token=0
        self.stats=dict(h2d_bytes=0,hits=0,loads=0,evictions=0)

    def begin_token(self,retain,static_keys=()):
        self.token+=1
        ranked=sorted((k for k,v in self.scores.items() if v>0),key=lambda k:(-self.scores[k],k))
        self.admitted=set(static_keys) if static_keys else set(ranked[:self.slots]) if retain else set()
        if len(self.admitted)>self.slots:
            raise ValueError('Too many static admissions')
        for key in sorted(set(self.entries)-self.admitted):
            self.entries.pop(key)
            self.stats['evictions']+=1
            self.sink(dict(kind='evict',token=self.token,key=list(key)))
        self.scores={k:v*.5 for k,v in self.scores.items()}

    def observe(self,key,value):
        if not math.isfinite(value) or value<0:
            raise ValueError('Nonfinite correction value')
        self.scores[key]=self.scores.get(key,0)+.5*value

    def get(self,key,host):
        if host.device.type!='cpu' or host.dtype!=torch.uint8 or host.shape!=self.staging.shape:
            raise ValueError('Invalid increment payload')
        if key in self.entries:
            slot=self.entries[key]
            kind='hit'
            self.stats['hits']+=1
        else:
            slot=next((i for i in range(self.slots) if i not in self.entries.values()),None) if key in self.admitted else self.slots
            if slot is None:
                raise RuntimeError('Cache admission overflow')
            self.staging.copy_(host)
            self.pool[slot].copy_(self.staging,non_blocking=False)
            if slot!=self.slots:
                self.entries[key]=slot
            kind='load'
            self.stats['loads']+=1
            self.stats['h2d_bytes']+=host.numel() if self.device.type=='cuda' else 0
        self.sink(dict(kind=kind,token=self.token,key=list(key),slot=slot,
                       bytes=host.numel() if kind=='load' and self.device.type=='cuda' else 0))
        return self.pool[slot]


class ProgressiveDraft:
    MODES=('q4','q6','q8','adaptive','retained','static')

    @torch.inference_mode()
    def __init__(self,mlps,*,device='cuda',group=128,slots=8,threshold=.02,
                 page_sink=lambda r:None,policy_sink=lambda r:None,check=lambda:None):
        dims=[dimensions(m) for m in mlps]
        if not dims or any(d!=dims[0] for d in dims) or not math.isfinite(threshold) or threshold<0:
            raise ValueError('Invalid progressive configuration')
        h,n=dims[0]
        self.device=torch.device(device)
        self.mode='q4'
        self.threshold=threshold
        self.pending={}
        self.originals=[]
        self.layers=[]
        self.host=[]
        self.policy_sink=policy_sink
        self.workspace=torch.empty(3,n,h,device=device)
        self.delta=torch.empty(n,h,device=device)
        self.byte_scratch=torch.empty(n,h//2,dtype=torch.uint8,device=device)
        self.float_scratch=torch.empty(n,h//2,device=device)
        self.cache=IncrementCache((3,n,h//4),slots,device,page_sink)
        self.construction_h2d_bytes=0
        self.construction_d2h_bytes=0
        try:
            for i,m in enumerate(mlps):
                check()
                module=m.source
                self.originals.append((module,module.forward,next(module.parameters()).device))
                module.to('cpu')
                layer=[]
                for source in (m.gate_proj.weight,m.up_proj.weight,m.down_proj.weight.T):
                    # Construct on CPU so full FP32 source copies do not consume
                    # the runtime's constrained CUDA allowance during setup.
                    w=source.detach().contiguous()
                    layer.append(EmbeddedWeight.encode(w,group))
                    del w
                slabs=[torch.stack([q.increments[s] for q in layer]).cpu() for s in (0,1)]
                for q in layer:
                    q.increments=()
                    for name in ('base','minimum','scale'):
                        value=getattr(q,name)
                        self.construction_h2d_bytes+=value.numel()*value.element_size() if self.device.type=='cuda' else 0
                        setattr(q,name,value.to(device))
                self.layers.append(layer)
                self.host.append(slabs)
                def forward(module_self,x,*,layer_id=i,act=m.act_fn):
                    flat=x.reshape(-1,h)
                    return torch.cat([self.forward_one(v[None],layer_id,act) for v in flat],0).reshape_as(x)
                module.forward=MethodType(forward,module)
        except BaseException:
            self.restore()
            raise

    def tensors(self):
        for i,layer in enumerate(self.layers):
            for j,q in enumerate(layer):
                for name in ('base','minimum','scale'):
                    yield f'{i}.{j}.{name}',getattr(q,name)

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for _,t in self.tensors())

    @property
    def workspace_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.workspace,self.delta,self.byte_scratch,self.float_scratch))

    def reset(self):
        self.cache.reset()
        self.pending.clear()

    def begin_token(self):
        keys=[(i,0) for i in range(min(self.cache.slots,len(self.layers)))] if self.mode=='static' else ()
        self.cache.begin_token(self.mode=='retained',keys)

    def restore(self):
        while self.originals:
            module,forward,device=self.originals.pop()
            module.forward=forward
            module.to(device)

    @torch.inference_mode()
    def forward_one(self,x,layer,activation):
        if self.mode not in self.MODES:
            raise ValueError('Invalid precision mode')
        def compute():
            return F.linear(activation(F.linear(x,self.workspace[0]))*F.linear(x,self.workspace[1]),self.workspace[2].T)
        for j,q in enumerate(self.layers[layer]):
            q.base_into(self.workspace[j],self.byte_scratch,self.float_scratch)
        y=compute()
        values=[]
        stages=0
        for stage in (0,1):
            if self.mode=='q4' or (stage==1 and (self.mode=='q6' or (self.mode in ('adaptive','retained','static') and values[0]/4<=self.threshold))):
                break
            payload=self.cache.get((layer,stage),self.host[layer][stage])
            for j,q in enumerate(self.layers[layer]):
                q.add_into(self.workspace[j],stage,payload[j],self.delta,self.byte_scratch,self.float_scratch)
            refined=compute()
            value=float(((refined-y).norm()/refined.norm().clamp_min(1e-12)).item())
            self.cache.observe((layer,stage),value)
            values.append(value)
            y=refined
            stages+=1
        self.policy_sink(dict(token=self.cache.token,layer=layer,mode=self.mode,stages=stages,
                              values=values,d2h_bytes=4*stages if x.is_cuda else 0))
        return y

"""Bayesian error-magnitude emulator and phase-explicit physical acquisition.

GP uncertainty is model-dependent. Document-block calibration is empirical here,
not an exactness certificate or a guarantee on adaptive generated trajectories.
"""
import hashlib
import math

import torch
from torch.nn import functional as F

from .progressive_precision import ProgressiveDraft


def block_quantile(scores, coverage=.9):
    if not scores or not 0<coverage<1 or not all(math.isfinite(v) for v in scores):
        raise ValueError('Invalid calibration blocks')
    rank=math.ceil((len(scores)+1)*coverage)
    return max(0.,sorted(scores)[rank-1]) if rank<=len(scores) else math.inf


class ErrorGP:
    """Small CPU float64 GP: shrinkage geometry, fixed RBF and noise prior."""
    def __init__(self,x,z):
        x,z=x.double().cpu(),z.double().cpu()
        if x.ndim!=2 or z.shape!=(len(x),) or len(x)<2 or not torch.isfinite(x).all() or not torch.isfinite(z).all():
            raise ValueError('Invalid GP training rows')
        self.xmean=x.mean(0)
        self.xscale=x.std(0,correction=0).clamp_min(1e-6)
        u=(x-self.xmean)/self.xscale
        cov=.8*(u.T@u/len(u))+.2*torch.eye(x.shape[1],dtype=torch.float64)
        self.geometry=torch.linalg.cholesky(cov)
        self.train=torch.linalg.solve_triangular(self.geometry,u.T,upper=False).T
        distance=(self.train[:,None]-self.train[None,:]).square().sum(-1)
        mask=~torch.eye(len(x),dtype=torch.bool)
        self.length2=distance[mask].median().clamp_min(1e-6)
        self.zmean=z.mean()
        self.zscale=z.std(correction=0).clamp_min(.1)
        self.factor=torch.linalg.cholesky(torch.exp(-.5*distance/self.length2)+
                                          (.1+1e-8)*torch.eye(len(x),dtype=torch.float64))
        self.alpha=torch.cholesky_solve(((z-self.zmean)/self.zscale)[:,None],self.factor)[:,0]
        self.q=math.inf
        self.constant_q=math.inf

    def predict(self,x):
        x=torch.as_tensor(x,dtype=torch.float64,device='cpu')
        if x.shape!=self.xmean.shape or not torch.isfinite(x).all():
            raise ValueError('Invalid GP feature')
        u=torch.linalg.solve_triangular(self.geometry,((x-self.xmean)/self.xscale)[:,None],upper=False)[:,0]
        k=torch.exp(-.5*(self.train-u).square().sum(-1)/self.length2)
        v=torch.linalg.solve_triangular(self.factor,k[:,None],upper=False)[:,0]
        mu=self.zmean+self.zscale*(k@self.alpha)
        sd=self.zscale*torch.sqrt((1.1-v.square().sum()).clamp_min(1e-12))
        return float(mu),float(sd)

    def calibrate(self,blocks):
        gp,constant=[],[]
        for x,z in blocks:
            if len(x)!=len(z) or not len(x):
                raise ValueError('Empty calibration block')
            predictions=[self.predict(v) for v in x]
            gp.append(max((float(y)-mu)/sd for y,(mu,sd) in zip(z,predictions,strict=True)))
            constant.append(max((float(y)-float(self.zmean))/float(self.zscale) for y in z))
        self.q=block_quantile(gp)
        self.constant_q=block_quantile(constant)

    def bounds(self,x):
        mu,sd=self.predict(x)
        return dict(mean=mu,sd=sd,upper=mu+self.q*sd,
                    constant=float(self.zmean+self.constant_q*self.zscale))

    def tensors(self):
        names=('xmean','xscale','geometry','train','length2','zmean','zscale','factor','alpha')
        return {**{n:getattr(self,n).contiguous() for n in names},
                'q':torch.tensor([self.q,self.constant_q],dtype=torch.float64)}


class CalibratedDraft(ProgressiveDraft):
    MODES=('q6','p8d6','p8d8','p8mean','p8gp','p8constant','shadow')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        h=self.workspace.shape[-1]
        generator=torch.Generator(device='cpu').manual_seed(20260915)
        self.projection=(torch.randint(0,2,(h,16),generator=generator).float()*2-1).to(self.device)/4
        self.models=[]
        self.kv_sink=lambda r:None
        self.mode='p8d8'
        self.prefill_tokens=8
        self.prefix_digest=None
        self.controller_seconds=0.

    def reset(self):
        super().reset()
        self.prefix_digest=None
        self.controller_seconds=0.

    def begin_token(self):
        if self.mode not in self.MODES or type(self.prefill_tokens) is not int or self.prefill_tokens<1:
            raise ValueError('Invalid phase configuration')
        self.cache.begin_token(False,[(i,0) for i in range(min(self.cache.slots,len(self.layers)))])

    @property
    def is_prefill(self):
        return self.cache.token<=self.prefill_tokens

    def after_cache_step(self,cache):
        self._check_prefix(cache,'step')

    def after_cache_crop(self,cache):
        self._check_prefix(cache,'crop')

    def _check_prefix(self,cache,kind):
        if self.cache.token<self.prefill_tokens:
            return
        if cache.get_seq_length()<self.prefill_tokens:
            raise RuntimeError('Rollback discarded prefill')
        sha=hashlib.sha256()
        size=0
        for layer in cache.layers:
            for value in (layer.keys,layer.values):
                if value.dtype!=torch.float32:
                    raise RuntimeError('Draft KV dtype changed')
                data=value[:,:,:self.prefill_tokens].detach().contiguous().cpu().numpy().tobytes()
                sha.update(data)
                size+=len(data)
        value=sha.hexdigest()
        if self.prefix_digest is None:
            self.prefix_digest=value
        elif value!=self.prefix_digest:
            raise RuntimeError('Previously computed prefix KV mutated')
        self.kv_sink(dict(kind=kind,token=self.cache.token,length=cache.get_seq_length(),
                          sha256=value,readback_bytes=size if self.device.type=='cuda' else 0))

    @torch.inference_mode()
    def forward_one(self,x,layer,activation):
        import time
        def compute():
            return F.linear(activation(F.linear(x,self.workspace[0]))*F.linear(x,self.workspace[1]),self.workspace[2].T)
        def acquire(stage):
            payload=self.cache.get((layer,stage),self.host[layer][stage])
            for j,q in enumerate(self.layers[layer]):
                q.add_into(self.workspace[j],stage,payload[j],self.delta,self.byte_scratch,self.float_scratch)
            return compute()
        for j,q in enumerate(self.layers[layer]):
            q.base_into(self.workspace[j],self.byte_scratch,self.float_scratch)
        y4=compute()
        y6=acquire(0)
        feature=None
        estimate=None
        observed=None
        readback=0
        decode=not self.is_prefill
        need_feature=decode and self.mode in ('shadow','p8mean','p8gp','p8constant')
        if need_feature:
            start=time.perf_counter()
            norm=y6.norm().clamp_min(1e-12)
            feature=torch.cat(((x/x.norm().clamp_min(1e-12)@self.projection).flatten(),
                torch.stack([((y6-y4).norm()/norm).clamp_min(1e-8).log(),norm.log()]))).cpu().tolist()
            readback+=18*4 if x.is_cuda else 0
            if self.models:
                estimate=self.models[layer].bounds(feature)
            self.controller_seconds+=time.perf_counter()-start
        if self.is_prefill:
            promoted=self.mode!='q6'
        elif self.mode in ('p8d8','shadow'):
            promoted=True
        elif self.mode in ('q6','p8d6'):
            promoted=False
        else:
            if estimate is None:
                raise RuntimeError('Fit and calibrate before acquisition')
            key={'p8mean':'mean','p8gp':'upper','p8constant':'constant'}[self.mode]
            promoted=estimate[key]>math.log(self.threshold)
        y=y6
        if promoted:
            y8=acquire(1)
            if need_feature:
                observed=float(((y8-y6).norm()/y6.norm().clamp_min(1e-12)).clamp_min(1e-8).log().item())
                readback+=4 if x.is_cuda else 0
            if self.mode!='shadow' or self.is_prefill:
                y=y8
        self.policy_sink(dict(token=self.cache.token,layer=layer,mode=self.mode,
            phase='prefill' if self.is_prefill else 'decode',stages=1+int(promoted),
            feature=feature,estimate=estimate,observed_log=observed,d2h_bytes=readback))
        return y

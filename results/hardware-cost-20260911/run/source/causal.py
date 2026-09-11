"""Causal masks and calibration-only ridge fitting, separate from dense audits."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .ffn import dimensions


def rank_mask(scores, keep=.9):
    count = math.ceil(scores.shape[-1]*keep)
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def group_scores(z, norms, width):
    values = z.float().abs()*norms
    count = math.ceil(values.shape[-1]/width)
    return F.pad(values, (0,count*width-values.shape[-1])).reshape(-1,count,width).sum(-1)


class RidgeFit:
    """Streaming FP64 calibration statistics; no diagnostic or generated labels."""
    def __init__(self, hidden, groups, features, seed, device):
        generator = torch.Generator().manual_seed(seed)
        self.projection = (torch.randn(hidden,features,generator=generator)/math.sqrt(hidden)).to(device)
        size = 2*features
        self.x = torch.zeros(size,dtype=torch.float64,device=device)
        self.y = torch.zeros(groups,dtype=torch.float64,device=device)
        self.xx = torch.zeros(size,size,dtype=torch.float64,device=device)
        self.xy = torch.zeros(size,groups,dtype=torch.float64,device=device)
        self.count = 0

    def features(self, x):
        projected = x.reshape(-1,self.projection.shape[0]).float() @ self.projection
        return torch.cat((projected,projected.abs()),-1)

    def observe(self, x, scores):
        features, labels = self.features(x).double(), scores.double()
        if len(features) != len(labels):
            raise ValueError("Calibration feature/label mismatch")
        self.x += features.sum(0)
        self.y += labels.sum(0)
        self.xx += features.T @ features
        self.xy += features.T @ labels
        self.count += len(features)

    def tensors(self):
        return {key:getattr(self,key).detach().cpu().contiguous() for key in ("projection","x","y","xx","xy")}

    def finish(self, ridge=.01):
        if self.count < 2:
            raise ValueError("Insufficient calibration")
        stats = self.tensors()
        mean, target = stats['x']/self.count, stats['y']/self.count
        cov = stats['xx']/self.count-mean[:,None]*mean[None,:]
        scale = cov.diag().clamp_min(0).sqrt().clamp_min(1e-6)
        cov = cov/scale[:,None]/scale[None,:]
        cross = (stats['xy']/self.count-mean[:,None]*target[None,:])/scale[:,None]
        weight = torch.linalg.solve(cov+torch.eye(len(mean),dtype=torch.float64)*ridge,cross)
        return {"projection":stats['projection'], "mean":mean.float(), "scale":scale.float(),
                "weight":weight.float().contiguous(), "intercept":target.float()}


class Selector:
    """State updates only receive scores from groups that this selector requested."""
    def __init__(self, mode, prior, norms, width, learned=None, keep=.9, alpha=.1):
        if mode not in ("static","recency","ema","learned"):
            raise ValueError("Unknown causal selector")
        if prior.ndim != 1 or norms.ndim != 1 or math.ceil(len(norms)/width) != len(prior):
            raise ValueError("Selector dimensions mismatch")
        self.mode, self.width, self.keep, self.alpha = mode, width, keep, alpha
        self.prior = prior.detach().clone()
        self.norms = norms.detach().clone() if mode in ("recency","ema") else None
        self.state = self.prior.clone() if mode in ("recency","ema") else None
        self.fixed = rank_mask(self.prior,keep) if mode == "static" else None
        self.learned = {k:v.to(prior.device).clone() for k,v in learned.items()} if mode == "learned" else {}

    def reset(self):
        if self.state is not None:
            self.state.copy_(self.prior)

    def tensors(self):
        result = {k:getattr(self,k) for k in ("prior","norms","state","fixed") if getattr(self,k) is not None}
        return {**result,**{"learned_"+k:v for k,v in self.learned.items()}}

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors().values())

    def query(self, x):
        if x.numel() != x.shape[-1]:
            raise ValueError("Causal selector requires exactly one token")
        if self.mode == "static":
            return self.fixed
        if self.mode in ("recency","ema"):
            return rank_mask(self.state,self.keep)
        t = self.learned
        feature = x.reshape(-1).float() @ t['projection']
        feature = (torch.cat((feature,feature.abs()))-t['mean'])/t['scale']
        return rank_mask(feature @ t['weight']+t['intercept'],self.keep)

    def observe_selected(self, selected_z, mask):
        if self.state is None:
            return
        # API contract: callers supply zero for every omitted activation.
        scores = group_scores(selected_z,self.norms,self.width).reshape(-1)
        if self.mode == "recency":
            self.state.copy_(torch.where(mask,scores,self.prior))
        else:
            self.state.copy_(torch.where(mask,(1-self.alpha)*self.state+self.alpha*scores,self.state))


class AppliedProbe:
    """Pre-FFN causal query; post-gate independent hindsight audit and masked update."""
    def __init__(self, mlp, selector, width=8, keep=.9, audit=True):
        self.mlp, self.selector, self.width, self.keep, self.audit = mlp,selector,width,keep,audit
        self.hidden,self.neurons = dimensions(mlp)
        self.norms = mlp.down_proj.weight.detach().float().norm(dim=0) if audit or selector is None else None
        self.mask = None
        self.masks, self.audits = [], []

    def before(self, module, args):
        if self.mask is not None:
            raise ValueError("Unconsumed selection")
        if args[0].numel() != self.hidden:
            raise ValueError("Probe requires incremental batch-one inference")
        if self.selector is not None:
            self.mask = self.selector.query(args[0])

    def down(self, module, args):
        z = args[0]
        if self.selector is None:
            self.mask = rank_mask(group_scores(z,self.norms,self.width),self.keep).reshape(-1)
        elif self.mask is None:
            raise ValueError("Causal query was not called before FFN")
        mask, self.mask = self.mask, None
        neuron_mask = mask.repeat_interleave(self.width)[:self.neurons]
        selected_z = z*neuron_mask
        if self.selector is not None:
            self.selector.observe_selected(selected_z,mask)
        self.masks.append(mask.detach().clone())
        if self.audit:
            scores = group_scores(z,self.norms,self.width).reshape(-1)
            hindsight = rank_mask(scores,self.keep)
            full = F.linear(z,self.mlp.down_proj.weight).float()
            omitted = F.linear(z-selected_z,self.mlp.down_proj.weight).float()
            numerator,denominator = omitted.norm(),full.norm()
            error = torch.where(denominator>0,numerator/denominator,
                                torch.where(numerator==0,0.,float('inf')))
            retained = torch.where(scores.sum()>0,(scores*mask).sum()/scores.sum(),1.)
            overlap = (mask & hindsight).float().sum()/hindsight.sum()
            self.audits.append(torch.stack((error,retained,overlap)))
        return (selected_z,*args[1:])

    def take(self):
        if self.mask is not None:
            raise ValueError("Unconsumed selection")
        masks = torch.stack(self.masks).cpu()
        audits = torch.stack(self.audits).cpu() if self.audit else None
        self.masks,self.audits = [],[]
        return masks,audits


@contextmanager
def probes(mlps, observers):
    handles=[]
    try:
        for mlp,observer in zip(mlps,observers,strict=True):
            handles.append(mlp.register_forward_pre_hook(observer.before))
            handles.append(mlp.down_proj.register_forward_pre_hook(observer.down))
        yield
    finally:
        for handle in handles:
            handle.remove()

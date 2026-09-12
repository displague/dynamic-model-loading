"""One-round causal acquisition with explicit observed state and isolated labels."""

from contextlib import contextmanager
import math

import numpy as np
import torch
from torch.nn import functional as F

from .causal import group_scores


CONDITIONS = (('initial',0), ('one_shot',28), ('one_shot',56), ('predetermined',28),
              ('predetermined',56), ('partial',28), ('partial',56), ('input_only',28),
              ('fallback',28), ('full',112))


class ExecutionFailure(ValueError):
    """Carry failing tensors across hook cleanup so the runner can persist them."""
    def __init__(self, message, tensors, context):
        super().__init__(message)
        self.tensors={k:v.detach().cpu().clone() for k,v in tensors.items()}
        self.context=dict(context)


def rank(scores, eligible, count):
    indices = scores.masked_fill(~eligible, -torch.inf).argsort(descending=True, stable=True)[:count]
    return torch.zeros_like(eligible).scatter_(0, indices, True) & eligible


def fit(features, targets, ridge=.01):
    """CPU FP64 fit from retained calibration examples, returning FP32 inference tensors."""
    x, y = features.cpu().double(), targets.cpu().double()
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x)<2:
        raise ValueError('Calibration dimensions mismatch')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite calibration')
    mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x-mean)/scale; intercept = y.mean(0)
    weight = torch.linalg.solve(x.T@x/len(x)+ridge*torch.eye(x.shape[1],dtype=x.dtype),
                                x.T@(y-intercept)/len(x))
    return dict(mean=mean.float(), scale=scale.float(), weight=weight.float(), intercept=intercept.float())


def predict(feature, model):
    return (feature-model['mean'])/model['scale']@model['weight']+model['intercept']


def projections(hidden, groups, seed, device):
    gen = torch.Generator().manual_seed(seed)
    return {'input_projection': (torch.randn(hidden,8,generator=gen)/math.sqrt(hidden)).to(device),
            'group_projection': (torch.randn(groups,8,generator=gen)/math.sqrt(groups)).to(device)}


class Controller:
    """Queries never accept dense output, omitted activations or evaluation labels."""
    def __init__(self, prior, hot, projection, models=None, mode='partial', extra=28, initial=1008):
        if mode not in {p[0] for p in CONDITIONS} or not 0 <= initial <= len(prior):
            raise ValueError('Invalid controller')
        if prior.ndim != 1 or prior.shape != hot.shape or hot.dtype != torch.bool or int(hot.sum())>initial:
            raise ValueError('Invalid resident core')
        if extra<0 or extra>len(prior)-initial:
            raise ValueError('Invalid acquisition size')
        self.prior, self.hot, self.projection, self.models = prior.clone(), hot.clone(), projection, models
        self.hot_count = int(hot.sum())
        self.mode, self.extra, self.initial = mode, extra, initial
        self.scale = prior.mean().clamp_min(1e-12)
        self.prior_log = (prior/self.scale).log1p()
        self.reset()

    def reset(self):
        self.history = self.prior_log.clone(); self.age = torch.zeros_like(self.prior)
        self.previous = torch.zeros(8, device=self.prior.device)
        self.base = self.first = self.partial_features = None
        self.fallback = False

    def begin(self, x):
        return self.begin_projected(x.reshape(-1)@self.projection['input_projection'])

    def begin_projected(self, projected):
        if self.first is not None: raise ValueError('Unconsumed controller visit')
        current = torch.cat((projected,projected.abs()))
        gp = self.projection['group_projection']
        change = (projected-self.previous).square().mean().sqrt().reshape(1)
        self.base = torch.cat((current,self.history@gp,(self.age/128).clamp_max(1)@gp,change))
        self.previous = projected.clone()
        if self.models is None:
            scores = self.history
        elif self.mode == 'input_only':
            scores = predict(current,self.models['input'])
        else:
            scores = predict(self.base,self.models['base'])
        count = self.initial + (self.extra if self.mode in ('one_shot','input_only') else 0)
        self.initial_scores = scores.clone()
        self.first = self.hot | rank(scores,~self.hot,count-self.hot_count)
        return self.first.clone()

    def evidence(self, observed_initial_log_scores, first_norm):
        if self.first is None or observed_initial_log_scores.shape != self.prior.shape:
            raise ValueError('Evidence outside a visit')
        # Mask at this boundary as well: caller cannot accidentally expose omitted scores.
        innovation = torch.where(self.first,observed_initial_log_scores-self.history,0.)
        gp = self.projection['group_projection']
        return torch.cat((self.base,innovation@gp,innovation.abs().mean().reshape(1),
                          torch.log1p(first_norm).reshape(1)))

    def acquire(self, observed_initial_log_scores, first_norm):
        if self.first is None: raise ValueError('Acquisition before initial selection')
        self.fallback = False
        self.acquisition_scores = torch.zeros_like(self.prior)
        self.detection = torch.zeros(1, device=self.prior.device)
        if self.mode in ('initial','one_shot','input_only'):
            return torch.zeros_like(self.first)
        if self.mode == 'full': return ~self.first
        if self.mode == 'predetermined' or self.models is None:
            scores = self.prior_log
        else:
            self.partial_features = self.evidence(observed_initial_log_scores,first_norm)
            scores = predict(self.partial_features,self.models['repair'])
            if self.mode == 'fallback':
                self.detection = predict(self.partial_features,self.models['detector'])
                self.fallback = bool(self.detection.item() > .05)
                if self.fallback: return ~self.first
        self.acquisition_scores = scores.clone()
        return rank(scores,~self.first,self.extra)

    def observe(self, observed_final_log_scores, final):
        if self.first is None or final.shape != self.first.shape:
            raise ValueError('Observation outside a visit')
        self.history = torch.where(final,.9*self.history+.1*observed_final_log_scores,self.history)
        self.age = torch.where(final,0.,self.age+1)
        self.base = self.first = self.partial_features = None


class Probe:
    """Dense diagnostic wrapper. Only masked observations cross the controller API."""
    def __init__(self, mlp, controller, collect=False):
        self.mlp, self.controller, self.collect = mlp, controller, collect
        self.norms = mlp.down_proj.weight.detach().float().norm(dim=0)
        self.x = self.z = self.initial = None
        self.receipts, self.calibration = [], []

    def before(self, module, args):
        if self.x is not None: raise ValueError('Unconsumed FFN input')
        if args[0].numel() != self.mlp.down_proj.weight.shape[0]:
            raise ValueError('Incremental FFN required')
        self.x = args[0]
        self.initial = self.controller.begin(self.x)

    def capture(self, module, args):
        if self.x is None or self.z is not None: raise ValueError('Unexpected gate/up result')
        self.z = args[0]

    def after(self, module, args, dense):
        c = self.controller; initial = self.initial
        if self.z is None: raise ValueError('Missing gate/up result')
        first_z = self.z*initial.repeat_interleave(8)
        first = F.linear(first_z,self.mlp.down_proj.weight)
        first_scores = group_scores(first_z,self.norms,8).reshape(-1).div(c.scale).log1p()
        first_norm = first.float().norm()
        base = c.base.clone()
        # Stored evidence is obtained from first_z only; dense output is audit-only.
        features = c.evidence(first_scores,first_norm)
        additions = c.acquire(first_scores,first_norm)
        final = initial | additions
        extra_z = self.z*additions.repeat_interleave(8)
        extra = F.linear(extra_z,self.mlp.down_proj.weight) if bool(additions.any()) else torch.zeros_like(first)
        corrected = first+extra
        selected_z = self.z*final.repeat_interleave(8)
        observed = group_scores(selected_z,self.norms,8).reshape(-1).div(c.scale).log1p()
        fallback = c.fallback
        scores = c.initial_scores.detach().cpu()
        acquisition = c.acquisition_scores.detach().cpu()
        detection = c.detection.detach().cpu()
        c.observe(observed,final)
        denominator = dense.float().norm().clamp_min(1e-12)
        audit = torch.stack(((first-dense).float().norm()/denominator,
                             (corrected-dense).float().norm()/denominator))
        row = {'input_projection':(self.x.reshape(-1)@c.projection['input_projection']).detach().cpu(),
            'x':self.x.reshape(-1).detach().cpu(),
            'observed':observed.detach().cpu(), 'initial':initial.detach().cpu(),
            'additions':additions.detach().cpu(), 'first_norm':first_norm.detach().cpu(),
            'audit':audit.detach().cpu(), 'fallback':torch.tensor(fallback)}
        row.update(initial_scores=scores, acquisition_scores=acquisition, detection=detection,
                   base_features=base.detach().cpu(), repair_features=features.detach().cpu())
        self.receipts.append(row)
        if self.collect:
            # Labels never feed this visit's actions or the controller's next state.
            labels = group_scores(self.z,self.norms,8).reshape(-1).div(c.scale).log1p()
            self.calibration.append({'base':base.cpu(), 'repair':features.cpu(),
                'labels':labels.cpu(), 'error':audit[:1].cpu(), 'x':self.x.reshape(-1).cpu()})
        if not torch.isfinite(corrected).all() or not torch.isfinite(observed).all() or not torch.isfinite(audit).all():
            raise ExecutionFailure('Nonfinite causal execution',dict(input=self.x,gate_up=self.z,
                first_output=first,corrected_output=corrected,dense_audit=dense,**row),
                {'position':len(self.receipts)-1,'layer':getattr(self,'layer',None),'mode':c.mode,'extra':c.extra})
        self.x = self.z = self.initial = None
        return corrected


@contextmanager
def apply(mlps, observers):
    handles = []
    try:
        for mlp, probe in zip(mlps,observers,strict=True):
            handles.append(mlp.source.register_forward_pre_hook(probe.before))
            handles.append(mlp.down_proj.register_forward_pre_hook(probe.capture))
            handles.append(mlp.source.register_forward_hook(probe.after))
        yield
    finally:
        for handle in handles: handle.remove()
        for probe in observers:
            probe.x = probe.z = probe.initial = None
            probe.controller.base = probe.controller.first = probe.controller.partial_features = None


def payload(observers, calibration=False):
    sequences = [p.calibration if calibration else p.receipts for p in observers]
    if not sequences or any(not rows for rows in sequences): raise ValueError('Empty observation grid')
    return {key:torch.stack([torch.stack([r[key] for r in rows]) for rows in sequences],1).contiguous()
            for key in sequences[0][0]}


def fit_models(data):
    result = []
    for layer in range(data['base'].shape[1]):
        base, repair = data['base'][:,layer], data['repair'][:,layer]
        labels, error = data['labels'][:,layer], data['error'][:,layer]
        result.append({'input':fit(base[:,:16],labels), 'base':fit(base,labels),
                       'repair':fit(repair,labels), 'detector':fit(repair,error)})
    return result


def flatten_models(models):
    return {f'{i}.{name}.{key}':value for i,layer in enumerate(models)
            for name,model in layer.items() for key,value in model.items()}


def unflatten_models(values, layers, device='cpu'):
    return [{name:{key:values[f'{i}.{name}.{key}'].to(device) for key in ('mean','scale','weight','intercept')}
             for name in ('input','base','repair','detector')} for i in range(layers)]


def traffic(receipt, hot, group_bytes=147456):
    initial, extra = receipt['initial'].numpy(), receipt['additions'].numpy()
    hot = np.asarray(hot,dtype=bool)
    if initial.dtype != bool or extra.dtype != bool or initial.shape != extra.shape or np.any(initial&extra):
        raise ValueError('Invalid acquisition masks')
    if not np.all((initial|extra)|~hot): raise ValueError('Resident groups were omitted')
    visits = initial.shape[0]*initial.shape[1]
    first_cold, added_cold = initial&~hot,extra&~hot
    return {'visits':visits, 'initial_cold_bytes':int(first_cold.sum())*group_bytes,
        'repair_cold_bytes':int(added_cold.sum())*group_bytes, 'resident_group_computations':int((initial&hot).sum()),
        'dispatch_bytes':int(first_cold.sum()+added_cold.sum())*8,
        'selected_groups':int((initial|extra).sum()),'rounds':int(extra.any(-1).sum()),
        'fallbacks':int(receipt['fallback'].sum()), 'preload_bytes':int(hot.sum())*group_bytes}

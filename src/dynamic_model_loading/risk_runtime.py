"""Physical final-FFN acquisition; only dense target verification commits tokens."""
import hashlib,time
from types import MethodType
import numpy as np
import torch
from .decision_acquisition import acquire
from .decision_field import fingerprint,kv_storage_bytes
from .fault_generation import kv_bytes


class KVJournal:
    def __init__(self): self.reset()
    def reset(self): self.entries={0:hashlib.sha256(b'').hexdigest()}
    def before(self,length,digest):
        if self.entries.get(length)!=digest: raise ValueError('Draft KV history drift')
    def append(self,length,digest): self.entries[length]=digest
    def crop(self,length,digest):
        self.before(length,digest)
        self.entries={n:v for n,v in self.entries.items() if n<=length}


class RiskReadout:
    def __init__(self,model,pager,index,record,context,*,budget=17,prefill_tokens=4,full_vocabulary=False):
        self.model,self.pager,self.index,self.record,self.context=model,pager,index,record,context
        self.budget,self.prefill_tokens=budget,prefill_tokens
        self.full_vocabulary=full_vocabulary
        if not 0<budget<=pager.count or prefill_tokens<1: raise ValueError('Invalid runtime budget/prefill')
        self.width=model.config.hidden_size
        self.original=model.forward
        self.captured={}; self.journal=KVJournal()
        self.hooks=[model.model.norm.register_forward_pre_hook(
            lambda m,a:self.captured.update(h=a[0].detach().clone())),
            model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(
            lambda m,a:self.captured.update(residual=a[0].detach().clone()))]
        def forward(_model,*args,**kwargs): return self.forward(*args,**kwargs)
        model.forward=MethodType(forward,model)
        self.reset('fixed')

    def reset(self,condition):
        allowed=('fixed','risk','all35','fullrisk') if self.full_vocabulary else ('fixed','risk','all35')
        if condition not in allowed: raise ValueError('Unknown physical condition')
        self.condition=condition; self.calls=0; self.pager.reset(); self.pending=self.pager.pending
        self.journal.reset(); self.captured.clear(); self.last_cache=None

    def begin_token(self):
        self.calls+=1
        self.pager.prefill=self.calls<=self.prefill_tokens
        self.pager.force_full=False
        self.context.update(call=self.calls,prefill=self.pager.prefill)
        self.pager.begin_token()

    @torch.inference_mode()
    def forward(self,*args,**kwargs):
        started=time.perf_counter(); cache=kwargs['past_key_values']
        length=cache.get_seq_length()
        before,size0=fingerprint(cache,length) if length else (hashlib.sha256(b'').hexdigest(),0)
        self.journal.before(length,before)
        result=self.original(*args,**kwargs)
        prior_after,size1=fingerprint(cache,length)
        post,size2=fingerprint(cache,length+1)
        if before!=prior_after or cache.get_seq_length()!=length+1:
            raise ValueError('Appending token changed prior KV')
        policy=None; axis_bytes=0; scalar_bytes=0; readout_bytes=16; full_trace=None
        if not self.pager.prefill:
            h=self.captured['h'].flatten(); residual=self.captured['residual']
            top=result.logits[0,-1].topk(2).indices.tolist()
            u,v=map(int,top)
            rms=torch.sqrt(h.square().mean()+self.model.model.norm.variance_epsilon)
            basis=(self.model.lm_head.weight[u]-self.model.lm_head.weight[v])*self.model.model.norm.weight
            axis=basis/rms
            margin=float((axis*h).sum().item())
            base_pair=float((result.logits[0,-1,u]-result.logits[0,-1,v]).item())
            base_rms=float(rms.item())
            observations=[]
            observation_seconds=0.
            def observe(page):
                nonlocal observation_seconds
                probe_started=time.perf_counter()
                delta=self.pager.acquire(int(page)).flatten()
                value=float((axis*delta).sum().item())
                observations.append(value)
                observation_seconds+=time.perf_counter()-probe_started
                return value
            controller_started=time.perf_counter()
            axis_cpu=None
            if self.condition=='fullrisk':
                from .vocabulary_risk import acquire as acquire_vocabulary
                axis_cpu=axis.cpu().numpy().astype(np.float64)
                axis_bytes=self.width*4
                def observe_vector(page):
                    nonlocal observation_seconds
                    probe_started=time.perf_counter()
                    delta=self.pager.acquire(int(page)).flatten()
                    observations.append(float((axis*delta).sum().item()))
                    observation_seconds+=time.perf_counter()-probe_started
                    return delta
                full_trace=acquire_vocabulary(self.index,h,self.model.model.norm.weight,
                    self.model.lm_head,observe_vector,seed=20260918+self.calls,budget=self.budget)
                selected=full_trace['pages']; trace=None
            elif self.condition=='risk':
                axis_cpu=axis.cpu().numpy().astype(np.float64)
                axis_bytes=self.width*4
                prior=self.index['vector_prior']@axis_cpu
                cov=self.index['vector_covariance']*float(np.sum(self.index['feature_variance']*axis_cpu**2))
                trace=acquire(prior,cov,margin,observe,'risk',budget=self.budget)
                selected=trace['pages'].tolist()
            else:
                selected=list(range(self.pager.count if self.condition=='all35' else self.budget))
                for page in selected: observe(page)
                trace=None
            acquisition_seconds=time.perf_counter()-controller_started
            # Acquired pages are already in the workspace; do NOT reload them.
            y=self.pager._compute()
            final_hidden=residual+y
            result.logits=self.model.lm_head(self.model.model.norm.forward(final_hidden))
            final_rms=float(torch.sqrt(final_hidden.square().mean()+self.model.model.norm.variance_epsilon).item())
            final_pair=float((result.logits[0,-1,u]-result.logits[0,-1,v]).item())
            predicted=(margin+sum(observations))*base_rms/final_rms
            if abs(base_pair-margin)>1e-4+1e-5*abs(base_pair) or abs(final_pair-predicted)>1e-4+1e-5*abs(final_pair):
                raise ValueError('Physical scalar/readout identity failed')
            scalar_bytes=len(selected)*4; readout_bytes=52
            policy=dict(pages=selected,observations=observations,axis=None if axis_cpu is None else axis_cpu.tolist(),
                base_margin=margin,base_pair=base_pair,final_pair=final_pair,predicted_pair=predicted,
                base_rms=base_rms,final_rms=final_rms,base_ids=[u,v],
                acquisition_seconds=acquisition_seconds,observation_seconds=observation_seconds,
                nonprobe_acquisition_seconds=acquisition_seconds-observation_seconds,
                **(dict(full_trace=full_trace) if self.full_vocabulary else {}),
                trace=None if trace is None else {k:a.tolist() for k,a in trace.items()})
        after,size3=fingerprint(cache,length+1)
        if after!=post: raise ValueError('Readout refinement changed KV')
        self.journal.append(length+1,post)
        consumed=int(args[0][0,0].item())
        next_id=int(result.logits[0,-1].argmax().item())
        self.record(dict(kind='call',call=self.calls,condition=self.condition,prefill=self.pager.prefill,
            consumed_id=consumed,next_id=next_id,base_length=length,end_length=length+1,
            prior_sha=before,prior_after_sha=prior_after,post_sha=post,post_after_sha=after,
            kv_fingerprint_d2h_bytes=size0+size1+size2+size3,
            draft_kv_logical_bytes=kv_bytes(cache),draft_kv_storage_bytes=kv_storage_bytes(cache),
            axis_d2h_bytes=axis_bytes,observation_d2h_bytes=scalar_bytes,
            readout_scalar_d2h_bytes=readout_bytes,
            **(dict(full_vocab_h2d_bytes=0 if full_trace is None else full_trace['h2d_bytes'],
                    full_vocab_d2h_bytes=0 if full_trace is None else full_trace['d2h_bytes']) if self.full_vocabulary else {}),
            inherited_correction_d2h_bytes=(len(self.pager.layers)-1)*(8 if self.pager.prefill else 4),policy=policy,
            wall_seconds=time.perf_counter()-started))
        self.last_cache=cache
        return result

    def after_cache_crop(self,cache):
        length=cache.get_seq_length(); digest,size=fingerprint(cache,length)
        self.journal.crop(length,digest)
        self.record(dict(kind='crop',after_call=self.calls,length=length,sha256=digest,
            kv_fingerprint_d2h_bytes=size,draft_kv_logical_bytes=kv_bytes(cache),
            draft_kv_storage_bytes=kv_storage_bytes(cache)))

    def close(self):
        self.model.forward=self.original
        for hook in self.hooks: hook.remove()
        self.captured.clear(); self.last_cache=None


class VocabularyReadout(RiskReadout):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs,full_vocabulary=True)

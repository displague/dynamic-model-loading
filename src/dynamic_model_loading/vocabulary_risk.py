"""Joint Gaussian page/remaining-field samples scored by full-vocabulary argmax.

This is a plug-in acquisition score, not calibrated acceptance probability or a
safety certificate. Only the external dense verifier commits generated tokens.
"""
import numpy as np
import torch


def common_normals(seed,width,draws=16):
    if type(seed) is not int or seed<0 or width<1 or draws!=16:
        raise ValueError('Invalid frozen Monte Carlo configuration')
    half=np.random.default_rng(seed).standard_normal((2,draws//2,width)).astype(np.float32)
    return np.concatenate((half,-half),axis=1)


def pick_action(rows):
    if not rows: raise ValueError('Missing candidate actions')
    scores=[]
    for row in rows:
        ids=np.asarray(row['ids'])
        if ids.shape!=(2,16) or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any():
            raise ValueError('Invalid sampled full-vocabulary decisions')
        wrong=int(np.sum(ids[0]!=ids[1]))
        if row['mismatches']!=wrong: raise ValueError('Monte Carlo score mismatch')
        scores.append((wrong,row['page']))
    return min(scores)[1]


def audit_trace(trace,index,axis,scalar_observations,call,*,width=1536,pages=35,budget=17,cuda=True):
    """Replay paid-action decisions/counters; neural sampled IDs are observations.

    This does not independently rerun the vocabulary head for every latent draw.
    Source binding and numerical head controls own that execution claim.
    """
    from .debt_analysis import demand
    from .recursive_field import update
    demand(trace['seed']==20260918+call and trace['draws']==16 and trace['full_vocabulary'] is True,
           'Monte Carlo identity changed')
    hidden=np.asarray(trace['hidden']); axis=np.asarray(axis)
    demand(hidden.shape==axis.shape==(width,) and np.isfinite(hidden).all() and np.isfinite(axis).all(),
           'Invalid initial full-vocabulary state')
    mean=np.asarray(index['vector_prior'],dtype=np.float64).copy()
    covariance=np.asarray(index['vector_covariance'],dtype=np.float64).copy()
    noise=np.diag(covariance)*1e-6
    demand(len(trace['steps'])==budget and len(scalar_observations)==budget,'Missing policy steps')
    chosen=[]; action_count=0
    for step,row in enumerate(trace['steps']):
        available=[p for p in range(pages) if p not in chosen]
        demand(row['step']==step and [a['page'] for a in row['actions']]==available,'Sampled action set changed')
        for action in row['actions']:
            ids=np.asarray(action['ids'])
            demand(ids.shape==(2,16) and np.issubdtype(ids.dtype,np.integer) and np.all((0<=ids)&(ids<151936)),
                   'Invalid full-vocabulary sample IDs')
        selected=pick_action(row['actions']); action_count+=len(available)
        demand(selected==row['page'],'Action does not minimize sampled full-vocabulary error')
        observed=np.asarray(row['observed'],dtype=np.float64)
        demand(observed.shape==(width,) and np.isfinite(observed).all(),'Invalid paid vector')
        projected=float(axis@observed)
        demand(abs(projected-scalar_observations[step])<=1e-4+1e-5*abs(projected),
               'Paid full vector differs from physical pair observation')
        mean,covariance=update(mean,covariance,[selected],observed[None],noise)
        demand(np.isfinite(mean).all() and np.isfinite(covariance).all(),'Nonfinite posterior replay')
        chosen.append(selected)
    demand(trace['pages']==chosen,'Selected page receipt changed')
    h2d=(sum(np.asarray(index[k],dtype=np.float32).nbytes for k in
        ('vector_prior','vector_covariance','feature_variance'))+2*16*width*4+action_count*8) if cuda else 0
    d2h=(width*4*(1+budget)+action_count*(1+2*16*8)) if cuda else 0
    demand(trace['h2d_bytes']==h2d and trace['d2h_bytes']==d2h,'Monte Carlo transfer accounting changed')
    return h2d,d2h


@torch.inference_mode()
def acquire(index,hidden,norm_weight,head,observe,*,seed,budget=17):
    """Observe callback returns ONLY the correction vector for one purchased page.

    Correlated page fields use a separable matrix-normal prior. For each possible
    next page, joint draws of its correction and the total missing correction
    preserve their covariance. All vocabulary IDs compete in both readouts.
    Common antithetic draws reduce comparison noise; the prior still may be wrong.
    """
    device=hidden.device; cuda=hidden.is_cuda; width=hidden.numel()
    cpu={k:np.asarray(index[k],dtype=np.float32).copy() for k in
         ('vector_prior','vector_covariance','feature_variance')}
    pages=cpu['vector_prior'].shape[0]
    if (cpu['vector_prior'].shape!=(pages,width) or cpu['vector_covariance'].shape!=(pages,pages)
        or cpu['feature_variance'].shape!=(width,) or not 1<=budget<=pages
        or not all(np.isfinite(a).all() for a in cpu.values())
        or np.any(cpu['feature_variance']<=0) or norm_weight.shape!=(width,)):
        raise ValueError('Invalid full-vocabulary risk state')
    tensors={k:torch.from_numpy(a).to(device) for k,a in cpu.items()}
    mean=tensors['vector_prior']; cov=tensors['vector_covariance']; feature=tensors['feature_variance'].sqrt()
    noise=cov.diag().clone()*1e-6
    normals=common_normals(seed,width)
    z=torch.from_numpy(normals).to(device)*feature
    partial=hidden.flatten().clone()
    original=partial.cpu().numpy().copy()
    h2d=(sum(a.nbytes for a in cpu.values())+normals.nbytes) if cuda else 0
    d2h=original.nbytes if cuda else 0
    selected=[]; trace=[]
    for step in range(budget):
        available=[p for p in range(pages) if p not in selected]
        ids=torch.tensor(available,dtype=torch.int64,device=device)
        h2d+=len(available)*8 if cuda else 0
        total_mean=mean.index_select(0,ids).sum(0)
        total_variance=cov.index_select(0,ids).index_select(1,ids).sum().clamp_min(1e-14)
        cross=cov.index_select(0,ids).sum(0)
        actions=[]
        for page in available:
            sd=cov[page,page].clamp_min(1e-14).sqrt()
            shared=cross[page]/sd
            independent=(total_variance-shared.square()).clamp_min(1e-14).sqrt()
            observed=mean[page]+sd*z[0]
            total=total_mean+shared*z[0]+independent*z[1]
            states=torch.cat((partial+observed,partial+total),dim=0)
            # RMS denominator is a positive scalar per state; norm weights remain.
            logits=head(states*norm_weight)
            valid=bool(torch.isfinite(logits).all().item())
            d2h+=int(cuda)
            if not valid: raise ValueError('Nonfinite Monte Carlo readout')
            sampled=logits.argmax(-1).reshape(2,16).cpu().numpy().copy()
            d2h+=sampled.nbytes if cuda else 0
            actions.append(dict(page=page,ids=sampled.tolist(),mismatches=int(np.sum(sampled[0]!=sampled[1]))))
            del logits,states,observed,total
        page=pick_action(actions)
        delta=observe(page).flatten()
        if delta.shape!=(width,) or delta.dtype!=torch.float32 or delta.device!=device:
            raise ValueError('Invalid paid correction observation')
        actual=delta.cpu().numpy().copy(); d2h+=actual.nbytes if cuda else 0
        if not np.isfinite(actual).all(): raise ValueError('Nonfinite paid observation')
        gain=cov[:,page]/(cov[page,page]+noise[page])
        mean=mean+gain[:,None]*(delta-mean[page])[None,:]
        cov=cov-gain[:,None]*cov[:,page][None,:]
        cov=(cov+cov.T)*.5
        partial=partial+delta
        selected.append(page)
        trace.append(dict(step=step,actions=actions,page=page,observed=actual.tolist()))
    return dict(pages=selected,seed=seed,draws=16,hidden=original.tolist(),steps=trace,
                h2d_bytes=h2d,d2h_bytes=d2h,full_vocabulary=True)

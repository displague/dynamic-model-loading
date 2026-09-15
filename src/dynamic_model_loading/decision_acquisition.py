"""Fixed-budget acquisition by plug-in expected pair-decision risk."""
import math
import numpy as np
from .recursive_field import fit_vector,update,observed_sites


def cdf(x):
    values=np.asarray(x,dtype=np.float64)
    return np.array([.5*math.erfc(-float(t)/math.sqrt(2)) for t in values.ravel()]).reshape(values.shape)


def expected_risk(margin,mean,covariance,available,page,quadrature=32):
    """Risk that actual partial pair sign after ONE load differs from full sign.

    Integrate at both sign-change breakpoints, not across a discontinuity.
    This is a plug-in Gaussian risk, not a calibrated safety certificate.
    """
    total_mean=margin+float(mean[available].sum())
    total_variance=max(float(covariance[np.ix_(available,available)].sum()),1e-14)
    variance=max(float(covariance[page,page]),1e-14)
    sd=math.sqrt(variance)
    cross=float(covariance[available,page].sum())
    conditional_sd=math.sqrt(max(total_variance-cross*cross/variance,1e-14))
    cuts=[-9.,9.]
    cuts.append(float(np.clip((-margin-mean[page])/sd,-9,9)))
    if abs(cross)>1e-14:
        cuts.append(float(np.clip(-total_mean*sd/cross,-9,9)))
    cuts=sorted(set(cuts))
    nodes,weights=quadrature_rule(quadrature)
    risk=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        t=(hi-lo)*nodes/2+(hi+lo)/2
        sign=np.where(margin+mean[page]+sd*t>=0,1.,-1.)
        conditional=total_mean+(cross/sd)*t
        integrand=cdf(-sign*conditional/conditional_sd)*np.exp(-t*t/2)/math.sqrt(2*math.pi)
        risk+=(hi-lo)/2*float(weights@integrand)
    return float(np.clip(risk,0,1))


from functools import lru_cache
@lru_cache(maxsize=4)
def quadrature_rule(order):
    return np.polynomial.legendre.leggauss(order)


def acquire(prior,covariance,margin,observe,policy,budget=17):
    """Controller receives a callback for ONE purchased measurement, no shadow."""
    mean=prior.copy(); cov=covariance.copy(); noise=np.diag(covariance)*1e-6
    selected=[]; values=[]; scores=[]; predictions=[]
    for _ in range(budget):
        available=[p for p in range(len(prior)) if p not in selected]
        if policy=='fixed':
            row={p:float(p) for p in available}
        elif policy=='contribution':
            row={p:-abs(float(mean[p])) for p in available}
        elif policy=='risk':
            row={p:expected_risk(margin,mean,cov,available,p) for p in available}
        else: raise ValueError('Unknown acquisition policy')
        page=min(available,key=lambda p:(row[p],p))
        value=float(observe(page))
        if not math.isfinite(value): raise ValueError('Nonfinite paid observation')
        scores.append(np.array([row.get(p,-1.) for p in range(len(prior))]))
        selected.append(page); values.append(value); margin+=value
        mean,cov=update(mean,cov,[page],np.array([value]),noise)
        remaining=[p for p in available if p!=page]
        predictions.append(margin+float(mean[remaining].sum()))
    return dict(pages=np.array(selected,dtype=np.int64),observations=np.array(values),
        scores=np.stack(scores),posterior_total=np.array(predictions),partial=np.array(margin))


def acquisition_predictions(inputs):
    vm,vc,d,_=fit_vector(inputs['delta'])
    raw=dict(vector_prior=vm,vector_covariance=vc,feature_variance=d)
    commutation=[]
    conditions={name:[] for name in ('fixed','contribution','risk','independent_risk')}
    numerical=[]
    for i in range(128):
        axis=inputs['axis'][i]; prior=vm@axis
        covariance=vc*float(np.sum(d*axis**2))
        sites=observed_sites(i%16)
        vector,pcov=update(vm,vc,sites,inputs['delta'][i,sites],np.diag(vc)*1e-6)
        scalar,scov=update(prior,covariance,sites,inputs['z'][i,sites],np.diag(covariance)*1e-6)
        commutation.append([float(np.max(abs(vector@axis-scalar))),
            float(np.max(abs(pcov*float(np.sum(d*axis**2))-scov)))])
        for name in conditions:
            cov=np.diag(np.diag(covariance)) if name=='independent_risk' else covariance
            policy='risk' if name=='independent_risk' else name
            result=acquire(prior,cov,float(inputs['margin'][i]),lambda p:inputs['z'][i,p],policy)
            conditions[name].append(result)
            # Higher-order check of EVERY available action along diagnostic risk paths.
            if i>=96 and policy=='risk':
                m=prior.copy(); c=cov.copy(); remaining=list(range(35))
                margin=float(inputs['margin'][i]); noise=np.diag(cov)*1e-6
                for j,page in enumerate(result['pages']):
                    numerical.append(max(abs(result['scores'][j,p]-
                        expected_risk(margin,m,c,remaining,p,128)) for p in remaining))
                    value=float(result['observations'][j]); margin+=value
                    m,c=update(m,c,[int(page)],np.array([value]),noise)
                    remaining.remove(int(page))
    for name,rows in conditions.items():
        for key in rows[0]: raw[name+'_'+key]=np.stack([r[key] for r in rows])
    raw['commutation_errors']=np.array(commutation)
    raw['quadrature_errors']=np.array(numerical)
    return raw


def acquisition_summary(inputs,raw):
    truth=inputs['margin']+inputs['z'].sum(1)
    metrics={}
    for label,indices in [('fit',np.arange(96)),('diagnostic',np.arange(96,128)),
                          ('diagnostic_0',np.arange(96,112)),('diagnostic_1',np.arange(112,128))]:
        block={}
        base=inputs['margin'][indices]>=0; gold=truth[indices]>=0
        for name in ('fixed','contribution','risk','independent_risk'):
            actual=raw[name+'_partial'][indices]>=0
            predicted=raw[name+'_posterior_total'][indices,-1]>=0
            block[name]=dict(pair_errors=int(np.sum(actual!=gold)),base_pair_errors=int(np.sum(base!=gold)),
                repairs=int(np.sum((base!=gold)&(actual==gold))),
                new_errors=int(np.sum((base==gold)&(actual!=gold))),
                posterior_forecast_errors=int(np.sum(predicted!=gold)),positions=len(indices))
        metrics[label]=block
    d=metrics['diagnostic']
    def candidate(name):
        return (d[name]['pair_errors']<=d['fixed']['pair_errors']-2 and
            all(metrics[l][name]['pair_errors']<=metrics[l]['fixed']['pair_errors']+1
                for l in ('diagnostic_0','diagnostic_1')))
    numerical=float(raw['quadrature_errors'].max())
    commute=float(raw['commutation_errors'].max())
    gates=dict(Hcommutation=commute<=1e-10,Hnumerical=numerical<=1e-4,
        Hrisk=candidate('risk') and d['risk']['pair_errors']<=min(d['contribution']['pair_errors'],d['independent_risk']['pair_errors']),
        Hcontribution=candidate('contribution'))
    nominee='risk' if gates['Hrisk'] else 'contribution' if gates['Hcontribution'] else None
    if not gates['Hcommutation'] or not gates['Hnumerical']: nominee=None
    return dict(metrics=metrics,gates=gates,nominated_policy=nominee,
        decision='eligible_for_short_physical_policy' if nominee else 'stop_this_decision_policy',
        max_commutation_error=commute,max_quadrature_error=numerical,
        new_inference=False,cuda_execution=False,new_h2d_bytes=0,accepted_tokens_measured=False,
        reused_development_data=True,pages_per_position=17,proposed_page_payload_bytes=17*884736,
        proposed_scalar_readback_bytes=17*4,proposed_full_vector_readback_bytes=17*1536*4,
        current_axis_bytes=1536*4,full_argmax_certified=False,
        risk_is_calibrated=False,recurrence_used=False)

"""Gaussian conditioning on physical page effects; missing sites are not zero."""
import numpy as np


def fit_field(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or min(values.shape)<2 or not np.isfinite(values).all():
        raise ValueError('Finite training matrix required')
    mean=values.mean(0)
    sample=np.cov(values,rowvar=False)
    variance=np.maximum(np.diag(sample),max(float(np.median(np.diag(sample)))*1e-8,1e-12))
    covariance=.5*sample+.5*np.diag(variance)
    return mean,covariance


def condition(mean,covariance,sites,observations):
    mean=np.asarray(mean,dtype=np.float64)
    covariance=np.asarray(covariance,dtype=np.float64)
    sites=np.asarray(sites,dtype=np.int64)
    observations=np.asarray(observations,dtype=np.float64)
    n=len(mean)
    if (covariance.shape!=(n,n) or sites.ndim!=1 or observations.shape!=sites.shape or
        len(set(sites.tolist()))!=len(sites) or np.any(sites<0) or np.any(sites>=n) or
        not all(np.isfinite(a).all() for a in (mean,covariance,observations))):
        raise ValueError('Invalid observed subset')
    if not len(sites):
        return mean.copy(),covariance.copy()
    noise=np.diag(covariance)[sites]*1e-6
    system=covariance[np.ix_(sites,sites)]+np.diag(noise)
    cross=covariance[:,sites]
    gain=np.linalg.solve(system,cross.T).T
    post=covariance-gain@cross.T
    return mean+gain@(observations-mean[sites]),(post+post.T)/2


def probe_order(covariance,count):
    """Fit-only reduction in OTHER sites' variance, not decision EVSI."""
    current=np.asarray(covariance,dtype=np.float64).copy()
    noise=np.diag(current)*1e-6
    selected=[]
    for _ in range(count):
        available=[i for i in range(len(current)) if i not in selected]
        scores={s:sum(current[j,s]**2 for j in available if j!=s)/(current[s,s]+noise[s]) for s in available}
        s=max(available,key=lambda i:(scores[i],-i))
        selected.append(s)
        current-=np.outer(current[:,s],current[s,:])/(current[s,s]+noise[s])
    return selected


def covariance_controls(covariance):
    sd=np.sqrt(np.diag(covariance))
    corr=covariance/sd[:,None]/sd[None,:]
    permutation=np.random.default_rng(2701).permutation(len(sd))
    return dict(independent=np.diag(sd**2),field=covariance,
        shuffled=corr[np.ix_(permutation,permutation)]*sd[:,None]*sd[None,:])


def spatial_predictions(values):
    mean,covariance=fit_field(values[:96])
    controls=covariance_controls(covariance)
    order=probe_order(covariance,4)
    predictions={}
    for budget in (1,4):
        for name,cov in controls.items():
            pair=[condition(mean,cov,order[:budget],v[order[:budget]]) for v in values]
            predictions[f'{name}_{budget}_mean']=np.stack([p[0] for p in pair])
            predictions[f'{name}_{budget}_variance']=np.stack([np.diag(p[1]) for p in pair])
    return dict(mean=mean,covariance=covariance,order=np.array(order,dtype=np.int64),**predictions)


def spatial_summary(values,raw):
    result={}
    for label,indices in [('fit',np.arange(96)),('diagnostic',np.arange(96,128)),
                          ('diagnostic_0',np.arange(96,112)),('diagnostic_1',np.arange(112,128))]:
        metrics={}
        for budget in (1,4):
            unseen=[p for p in range(35) if p not in raw['order'][:budget]]
            for name in ('independent','field','shuffled'):
                key=f'{name}_{budget}'
                error=(raw[key+'_mean']-values)[np.ix_(indices,unseen)]
                variance=raw[key+'_variance'][np.ix_(indices,unseen)]
                metrics[key]=dict(mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),
                    coverage90=float(np.mean(abs(error)<=1.6448536269514722*np.sqrt(np.maximum(variance,0)))),
                    unobserved_predictions=int(error.size))
        result[label]=metrics
    ratios={label:result[label]['field_1']['mse']/result[label]['independent_1']['mse'] for label in result}
    passed=(ratios['diagnostic']<=.9 and max(ratios['diagnostic_0'],ratios['diagnostic_1'])<=1.05 and
            result['diagnostic']['field_1']['mse']<result['diagnostic']['shuffled_1']['mse'])
    return dict(metrics=result,field_to_independent_mse=ratios,probe_order=raw['order'].tolist(),
        gates=dict(Hspatial=bool(passed)),decision='eligible_for_acquisition_screen' if passed else 'stop_this_static_field',
        new_inference=False,new_h2d_bytes=0,cuda_execution=False,accepted_tokens_measured=False,
        reused_development_data=True,primary_budget_pages=1,page_payload_bytes=884736,
        gaussian_bands_are_calibration_guarantees=False)

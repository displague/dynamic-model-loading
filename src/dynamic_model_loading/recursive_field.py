"""Separable matrix-normal page filter with a changing readout operator."""
import numpy as np
from .page_field import fit_field


def update(mean,covariance,sites,observed,noise):
    sites=np.asarray(sites,dtype=np.int64)
    observed=np.asarray(observed,dtype=np.float64)
    if (mean.ndim not in (1,2) or observed.shape!=(len(sites),*mean.shape[1:]) or
        sites.ndim!=1 or len(set(sites.tolist()))!=len(sites) or
        np.any(sites<0) or np.any(sites>=len(mean)) or not np.isfinite(observed).all()):
        raise ValueError('Invalid causal observations')
    system=covariance[np.ix_(sites,sites)]+np.diag(noise[sites])
    cross=covariance[:,sites]
    gain=np.linalg.solve(system,cross.T).T
    posterior=covariance-gain@cross.T
    return mean+gain@(observed-mean[sites]),(posterior+posterior.T)/2


def ar_coefficient(centered):
    # Documents never share transition pairs; fit uses six 16-position trajectories.
    sequence=centered.reshape(6,16,*centered.shape[1:])
    previous,current=sequence[:,:-1],sequence[:,1:]
    return float(np.clip(np.sum(previous*current)/max(float(np.sum(previous*previous)),1e-12),0,.95))


def fit_vector(delta):
    fit=delta[:96]
    mean=fit.mean(0)
    centered=fit-mean
    feature_variance=np.maximum(np.mean(centered**2,axis=(0,1)),1e-12)
    whitened=centered/np.sqrt(feature_variance)
    sample=np.einsum('tih,tjh->ij',whitened,whitened)/((len(fit)-1)*fit.shape[2])
    covariance=.5*sample+.5*np.diag(np.maximum(np.diag(sample),1e-12))
    return mean,covariance,feature_variance,ar_coefficient(whitened)


def observed_sites(position):
    return np.array([(4*position+j)%35 for j in range(4)],dtype=np.int64)


def recursive_predictions(inputs):
    delta,axis,z=inputs['delta'],inputs['axis'],inputs['z']
    vm,vc,feature_var,vrho=fit_vector(delta)
    sm,sc=fit_field(z[:96]); srho=ar_coefficient(z[:96]-sm)
    models={'static_scalar':(sm,sc,0.),'recursive_scalar':(sm,sc,srho),
            'static_vector':(vm,vc,0.),'recursive_vector':(vm,vc,vrho)}
    raw=dict(vector_prior=vm,vector_covariance=vc,feature_variance=feature_var,
        scalar_prior=sm,scalar_covariance=sc,rho=np.array([srho,vrho]),
        observed=np.stack([observed_sites(i%16) for i in range(128)]))
    for name,(prior,cov,rho) in models.items():
        means=[]; variances=[]
        state=prior.copy(); uncertainty=cov.copy()
        for i in range(128):
            if i%16==0:
                state=prior.copy(); uncertainty=cov.copy()
            else:
                state=prior+rho*(state-prior)
                uncertainty=rho*rho*uncertainty+(1-rho*rho)*cov
            sites=observed_sites(i%16)
            measurements=delta[i,sites] if name.endswith('vector') else z[i,sites]
            state,uncertainty=update(state,uncertainty,sites,measurements,np.diag(cov)*1e-6)
            if name.endswith('vector'):
                means.append(state@axis[i])
                variances.append(np.diag(uncertainty)*float(np.sum(feature_var*axis[i]**2)))
            else:
                means.append(state.copy()); variances.append(np.diag(uncertainty))
        raw[name+'_mean']=np.stack(means)
        raw[name+'_variance']=np.stack(variances)
    return raw


def recursive_summary(inputs,raw):
    metrics={}
    for label,indices in [('fit',range(96)),('diagnostic',range(96,128)),
                           ('diagnostic_0',range(96,112)),('diagnostic_1',range(112,128))]:
        block={}
        for name in ('static_scalar','recursive_scalar','static_vector','recursive_vector'):
            errors=[]; variances=[]
            for i in indices:
                unseen=[s for s in range(35) if s not in raw['observed'][i]]
                errors.extend((raw[name+'_mean'][i]-inputs['z'][i])[unseen])
                variances.extend(raw[name+'_variance'][i,unseen])
            error=np.array(errors); variance=np.array(variances)
            block[name]=dict(mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),
                coverage90=float(np.mean(abs(error)<=1.6448536269514722*np.sqrt(np.maximum(variance,0)))),
                unobserved_predictions=len(error))
        metrics[label]=block
    def ratios(a,b):
        return {label:m[a]['mse']/m[b]['mse'] for label,m in metrics.items()}
    geometry=ratios('static_vector','static_scalar')
    temporal=ratios('recursive_vector','static_vector')
    def passes(r): return r['diagnostic']<=.9 and max(r['diagnostic_0'],r['diagnostic_1'])<=1.05
    gates=dict(Hgeometry=passes(geometry),Htemporal=passes(temporal))
    return dict(metrics=metrics,geometry_mse_ratios=geometry,temporal_mse_ratios=temporal,
        rho_scalar=float(raw['rho'][0]),rho_vector=float(raw['rho'][1]),gates=gates,
        decision='eligible_for_decision_acquisition_screen' if any(gates.values()) else 'stop_this_vector_filter',
        new_inference=False,cuda_execution=False,new_h2d_bytes=0,accepted_tokens_measured=False,
        reused_development_data=True,observation_pages_per_position=4,
        hypothetical_page_payload_per_position=4*884736,vector_observation_d2h_per_position=4*1536*4,
        vector_state_bytes=35*1536*8,vector_page_covariance_bytes=35*35*8,
        diagonal_feature_covariance_bytes=1536*8,full_argmax_certified=False)

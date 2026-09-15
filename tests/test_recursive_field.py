import numpy as np
import pytest
from dynamic_model_loading.recursive_field import update,ar_coefficient,recursive_predictions,observed_sites


def test_matrix_update_matches_dense_kronecker():
    m=np.array([[1.,2.],[3.,4.]])
    c=np.array([[2.,.7],[.7,1.]])
    d=np.diag([.3,2.]); noise=np.diag(c)*1e-6
    actual,p=update(m,c,[0],np.array([[5.,6.]]),noise)
    full=np.kron(c,d); indices=[0,1]
    gain=full[:,indices]@np.linalg.inv(full[np.ix_(indices,indices)]+noise[0]*d)
    expected=m.ravel()+gain@(np.array([5.,6.])-m[0])
    np.testing.assert_allclose(actual.ravel(),expected,atol=1e-12)
    np.testing.assert_allclose(np.kron(p,d),full-gain@full[indices,:],atol=1e-12)


def test_missing_state_is_not_zero():
    m,p=update(np.array([[2.,3.],[4.,5.]]),np.eye(2),[],np.empty((0,2)),np.ones(2)*1e-6)
    np.testing.assert_array_equal(m,[[2,3],[4,5]])
    np.testing.assert_array_equal(p,np.eye(2))


def test_ar_does_not_link_documents():
    a=np.zeros((6,16,2)); a[:,-1]=100
    assert ar_coefficient(a.reshape(96,2))==0


def toy_inputs():
    rng=np.random.default_rng(28)
    delta=rng.normal(size=(128,35,3)); axis=rng.normal(size=(128,3))
    return dict(delta=delta,axis=axis,z=np.einsum('tih,th->ti',delta,axis))


def test_no_unobserved_data_leakage():
    inputs=toy_inputs(); a=recursive_predictions(inputs)
    changed={k:v.copy() for k,v in inputs.items()}
    for t in range(96,128):
        unseen=[s for s in range(35) if s not in observed_sites(t%16)]
        changed['delta'][t,unseen]+=1e4; changed['z'][t,unseen]+=1e4
    b=recursive_predictions(changed)
    for key in a: np.testing.assert_array_equal(a[key],b[key])


def test_document_reset_and_future_isolation():
    inputs=toy_inputs(); a=recursive_predictions(inputs)
    changed={k:v.copy() for k,v in inputs.items()}
    changed['delta'][105:112]*=100; changed['z'][105:112]*=100
    b=recursive_predictions(changed)
    for name in ('static_scalar','recursive_scalar','static_vector','recursive_vector'):
        np.testing.assert_array_equal(a[name+'_mean'][:105],b[name+'_mean'][:105])
        np.testing.assert_array_equal(a[name+'_mean'][112:],b[name+'_mean'][112:])


def test_current_axis_not_last_axis():
    inputs=toy_inputs(); a=recursive_predictions(inputs)
    changed={k:v.copy() for k,v in inputs.items()}; changed['axis'][100]*=-1
    b=recursive_predictions(changed)
    np.testing.assert_allclose(a['recursive_vector_mean'][100],-b['recursive_vector_mean'][100])
    np.testing.assert_allclose(a['recursive_vector_variance'],b['recursive_vector_variance'])


@pytest.mark.parametrize('sites,obs',[([0,0],[[1.,2.],[3.,4.]]),([3],[[1.,2.]]),([0],[[np.nan,2.]])])
def test_bad_matrix_observations(sites,obs):
    with pytest.raises(ValueError): update(np.ones((2,2)),np.eye(2),sites,np.array(obs),np.ones(2))

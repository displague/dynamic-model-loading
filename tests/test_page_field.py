import numpy as np
import pytest
from dynamic_model_loading.page_field import condition,fit_field,probe_order,covariance_controls,spatial_predictions


def test_condition_matches_information_form():
    mean=np.array([1.,2.,3.]); cov=np.array([[2.,.4,.2],[.4,1.,.1],[.2,.1,3.]])
    actual,variance=condition(mean,cov,[1],[4.])
    precision=np.linalg.inv(cov); eta=precision@mean
    precision[1,1]+=1e6; eta[1]+=4e6
    expected=np.linalg.inv(precision)
    np.testing.assert_allclose(variance,expected,atol=1e-12)
    np.testing.assert_allclose(actual,expected@eta,atol=1e-9)


def test_missing_not_zero_and_signed_transfer():
    mean=np.array([2.,3.]); cov=np.array([[1.,-.5],[-.5,1.]])
    np.testing.assert_array_equal(condition(mean,cov,[],[])[0],mean)
    assert condition(mean,cov,[0],[4.])[0][1]<3


@pytest.mark.parametrize('sites,obs',[([0,0],[1,2]),([2],[1]),([-1],[1]),([0],[]),([0],[float('nan')])])
def test_invalid_observations(sites,obs):
    with pytest.raises(ValueError): condition(np.zeros(2),np.eye(2),sites,obs)


def test_independent_never_updates_unseen():
    m,c=condition(np.array([1.,2.]),np.eye(2),[0],[20.])
    assert m[1]==2 and c[1,1]==1


def test_shrinkage_spd_and_shuffle_marginals():
    x=np.arange(100.).reshape(20,5)
    mean,cov=fit_field(x)
    assert np.linalg.eigvalsh(cov).min()>0
    for value in covariance_controls(cov).values():
        np.testing.assert_allclose(np.diag(value),np.diag(cov))
        assert np.linalg.eigvalsh(value).min()>0


def test_probe_order_values_other_sites_not_own_variance():
    cov=np.array([[100.,0.,0.],[0.,1.,.8],[0.,.8,1.]])
    assert probe_order(cov,2)==[1,0]


def test_no_diagnostic_fit_or_shadow_leakage():
    x=np.random.default_rng(0).normal(size=(128,35))
    a=spatial_predictions(x)
    changed=x.copy(); unseen=[p for p in range(35) if p not in a['order']]
    changed[96:,unseen]+=1e4
    b=spatial_predictions(changed)
    for key in a: np.testing.assert_array_equal(a[key],b[key])

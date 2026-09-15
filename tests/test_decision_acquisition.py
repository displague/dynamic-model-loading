import numpy as np
import pytest
from dynamic_model_loading.decision_acquisition import acquire,cdf,expected_risk


def test_normal_cdf_symmetry():
    x=np.array([-3.,0.,3.])
    np.testing.assert_allclose(cdf(x)+cdf(-x),1)
    assert cdf(np.array(0.))==.5


def test_last_page_removes_pair_risk():
    assert expected_risk(.3,np.array([.1]),np.array([[1.]]),[0],0)<1e-10


def test_independent_zero_mean_pair_disagreement():
    # Sign(X) vs sign(X+Y), independent equal-variance normals, differs 1/4.
    value=expected_risk(0.,np.zeros(2),np.eye(2),[0,1],0)
    assert abs(value-.25)<1e-10


def test_risk_sign_reversal_symmetry_and_quadrature():
    mean=np.array([.3,-.8,.1]); cov=np.array([[2.,.6,.2],[.6,1.,-.1],[.2,-.1,.5]])
    for page in range(3):
        a=expected_risk(.4,mean,cov,[0,1,2],page)
        b=expected_risk(-.4,-mean,cov,[0,1,2],page)
        assert abs(a-b)<1e-10
        assert abs(a-expected_risk(.4,mean,cov,[0,1,2],page,128))<1e-6


@pytest.mark.parametrize('policy',['fixed','risk','contribution'])
def test_purchases_only_unique_callback_observations(policy):
    called=[]
    def observe(page): called.append(page); return .1*page
    result=acquire(np.arange(5.)*.1,np.eye(5),.2,observe,policy,budget=3)
    assert called==result['pages'].tolist() and len(set(called))==3
    assert abs(float(result['partial'])-(.2+sum(.1*p for p in called)))<1e-12


def test_first_choice_cannot_use_future_observation():
    a=acquire(np.array([1.,.3,.2]),np.eye(3),.1,lambda p:100.,'risk',budget=1)
    b=acquire(np.array([1.,.3,.2]),np.eye(3),.1,lambda p:-100.,'risk',budget=1)
    np.testing.assert_array_equal(a['pages'],b['pages'])
    np.testing.assert_array_equal(a['scores'],b['scores'])


def test_invalid_policy_and_nonfinite_observation():
    with pytest.raises(ValueError): acquire(np.ones(3),np.eye(3),0,lambda p:1,'bad')
    with pytest.raises(ValueError): acquire(np.ones(3),np.eye(3),0,lambda p:np.nan,'fixed')

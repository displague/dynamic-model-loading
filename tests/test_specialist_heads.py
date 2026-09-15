import copy
import json
from pathlib import Path
import numpy as np
import pytest
from dynamic_model_loading.specialist_heads import features, fit_bank, evaluate, summarize, projection, fit_head
from dynamic_model_loading.specialist_screen import validate_config, validate_documents
from dynamic_model_loading.specialist_analysis import numerical, audit_resources

ROOT = Path(__file__).resolve().parents[1]


def data():
    rng = np.random.default_rng(40)
    h = rng.normal(size=(144, 20))
    draft = rng.normal(size=(144, 13))
    target = draft + rng.normal(size=(144, 13))
    domain = np.repeat([0,1,2,0,1,2], [32,32,32,16,16,16])
    fit = np.arange(144) < 96
    return h, target, draft, domain, fit


def test_fit_has_no_diagnostic_target_leakage():
    h,t,d,domain,fit = data()
    bank = fit_bank(h,t,d,domain,fit)
    t[~fit] += 1000
    assert np.array_equal(bank,fit_bank(h,t,d,domain,fit))


def test_constant_row_logit_offsets_do_not_change_fit():
    h,t,d,domain,fit = data()
    a = fit_bank(h,t,d,domain,fit)
    b = fit_bank(h,t+100,d,domain,fit)
    np.testing.assert_allclose(a,b,atol=1e-13)


def test_zero_residual_zero_head():
    h,t,d,domain,fit = data()
    bank = fit_bank(h,d,d,domain,fit)
    assert not bank.any()
    ids = evaluate(h,d,d,domain,bank)
    assert np.all(ids == ids[:, :1])


def test_projection_frozen_and_features_rms_invariant():
    h,*_ = data()
    assert np.array_equal(projection(20),projection(20))
    np.testing.assert_allclose(features(h),features(h*3),atol=1e-10)
    assert np.all(features(h)[:,-1] == 1)


def test_ridge_satisfies_normal_equations():
    h,t,d,*_ = data()
    x = features(h)
    bank = fit_head(x,t,d)
    residual=t-d
    residual-=residual.mean(1,keepdims=True)
    np.testing.assert_allclose((x.T@x+32*np.eye(17))@bank,x.T@residual,atol=1e-12)


def test_general_and_matching_and_wrong_domain_are_distinct():
    h,t,d,domain,fit = data()
    bank=np.zeros((4,17,13))
    for i in range(4):
        bank[i,-1,i]=100
    ids=evaluate(h,t,d,domain,bank)
    assert np.all(ids[:,2] == 0)
    assert np.array_equal(ids[:,3],domain+1)
    assert np.array_equal(ids[:,4],(domain+1)%3+1)


def test_full_vocabulary_outside_base_top_two_can_win():
    h,t,d,domain,fit=data()
    d[:]=0
    d[:,0]=2
    d[:,1]=1
    bank=np.zeros((4,17,13))
    bank[:,-1,12]=3
    assert np.all(evaluate(h,t,d,domain,bank)[:,2:] == 12)


def test_nomination_requires_domain_not_general_improvement():
    _,_,_,domain,fit=data()
    ids=np.zeros((144,5),dtype=np.int64)
    ids[~fit,1:]=1
    diag=np.flatnonzero(~fit)
    ids[diag[:4],2]=0
    result=summarize(ids,domain,fit)
    assert result['gates'] == {'Hgeneral':True,'Hspecialization':False}
    assert result['decision'].startswith('stop')


def test_specialization_requires_three_and_no_domain_regression():
    _,_,_,domain,fit=data()
    ids=np.ones((144,5),dtype=np.int64)
    ids[:,0]=0
    diag=np.flatnonzero(~fit)
    ids[diag[[0,16,32]],3]=0
    assert summarize(ids,domain,fit)['gates']['Hspecialization']
    ids[diag[[16,17]],2]=0
    assert not summarize(ids,domain,fit)['gates']['Hspecialization']


@pytest.mark.parametrize('kind',['nan','no_fit','all_fit','missing_domain'])
def test_invalid_fit_rejected(kind):
    h,t,d,domain,fit=data()
    if kind=='nan': h[0,0]=np.nan
    if kind=='no_fit': fit[:]=False
    if kind=='all_fit': fit[:]=True
    if kind=='missing_domain': fit[domain==1]=False
    with pytest.raises(ValueError): fit_bank(h,t,d,domain,fit)


def test_frozen_documents_and_configuration():
    cfg=json.loads((ROOT/'configs/specialist-screen.json').read_text())
    docs=json.loads((ROOT/cfg['documents']).read_text())
    validate_config(cfg)
    validate_documents(docs)
    cfg['rank']=32
    with pytest.raises(ValueError): validate_config(cfg)
    docs[12]['split']='fit'
    with pytest.raises(ValueError): validate_documents(docs)


def test_resource_summary_cannot_lie_about_peak_or_cap():
    rows=[dict(monotonic=1.,gpu_used=2**30,host_available=3*2**30,process={'rss':1000})]
    receipt=dict(samples=1,peak_gpu_used=2**30,min_host_available=3*2**30,peak_rss=1000,passed=True,error=None)
    audit_resources(rows,receipt)
    receipt['peak_rss']=999
    with pytest.raises(ValueError): audit_resources(rows,receipt)
    rows[0]['host_available']=100
    with pytest.raises(ValueError): audit_resources(rows,receipt)


def test_numerical_controls_check_argmax_and_finite():
    x=np.eye(3)
    assert numerical(x,x)['argmax_identical']
    for bad in (np.roll(x,1,axis=1), x*np.nan, x[:2]):
        with pytest.raises(ValueError): numerical(x,bad)


def test_output_only_head_does_not_mutate_inputs():
    h,t,d,domain,fit=data()
    saved=[a.copy() for a in (h,t,d,domain,fit)]
    bank=fit_bank(h,t,d,domain,fit)
    evaluate(h,t,d,domain,bank)
    assert all(np.array_equal(a,b) for a,b in zip((h,t,d,domain,fit),saved))


def test_blas_thread_count_is_enforced_and_library_bound():
    from dynamic_model_loading.numpy_backend import set_blas_threads
    receipt=set_blas_threads(4)
    assert receipt['threads']==4 and len(receipt['sha256'])==64
    assert set_blas_threads(4)==receipt
    with pytest.raises(ValueError): set_blas_threads(0)


@pytest.mark.parametrize('status',['error','timeout','interrupted'])
def test_cli_failure_is_nonzero(status,monkeypatch):
    from dynamic_model_loading import specialist_screen as screen
    monkeypatch.setattr('sys.argv',['specialist_screen','--output','unused'])
    monkeypatch.setattr(screen,'supervise',lambda *a,**k:{'status':status})
    with pytest.raises(SystemExit) as exc: screen.main()
    assert exc.value.code==1


@pytest.mark.parametrize('hidden,intermediate,layers,heads,parameters,buffers',[
    (896,4864,24,14,1976131072,256),
    (1536,8960,28,12,6174857216,512),
])
def test_meta_architecture_allocation(hidden,intermediate,layers,heads,parameters,buffers):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg=Qwen2Config(vocab_size=151936,hidden_size=hidden,intermediate_size=intermediate,
        num_hidden_layers=layers,num_attention_heads=heads,num_key_value_heads=2,tie_word_embeddings=True)
    with torch.device('meta'):
        model=Qwen2ForCausalLM(cfg)
    assert sum(p.numel()*p.element_size() for p in model.parameters())==parameters
    assert sum(b.numel()*b.element_size() for b in model.buffers())==buffers

import pytest
import torch
import numpy as np
from types import SimpleNamespace
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from dynamic_model_loading.adapters import FFNView
from dynamic_model_loading import causal_evidence as ce
from dynamic_model_loading.causal_evidence_cost import rounded_cost,action_cost
from dynamic_model_loading.causal_evidence_analysis import generation_alignment,verify_trace
from dynamic_model_loading import causal_evidence_study as study


def controller(mode='partial',extra=2,models=None):
    return ce.Controller(torch.arange(8,0,-1,dtype=torch.float32),torch.tensor([True,True,False,False,False,False,False,False]),
                         ce.projections(16,8,1729,'cpu'),models,mode,extra,initial=6)


def test_observation_cannot_expose_omitted_groups():
    c=controller();initial=c.begin(torch.arange(16,dtype=torch.float32));first_norm=torch.tensor(2.)
    observed=torch.arange(8,dtype=torch.float32);changed=observed.clone();changed[~initial]=1e20
    assert torch.equal(c.evidence(observed,first_norm),c.evidence(changed,first_norm))
    before=c.history.clone();c.observe(changed,initial)
    assert torch.equal(c.history[~initial],before[~initial])
    assert torch.equal(c.age[~initial],torch.ones_like(c.age[~initial]))
    assert (c.age[initial]==0).all()


def test_matching_one_shot_and_repair_cardinality_and_cold_bytes():
    for mode in ('one_shot','predetermined','partial'):
        c=controller(mode);initial=c.begin(torch.ones(16));extra=c.acquire(torch.ones(8),torch.tensor(1.))
        assert not (initial&extra).any()
        assert (initial|extra).sum()==8 and initial[c.hot].all()
        receipt={'initial':initial.reshape(1,1,8),'additions':extra.reshape(1,1,8),'fallback':torch.zeros(1,1,dtype=torch.bool)}
        assert sum(ce.traffic(receipt,c.hot.reshape(1,8).numpy(),10)[k] for k in ('initial_cold_bytes','repair_cold_bytes'))==60


class TinyFFN(nn.Module):
    def __init__(self):
        super().__init__();self.gate_proj=nn.Linear(16,64,bias=False);self.up_proj=nn.Linear(16,64,bias=False);self.down_proj=nn.Linear(64,16,bias=False)
    def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


def test_full_addition_reconstructs_and_hooks_cleanup():
    torch.manual_seed(7);source=TinyFFN();mlp=FFNView(source,('gate_proj','up_proj','down_proj'),F.silu)
    x=torch.randn(1,1,16);reference=source(x);probe=ce.Probe(mlp,controller('full'),collect=True)
    with ce.apply([mlp],[probe]):value=source(x)
    assert torch.allclose(reference,value,rtol=1e-5,atol=1e-6)
    data=ce.payload([probe]);training=ce.payload([probe],True)
    assert data['initial'].shape==(1,1,8) and training['labels'].shape==(1,1,8)
    assert not source._forward_hooks and not source._forward_pre_hooks and not source.down_proj._forward_pre_hooks
    with pytest.raises(RuntimeError,match='fixture'):
        with ce.apply([mlp],[ce.Probe(mlp,controller())]):raise RuntimeError('fixture')
    assert not source._forward_hooks and not source.down_proj._forward_pre_hooks


def test_calibration_fit_is_finite_and_not_an_observation_update():
    torch.manual_seed(7);x=torch.randn(80,5);weights=torch.randn(5,3);y=x@weights+2
    model=ce.fit(x,y,ridge=.00001)
    assert torch.allclose(ce.predict(x,model),y,rtol=.002,atol=.002)
    bad=x.clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):ce.fit(bad,y)


def constant_model(size,outputs,value):
    return dict(mean=torch.zeros(size),scale=torch.ones(size),weight=torch.zeros(size,outputs),intercept=torch.full((outputs,),value))


def test_detector_is_separate_from_acquisition_and_fallback_is_bounded():
    models={'input':constant_model(16,8,1.),'base':constant_model(33,8,1.),
            'repair':constant_model(43,8,1.),'detector':constant_model(43,1,.06)}
    c=controller('fallback',1,models);initial=c.begin(torch.ones(16));extra=c.acquire(torch.ones(8),torch.tensor(1.))
    assert c.fallback and torch.equal(extra,~initial)
    c.observe(torch.ones(8),initial|extra)
    models['detector']['intercept'].fill_(.04)
    initial=c.begin(torch.zeros(16));extra=c.acquire(torch.ones(8),torch.tensor(1.))
    assert not c.fallback and extra.sum()==1
    with pytest.raises(ValueError,match='Unconsumed'):c.begin(torch.ones(16))


def test_zero_batch_is_free_but_a_second_round_is_charged():
    hardware={'aggregates':[{'workload':w,'groups':g,'medians':{'wall_ms':float(g)}} for w in ('resident_ffn','gather_transfer_pack_ffn') for g in (1,2,4)]}
    assert rounded_cost(0,'resident_ffn',hardware)==0
    assert rounded_cost(3,'resident_ffn',hardware)==4
    receipt={'initial':torch.tensor([[[True,True,False,False]]]),'additions':torch.tensor([[[False,False,True,False]]])}
    result=action_cost(receipt,torch.tensor([[True,False,False,False]]).numpy(),hardware,.5)
    assert result==dict(batch_ms=2.,resident_ms=1.,controller_ms=.5,output_addition_ms=0.,total_ms=3.5)
    assert action_cost(receipt,torch.tensor([[True,False,False,False]]).numpy(),hardware,.5,.2)['total_ms']==3.7


def test_replay_preserves_recorded_detector_branch_with_accepted_roundoff():
    # An accepted CPU/GPU reduction difference may straddle 0.05. Replay must
    # verify and use the recorded decision, not silently substitute the CPU one.
    gpu_models={'input':constant_model(16,1120,1.),'base':constant_model(33,1120,1.),
        'repair':constant_model(43,1120,1.),'detector':constant_model(43,1,.04999)}
    parent={'importance':np.ones((28,1120)),'hot':np.tile(np.arange(1120)<400,(28,1))}
    records=[]
    for layer in range(28):
        c=ce.Controller(torch.ones(1120),torch.tensor(parent['hot'][layer]),ce.projections(1536,1120,1729+layer,'cpu'),gpu_models,'fallback',28)
        x=torch.zeros(1536);initial=c.begin(x);base=c.base.clone();norm=torch.tensor(1.)
        raw=torch.linspace(0,2,1120);observed_first=torch.where(initial,raw,0.)
        features=c.evidence(observed_first,norm);extra=c.acquire(observed_first,norm);final=initial|extra
        records.append(dict(x=x,input_projection=x@c.projection['input_projection'],observed=torch.where(final,raw,0.),
            initial=initial,additions=extra,initial_scores=c.initial_scores,acquisition_scores=c.acquisition_scores,
            first_norm=norm,audit=torch.zeros(2),fallback=torch.tensor(False),detection=c.detection,
            base_features=base,repair_features=features))
    data={key:torch.stack([r[key] for r in records]).unsqueeze(0) for key in records[0]}
    cpu_models={**gpu_models,'detector':constant_model(43,1,.05001)}
    verify_trace(data,parent,[cpu_models]*28,'fallback',28,1)
    data['observed'][0,0,-1]=7.
    with pytest.raises(ValueError,match='Omitted observation'):
        verify_trace(data,parent,[cpu_models]*28,'fallback',28,1)


def test_failure_survives_hook_cleanup_with_tokens_layer_and_raw_tensors(monkeypatch,tmp_path):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__();self.embed=nn.Embedding(16,16);self.ffn=TinyFFN()
        def forward(self,input_ids,past_key_values=None,use_cache=True):
            return SimpleNamespace(logits=self.ffn(self.embed(input_ids)),past_key_values=None)
    model=Decoder();model.ffn.down_proj.weight.data[0,0]=float('nan')
    mlp=FFNView(model.ffn,('gate_proj','up_proj','down_proj'),F.silu)
    monkeypatch.setattr(study,'controllers',lambda *args:[controller('full')])
    ids=torch.tensor([[1,2,3]])
    with pytest.raises(ce.ExecutionFailure) as caught:
        study.execute(model,[mlp],None,None,'full',2,ids,label='failure-fixture')
    error=caught.value
    assert error.context['layer']==0 and error.context['position']==0 and error.context['label']=='failure-fixture'
    assert torch.equal(error.tensors['token_inputs'],ids)
    assert torch.equal(error.tensors['consumed_token_inputs'],ids[:,:1])
    assert not model._forward_pre_hooks and not model.ffn._forward_hooks and not model.ffn.down_proj._forward_pre_hooks
    (tmp_path/'tensors').mkdir()
    receipt=study.save_tensors(tmp_path,'failure',error.tensors)
    saved=load_file(tmp_path/receipt['file'])
    assert not torch.isfinite(saved['corrected_output']).all()
    assert 'partial_trace.0.audit' in saved and 'initial' in saved


def test_matching_tokens_do_not_hide_changed_generation_logits():
    reference=torch.tensor([[0.,1.]])
    own=reference*100
    assert own.argmax(-1).tolist()==reference.argmax(-1).tolist()
    with pytest.raises(ValueError,match='Contradictory'):
        generation_alignment(reference,own,reference.clone(),[1],[1])
    result=generation_alignment(reference,own,own.clone(),[1],[1])
    assert result['relative_l2']>.01 and result['mean_kl']>.001

import pytest
from pathlib import Path
import gzip
import json
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading import causal
from dynamic_model_loading.causal_study import generate, incremental
from dynamic_model_loading import causal_study
from dynamic_model_loading.experiment import digest
from safetensors.torch import save
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj=nn.Linear(4,20,bias=False)
        self.up_proj=nn.Linear(4,20,bias=False)
        self.down_proj=nn.Linear(20,4,bias=False)
        self.act_fn=nn.functional.silu
    def forward(self,x):return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


@pytest.mark.parametrize('mode',['static','recency','ema','learned'])
def test_query_is_pre_activation_and_audit_never_updates_omitted_groups(mode):
    torch.manual_seed(5);mlp=TinyMLP();x=torch.randn(1,4)
    prior=torch.arange(10,dtype=torch.float32);norms=mlp.down_proj.weight.detach().norm(dim=0)
    fit=causal.RidgeFit(4,10,3,17,'cpu')
    fit.observe(torch.randn(30,4),torch.rand(30,10))
    learned=fit.finish()
    selectors=[causal.Selector(mode,prior,norms,2,learned=learned,keep=.5) for _ in range(2)]
    probes=[causal.AppliedProbe(mlp,s,2,.5) for s in selectors]
    for probe in probes:probe.before(mlp,(x,))
    mask=probes[0].mask.clone()
    assert torch.equal(mask,probes[1].mask)
    z=torch.rand(1,20)
    changed=z.clone();changed[:,~mask.repeat_interleave(2)]=10000
    first=probes[0].down(mlp.down_proj,(z,))[0]
    second=probes[1].down(mlp.down_proj,(changed,))[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    for key,value in selectors[0].tensors().items():
        torch.testing.assert_close(value,selectors[1].tensors()[key],rtol=0,atol=0)
    assert selectors[0].bytes()==sum(v.numel()*v.element_size() for v in selectors[0].tensors().values())
    with pytest.raises(ValueError,match='one token'): selectors[0].query(torch.randn(2,4))


@pytest.mark.parametrize('mode',['recency','ema'])
def test_selected_observation_state_rule_and_episode_reset(mode):
    prior=torch.tensor([1.,2.,3.,4.]); selector=causal.Selector(mode,prior,torch.ones(8),2,keep=.5)
    mask=selector.query(torch.zeros(1,4))
    assert mask.tolist()==[False,False,True,True]
    selector.observe_selected(torch.tensor([[0.,0.,0.,0.,5.,6.,7.,8.]]),mask)
    expected=torch.tensor([1.,2.,11.,15.]) if mode=='recency' else torch.tensor([1.,2.,3.8,5.1])
    torch.testing.assert_close(selector.state,expected)
    selector.reset();torch.testing.assert_close(selector.state,prior,rtol=0,atol=0)


def test_streaming_ridge_matches_batch_standardized_solution():
    torch.manual_seed(4);x=torch.randn(43,5);y=torch.randn(43,7)
    fit=causal.RidgeFit(5,7,3,1729,'cpu')
    fit.observe(x[:13],y[:13]);fit.observe(x[13:],y[13:]);trained=fit.finish()
    features=fit.features(x).double();mean=features.mean(0);scale=features.std(0,correction=0).clamp_min(1e-6)
    normalized=(features-mean)/scale
    expected=torch.linalg.solve(normalized.T@normalized/len(x)+torch.eye(6,dtype=torch.float64)*.01,
                                normalized.T@(y.double()-y.double().mean(0))/len(x))
    torch.testing.assert_close(trained['weight'],expected.float(),rtol=1e-5,atol=1e-6)
    assert fit.count==43


def test_applied_probe_matches_manual_mask_and_cleans_hooks():
    torch.manual_seed(2);mlp=TinyMLP();x=torch.randn(1,1,4)
    selector=causal.Selector('static',torch.arange(10,dtype=torch.float32),torch.ones(20),2,keep=.5)
    z=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
    expected=mlp.down_proj(z*selector.query(x).repeat_interleave(2))
    probe=causal.AppliedProbe(mlp,selector,2,.5)
    with causal.probes([mlp],[probe]):actual=mlp(x)
    torch.testing.assert_close(actual,expected)
    masks,audits=probe.take();assert masks.shape==(1,10) and audits.shape==(1,3)
    assert not mlp._forward_pre_hooks and not mlp.down_proj._forward_pre_hooks
    assert 0<=audits[0,1]<=1 and 0<=audits[0,2]<=1


def test_incremental_logits_and_fixed_length_closed_loop_use_real_qwen():
    torch.manual_seed(7)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    ids=torch.tensor([[1,3,5,7,9]])
    logits=incremental(model,ids)
    with torch.inference_mode():expected=model(input_ids=ids,use_cache=False).logits
    torch.testing.assert_close(logits,expected,rtol=1e-5,atol=1e-6)
    generated=generate(model,ids,6)
    assert generated.shape==(1,11) and torch.equal(generated[:,:5],ids)
    with torch.inference_mode():
        for index in range(5,11):
            value=model(input_ids=generated[:,:index],use_cache=False).logits[:,-1].argmax(-1)
            assert torch.equal(value,generated[:,index])


@pytest.fixture
def causal_fixture(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir()
    torch.manual_seed(29)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    model.save_pretrained(checkpoint)
    tok=Tokenizer(WordLevel({'[UNK]':0,'a':1,'b':2,'c':3},unk_token='[UNK]'))
    PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]').save_pretrained(checkpoint)
    monkeypatch.setattr(causal_study,'snapshot_download',lambda *a,**kw:str(checkpoint))
    parent=tmp_path/'parent'; parent.mkdir()
    corpus=[dict(id=name,split=split,domain='fixture',text=name+' unique fixture')
            for name,split in [('cal','calibration'),('dev1','diagnostic'),('dev2','diagnostic')]]
    (parent/'corpus.jsonl').write_text('\n'.join(json.dumps(r) for r in corpus))
    token_ids={'cal':[[1,2,3,2,1]],'dev1':[[2,3,1,2]],'dev2':[[3,2,1,3]]}
    (parent/'token-ids.json').write_text(json.dumps(token_ids))
    layouts={name:[list(range(32)),list(range(32))] for name in ('native','popularity')}
    (parent/'layouts.json.gz').write_bytes(gzip.compress(json.dumps(layouts).encode(),mtime=0))
    (parent/'calibration-importance.safetensors.gz').write_bytes(gzip.compress(save({'0':torch.ones(32),'1':torch.ones(32)}),mtime=0))
    manifest={'checkpoint':{'files':{p.name:{'sha256':digest(p)} for p in checkpoint.iterdir() if p.is_file()}}}
    (parent/'manifest.json').write_text(json.dumps(manifest))
    cfg=json.loads(Path('configs/causal-control.json').read_text())
    cfg.update(layers=2,hidden=16,neurons=32,document_counts=[1,2],generation_documents=2,
               prompt_tokens=2,generation_tokens=3,timing_tokens=2,
               expected_torch=str(torch.__version__),
               parent_hashes={p.name:digest(p) for p in parent.iterdir() if p.is_file()})
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    return config,parent,tmp_path/'output'


def test_complete_causal_fixture_records_every_mode_and_blocks_cpu_screen(causal_fixture):
    from dynamic_model_loading.causal_analysis import analyze
    result=causal_study.run(*causal_fixture,device='cpu')
    assert result['status']=='cpu_fixture_completed'
    assert result['correctness_passed'] and result['nominated_selector'] is None
    assert len(result['selectors'])==5 and all(not r['passes_screen'] for r in result['selectors'])
    assert analyze(causal_fixture[2],causal_fixture[1])['status']=='validated'
    rows=[json.loads(line) for line in (causal_fixture[2]/'results.jsonl').read_text().splitlines()]
    assert sum(r['kind']=='selector_document' for r in rows)==15
    assert sum(r['kind']=='selector_generation' for r in rows)==10
    assert sum(r['kind']=='transition_after_boundary' for r in rows)==5
    for row in rows:
        if row['kind']=='selector_generation':
            assert len(row['token_ids'])==5
            assert digest(causal_fixture[2]/row['trace_file'])==row['trace_sha256']
    with pytest.raises(FileExistsError):causal_study.run(*causal_fixture,device='cpu')


@pytest.mark.parametrize('corrupt',['gate','charged_bytes','mask','reference_nll','summary','generation',
                                   'checkpoint','drift_document','drift_metric'])
def test_causal_analysis_rejects_corrupted_artifacts(causal_fixture,corrupt):
    from dynamic_model_loading.causal_analysis import analyze
    causal_study.run(*causal_fixture,device='cpu')
    root=causal_fixture[2]; path=root/'results.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if corrupt=='gate':rows.pop(next(i for i,r in enumerate(rows) if r['kind']=='layout_correctness'))
    elif corrupt=='charged_bytes':next(r for r in rows if r['kind']=='selector_document')['selector_bytes']=0
    elif corrupt=='reference_nll':
        r=next(r for r in rows if r['kind']=='selector_document');r['dense_nll']+=1
    elif corrupt=='checkpoint':
        p=root/'manifest.json';s=json.loads(p.read_text());s['checkpoint_files']={'fake':'fake'};p.write_text(json.dumps(s))
    elif corrupt=='drift_document':next(r for r in rows if r['kind']=='prefill_incremental_drift')['document']='missing'
    elif corrupt=='drift_metric':next(r for r in rows if r['kind']=='prefill_incremental_drift')['candidate_nll']=-1
    elif corrupt=='summary':
        p=root/'summary.json';s=json.loads(p.read_text());s['nominated_selector']='static';p.write_text(json.dumps(s))
    elif corrupt=='mask':
        r=next(r for r in rows if r['kind']=='selector_document');(root/r['trace_file']).write_bytes(b'changed')
    else:
        r=next(r for r in rows if r['kind']=='selector_generation');r['token_ids'][-1]+=1
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):analyze(root,causal_fixture[1])


def test_causal_gate_failure_blocks_calibration_and_all_selectors(causal_fixture,monkeypatch):
    monkeypatch.setattr(causal_study,'grouped_forward',lambda mlp,x,width:mlp(x)*0)
    result=causal_study.run(*causal_fixture,device='cpu')
    assert result['status']=='correctness_gate_failed'
    rows=[json.loads(line) for line in (causal_fixture[2]/'results.jsonl').read_text().splitlines()]
    assert not any(r['kind'].startswith('selector') or r['kind']=='calibration_fit' for r in rows)


def test_cleanup_interruption_cannot_leave_a_success_summary(causal_fixture,monkeypatch):
    from dynamic_model_loading import relu
    original=relu.restore_originals
    def interrupt(restored):
        original(restored)
        raise KeyboardInterrupt('cleanup interrupted after restoration')
    monkeypatch.setattr(relu,'restore_originals',interrupt)
    with pytest.raises(KeyboardInterrupt):causal_study.run(*causal_fixture,device='cpu')
    assert not (causal_fixture[2]/'summary.json').exists()
    assert json.loads((causal_fixture[2]/'failure.json').read_text())['error_type']=='KeyboardInterrupt'


def test_timing_rows_have_distinct_ledger_and_workload_fields(monkeypatch):
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):pass
        def elapsed_time(self,end):return 1.0
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    mlp=TinyMLP();selector=causal.Selector('static',torch.ones(10),torch.ones(20),2)
    rows,medians=causal_study.timing([mlp],[selector],[(torch.ones(2,4),torch.ones(2,20))],
        {'timing_tokens':2,'timing_warmups':1,'timing_repetitions':2})
    assert len(rows)==6 and set(medians)=={'dense','selector'}
    def record(kind,**values):return {'kind':kind,**values}
    ledger=[record('selector_timing',mode='static',**row) for row in rows]
    assert all(r['kind']=='selector_timing' for r in ledger)
    assert {r['workload'] for r in ledger}=={'dense','selector'}

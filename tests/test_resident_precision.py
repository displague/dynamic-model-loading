import json
import numpy as np
import pytest
import torch
from dynamic_model_loading.resident_precision import aligned_metrics,decision,audit_episode,CONFIG,validate_config


def test_precision_error_does_not_compare_diverged_contexts():
    a=np.array([[2.,0,0],[2,0,0],[2,0,0],[2,0,0]])
    b=a.copy(); b[1]=[0,3,0]; b[2:]=1e9
    r=aligned_metrics(a,b)
    assert r['first_divergence']==1 and r['aligned_positions']==2 and len(r['relative_l2'])==2
    assert not r['all_ids_match'] and not r['fp32_numerical']


def test_same_greedy_ids_are_not_full_fp32_numerical_fidelity():
    a=np.array([[2.,1],[3.,1]]); b=a+.01
    r=aligned_metrics(a,b)
    assert r['all_ids_match'] and not r['fp32_numerical']
    assert aligned_metrics(a,a)['fp32_numerical']


def test_short_output_is_not_complete_greedy_match():
    a=np.array([[2.,1],[3.,1]])
    r=aligned_metrics(a,a[:1]); assert r['aligned_positions']==1 and not r['all_ids_match']


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_comparison_rejected(value):
    with pytest.raises(ValueError): aligned_metrics([[1,0]],[[value,0]])


@pytest.mark.parametrize('wall,worker,want',[(5.,60.,True),(np.nextafter(5.,np.inf),60.,False),(5.,np.nextafter(60.,np.inf),False)])
def test_useful_ordinary_access_is_distinct_from_fp32_numerical_gate(wall,worker,want):
    r=decision([{'all_ids_match':True,'fp32_numerical':False}],[{'wall_seconds':wall}],worker)
    assert r['checks']['Hordinary_access']==want and not r['checks']['Hfp32_numerical']


def fixture():
    prefix=[17]*16; arrays={}; calls=[]; ids=[4,5,2]
    for i,token in enumerate(ids):
        a=np.zeros(50272,np.float32); a[token]=1; arrays[f'logits.{i}']=a
        calls.append(dict(step=i,input_ids=prefix if i==0 else [ids[i-1]],token=token,
            started=1.+i,finished=1.5+i,kv_length=16+i,kv_bytes=(16+i)*196608,
            kv_storage_bytes=(16+i)*196608,kv_dtypes=['torch.float16'],logit_dtype='torch.float16'))
    row=dict(started=0.,finished=4.,ids=ids,stop_reason='eos',ttft_seconds=1.5,
        raw_tensor_bytes=3*50272*4,explicit_copies=dict(h2d_bytes=144,d2h_bytes=(16+2)*8+3*50272*4))
    reference=np.zeros((8,50272),np.float32); reference[:,4]=1
    return row,calls,arrays,prefix,({'stop_reason':'length'},reference)


def test_own_trajectory_eos_and_copy_audit():
    metrics,copies=audit_episode(*fixture())
    assert not metrics['all_ids_match'] and metrics['first_divergence']==1
    assert copies['h2d_bytes']==144


@pytest.mark.parametrize('key,value',[('input_ids',[4]),('kv_storage_bytes',1),('kv_dtypes',['torch.float32']),
    ('logit_dtype','torch.float32'),('token',9),('finished',99.)])
def test_episode_tampering_rejected(key,value):
    args=fixture(); args[1][0][key]=value
    with pytest.raises(ValueError): audit_episode(*args)


def test_config_freezes_dtype_not_only_budget():
    cfg=json.loads(CONFIG.read_text()); validate_config(cfg)
    cfg['parameter_dtype']='bfloat16'
    with pytest.raises(ValueError): validate_config(cfg)


def test_tiny_half_precision_model_owns_half_kv_and_full_readout():
    from transformers import OPTConfig,OPTForCausalLM,DynamicCache
    from dynamic_model_loading.decision_field import kv_storage_bytes
    from dynamic_model_loading.fault_generation import kv_bytes
    c=OPTConfig(vocab_size=37,hidden_size=16,ffn_dim=32,num_hidden_layers=2,num_attention_heads=4,
        word_embed_proj_dim=16,max_position_embeddings=64,dropout=0,attention_dropout=0)
    model=OPTForCausalLM(c).eval().half(); cache=DynamicCache(config=c)
    with torch.inference_mode():
        for ids in ([[1,3,5,7]],[[9]]):
            logits=model(torch.tensor(ids),past_key_values=cache,use_cache=True).logits
            assert logits.dtype==torch.float16 and logits.shape[-1]==37
            assert all(l.keys.dtype==l.values.dtype==torch.float16 for l in cache.layers)
            assert kv_bytes(cache)==kv_storage_bytes(cache)==cache.get_seq_length()*2*2*16*2


def test_full_shape_half_parameter_bytes_and_ties_without_checkpoint():
    from transformers import OPTConfig,OPTForCausalLM
    c=OPTConfig(vocab_size=50272,hidden_size=2048,ffn_dim=8192,num_hidden_layers=24,num_attention_heads=32,
        word_embed_proj_dim=2048,max_position_embeddings=2048)
    with torch.device('meta'): m=OPTForCausalLM(c).half()
    assert sum(p.numel()*p.element_size() for p in m.parameters())==2631516160
    assert sum(b.numel()*b.element_size() for b in m.buffers())==0
    assert m.lm_head.weight is m.model.decoder.embed_tokens.weight

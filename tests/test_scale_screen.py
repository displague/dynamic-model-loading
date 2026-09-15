import copy
import pytest
from dynamic_model_loading.scale_support import audit_allocation,TOTAL,OUTGOING,NON_OUTGOING,LIMIT


def allocation():
    return dict(cpu_first=True,full_gpu_model_loaded=False,tied_embeddings=True,
        before_candidate_cuda=dict(allocated_bytes=0,reserved_bytes=0,peak_allocated_bytes=0,peak_reserved_bytes=0),
        cpu_loaded_parameter_bytes=TOTAL,construction_h2d_bytes=NON_OUTGOING,
        candidate_parameter_bytes=NON_OUTGOING,construction_d2h_bytes=0,
        host_weights_bytes=OUTGOING,extra_host_linear_bytes=OUTGOING,
        parameter_dtype='torch.float16',direct_host_contiguous=True,
        reference_kind='dense_contiguous_fp16_stream',allocator_fraction=LIMIT/(16*2**30),
        candidate_cuda=dict(peak_reserved_bytes=LIMIT))


def test_capacity_exact_boundary_and_corruption():
    a=allocation(); audit_allocation(a,16*2**30)
    for key,value in [('full_gpu_model_loaded',True),('cpu_first',False),('tied_embeddings',False),
        ('construction_h2d_bytes',NON_OUTGOING-1),('extra_host_linear_bytes',0),
        ('allocator_fraction',1.),('parameter_dtype','torch.float32')]:
        bad=copy.deepcopy(a); bad[key]=value
        with pytest.raises(ValueError): audit_allocation(bad,16*2**30)
    for section,key in [('before_candidate_cuda','allocated_bytes'),('before_candidate_cuda','reserved_bytes'),
        ('before_candidate_cuda','peak_allocated_bytes'),('before_candidate_cuda','peak_reserved_bytes'),
        ('candidate_cuda','peak_reserved_bytes')]:
        bad=copy.deepcopy(a); bad[section][key]+=1
        with pytest.raises(ValueError): audit_allocation(bad,16*2**30)


def test_scale_integer_gate_boundaries():
    from dynamic_model_loading.scale_screen import decision
    c=dict(packet=dict(h2d_bytes=50,wall_seconds=8),stream=dict(h2d_bytes=100,wall_seconds=10))
    assert all(decision(c)['checks'].values())
    c['packet']['h2d_bytes']=51; assert not decision(c)['checks']['Hacquisition']
    c['packet']['wall_seconds']=8.01; assert not decision(c)['checks']['Hruntime']


def test_larger_dense_control_raw_dimensions():
    from dynamic_model_loading.scale_screen import audit_pages
    rows=[dict(layer=i,call=i+1,tokens=1,condition='stream',direct_contiguous_copy=True,
        weight_h2d_bytes=52428800,metadata_h2d_bytes=0,activity_d2h_bytes=0,
        started=1.+i,selection_finished=1.+i,acquisition_finished=1.1+i,finished=1.2+i) for i in range(32)]
    calls=[dict(input_ids=[3],started=1.,finished=40.)]
    assert audit_pages(rows,{},calls,'stream',0)['weight_h2d_bytes']==OUTGOING
    rows[0]['weight_h2d_bytes']=33554432
    with pytest.raises(ValueError): audit_pages(rows,{},calls,'stream',0)


def test_meta_scale_parameter_accounting_without_inference():
    import torch
    from transformers import OPTConfig,OPTForCausalLM
    c=OPTConfig(hidden_size=2560,ffn_dim=10240,num_hidden_layers=32,num_attention_heads=32,
        word_embed_proj_dim=2560,vocab_size=50272,max_position_embeddings=2048)
    with torch.device('meta'): model=OPTForCausalLM(c)
    assert sum(p.numel()*2 for p in model.parameters())==TOTAL
    assert sum(l.fc2.weight.numel()*2 for l in model.model.decoder.layers)==OUTGOING
    assert TOTAL-OUTGOING==NON_OUTGOING and TOTAL>LIMIT

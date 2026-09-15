import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from dynamic_model_loading.capacity_down import CapacityDown,move_except_down
from dynamic_model_loading.capacity_screen import CONFIG,validate_config,decision
from dynamic_model_loading.capacity_support import LIMIT,bounded,audit_allocation


@pytest.mark.parametrize('mode',['stream','packet'])
def test_original_layout_direct_stream_and_packet_match(mode):
    torch.manual_seed(52)
    layers=[SimpleNamespace(fc2=torch.nn.Linear(7,3)) for _ in range(2)]
    weights=[l.fc2.weight.detach().clone() for l in layers]; rows=[]
    bank=CapacityDown(layers,rows.append,width=2,device='cpu')
    bank.begin('tiny',mode)
    for i,x in [(1,torch.ones(2,7)),(0,torch.tensor([[0.,2.,0,0,0,0,3.]])),(1,torch.zeros(1,7))]:
        torch.testing.assert_close(layers[i].fc2(x),torch.nn.functional.linear(x,weights[i],layers[i].fc2.bias),rtol=0,atol=0)
    assert bank.allocation()['extra_host_linear_bytes']==2*7*3*4
    assert bank.allocation()['direct_dense_host_contiguous']
    if mode=='stream': assert all(r['direct_contiguous_copy'] for r in rows)


def test_move_excludes_weights_and_preserves_tied_parameter_objects():
    m=torch.nn.Module(); m.fc2=torch.nn.Linear(7,3); m.embedding=torch.nn.Embedding(9,3)
    m.head=torch.nn.Linear(3,9,bias=False); m.head.weight=m.embedding.weight
    original=m.fc2.weight; tied=m.head.weight; checks=[]
    copied=move_except_down(m,[SimpleNamespace(fc2=m.fc2)],device='cpu',check=lambda:checks.append(1))
    assert copied==(3+9*3)*4 and len(checks)==2
    assert m.fc2.weight is original and m.head.weight is m.embedding.weight is tied


def cuda(allocated=0,reserved=0):
    return dict(allocated_bytes=allocated,reserved_bytes=reserved,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)


@pytest.mark.parametrize('key',['peak_allocated_bytes','peak_reserved_bytes'])
def test_capacity_allocator_highwater_not_only_current(key):
    r=cuda(); r[key]=LIMIT+1
    if key=='peak_allocated_bytes': r['peak_reserved_bytes']=LIMIT+1
    with pytest.raises(ValueError): bounded(r)


def allocation():
    return dict(parameters_bytes=5263032320,registered_buffers_bytes=0,cpu_first=True,
        cpu_loaded_parameter_bytes=5263032320,construction_h2d_bytes=3652419584,
        host_down_bytes=1610612736,workspace_bytes=67108864,pinned_staging_bytes=67108864,
        construction_d2h_bytes=0,hybrid_cuda_parameters_bytes=3652419584,host_weight_aliases=True,
        packet_host_bytes=67174400,packet_cuda_bytes=67174400,total_pinned_bytes=134283264,
        original_workspace_layout=True,weight_workspace_contiguous=True,workspace_strides=[1,8192],
        extra_host_linear_bytes=1610612736,direct_dense_host_contiguous=True,allocator_limit_bytes=LIMIT,
        allocator_fraction=LIMIT/(16*2**30),before_residency_cuda=cuda(),hybrid_cuda=cuda(3800000000,3900000000),
        conversion_seconds=1.,setup_seconds=2.)


def test_capacity_static_accounting_includes_duplicate_host_and_all_scratch():
    audit_allocation(allocation(),{'gpu':{'total_bytes':16*2**30}})


@pytest.mark.parametrize('key,value',[('extra_host_linear_bytes',0),('construction_h2d_bytes',5263032320),
    ('cpu_first',False),('restore_h2d_bytes',1),('before_residency_cuda',cuda(1,1)),('allocator_fraction',1.)])
def test_capacity_accounting_fails_closed(key,value):
    a=allocation(); a[key]=value
    with pytest.raises(ValueError): audit_allocation(a,{'gpu':{'total_bytes':16*2**30}})


def test_capacity_new_gate_inclusive_and_no_latency_waiver():
    rows={'stream':dict(h2d_bytes=100,wall_seconds=1.),'packet':dict(h2d_bytes=50,wall_seconds=.8)}
    assert all(decision(rows)['checks'].values())
    rows['packet']['wall_seconds']=.8001
    assert not decision(rows)['checks']['Hruntime']


def test_capacity_frozen_budget_and_workload():
    cfg=json.loads(CONFIG.read_text()); validate_config(cfg)
    cfg['gpu_limit_mib']=5000
    with pytest.raises(ValueError): validate_config(cfg)


def test_external_reference_hash_and_greedy_binding(tmp_path,monkeypatch):
    from dynamic_model_loading import capacity_support as support
    from dynamic_model_loading.specialist_screen import sha
    folder=tmp_path/'reference'; folder.mkdir(); prefixes=[[1]*16,[2]*16]
    (folder/'manifest.json').write_text(json.dumps({'source_commit':support.REFERENCE_SOURCE}))
    (folder/'tokens.json').write_text(json.dumps(prefixes))
    for doc in (0,1):
        dest=folder/f'episode-{doc}'; dest.mkdir()
        (dest/'episode.json').write_text(json.dumps(dict(condition='resident',document=doc,stop_reason='length',ids=[0]*8)))
        save_file({f'logits.{i}':np.zeros(50272,np.float32) for i in range(8)},dest/'tensors.safetensors')
    raw=json.dumps({'inventory':{'row-packet-screen-20260915-v2/worker/'+name:sha(folder/name) for name in support.FILES}}).encode()
    (folder/'archive.json').write_bytes(raw); monkeypatch.setattr(support,'parent_receipt',lambda:raw)
    assert len(support.load_references(tmp_path,prefixes))==2
    (folder/'tokens.json').write_text('[]')
    with pytest.raises(ValueError): support.load_references(tmp_path,prefixes)

from types import SimpleNamespace
import pytest
import torch
from dynamic_model_loading.precision_rows import PrecisionRows


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_direct_stream_and_packet_match_original_dense(dtype):
    torch.manual_seed(39)
    layers=[SimpleNamespace(fc2=torch.nn.Linear(7,3,dtype=dtype)) for _ in range(2)]
    original=[l.fc2.weight.detach().clone() for l in layers]; records=[]
    bank=PrecisionRows(layers,capacity=0,device='cpu',record=records.append)
    for condition in ('packet','stream','packet'):
        bank.begin('test',condition)
        for i,selected in [(0,[0,1]),(1,[1,4]),(0,[]),(1,list(range(7)))]:
            x=torch.zeros(2,7,dtype=dtype); x[:,selected]=3
            torch.testing.assert_close(layers[i].fc2(x),torch.nn.functional.linear(x,original[i],layers[i].fc2.bias),rtol=0,atol=0)
    a=bank.allocation(); assert a['extra_host_linear_bytes']==2*7*3*torch.tensor([],dtype=dtype).element_size()
    assert a['direct_host_contiguous']


def test_transport_and_resident_speed_gates_are_separate():
    from dynamic_model_loading.precision_packet_screen import decision
    conditions=dict(packet=dict(h2d_bytes=50,wall_seconds=8),stream=dict(h2d_bytes=100,wall_seconds=10),
        resident=dict(h2d_bytes=0,wall_seconds=1))
    result=decision(conditions)
    assert result['checks']['Htransport'] and not result['checks']['Hresident_speed']
    conditions['packet']['h2d_bytes']=51; assert not decision(conditions)['checks']['Htransport']


def test_dense_control_raw_byte_audit():
    from dynamic_model_loading.precision_packet_screen import audit_pages
    rows=[dict(layer=i,call=i+1,tokens=1,condition='stream',direct_contiguous_copy=True,
        weight_h2d_bytes=33554432,metadata_h2d_bytes=0,activity_d2h_bytes=0,
        started=1.+i,selection_finished=1.+i,acquisition_finished=1.1+i,finished=1.2+i) for i in range(24)]
    calls=[dict(input_ids=[3],started=1.,finished=30.)]
    assert audit_pages(rows,{},calls,'stream',0)['weight_h2d_bytes']==805306368
    rows[0]['weight_h2d_bytes']+=1
    with pytest.raises(ValueError): audit_pages(rows,{},calls,'stream',0)

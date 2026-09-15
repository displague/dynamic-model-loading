from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamic_model_loading.row_packet import RowPacket
from dynamic_model_loading.packet_analysis import audit_pages, decision


def test_single_packet_exact_weights_indices_and_stale_zero_rows():
    torch.manual_seed(92)
    ls=[SimpleNamespace(fc2=torch.nn.Linear(7,3)) for _ in range(2)]
    weights=[l.fc2.weight.detach().clone() for l in ls]; records=[]
    bank=RowPacket(ls,records.append,width=2,device='cpu')
    for condition in ('stream','sparse','packet'):
        bank.begin('test',condition)
        for layer,x in [(1,torch.ones(2,7)),(0,torch.tensor([[0.,2.,0,0,0,0,3.],[0,0,0,0,0,0,0]]))]:
            expected=torch.nn.functional.linear(x,weights[layer],ls[layer].fc2.bias)
            torch.testing.assert_close(ls[layer].fc2(x),expected,rtol=1e-6,atol=1e-6)
        if condition=='packet':
            assert records[-1]['rows']==[1,6] and records[-1]['packets']==1
            torch.testing.assert_close(bank.packet_host[:16].view(torch.int64),torch.tensor([1,6]))
            torch.testing.assert_close(bank.packet_host[16:40].view(torch.float32).reshape(2,3),weights[0][:,[1,6]].T,rtol=0,atol=0)
    bank.restore()
    for layer,w in zip(ls,weights): torch.testing.assert_close(layer.fc2.weight,w,rtol=0,atol=0)


def test_empty_packet_keeps_bias():
    ls=[SimpleNamespace(fc2=torch.nn.Linear(7,3))]; rows=[]
    bank=RowPacket(ls,rows.append,width=2,device='cpu'); bank.begin('zero','packet')
    torch.testing.assert_close(ls[0].fc2(torch.zeros(2,7)),ls[0].fc2.bias.expand(2,-1),rtol=0,atol=0)
    assert rows[0]['packets']==0 and rows[0]['rows']==[]


def test_packet_nonfinite_rejected():
    bank=RowPacket([SimpleNamespace(fc2=torch.nn.Linear(7,3))],device='cpu')
    bank.begin('nan','packet')
    with pytest.raises(ValueError): bank.forward(0,torch.full((1,7),float('nan')))


def packet_fixture():
    calls=[dict(input_ids=[17],started=1.,finished=5.)]
    episode=dict(condition='packet',episode='episode-4')
    arrays={f'activity.{i+1}':np.zeros((1,1024),dtype=np.uint8) for i in range(24)}
    for a in arrays.values(): a[0,0]=128
    pages=[dict(call=i+1,layer=i,tokens=1,episode='episode-4',condition='packet',
        started=2+i*.1,selection_finished=2.01+i*.1,packing_finished=2.02+i*.1,
        transfer_finished=2.03+i*.1,acquisition_finished=2.04+i*.1,compute_finished=2.08+i*.1,
        rows=[0],packets=1,weight_h2d_bytes=8192,metadata_h2d_bytes=8,packet_h2d_bytes=8200,
        activity_d2h_bytes=8193) for i in range(24)]
    return pages,arrays,calls,episode


def test_packet_replay_charges_indices():
    stats=audit_pages(*packet_fixture())
    assert stats['h2d_bytes']==24*8200 and stats['metadata_h2d_bytes']==24*8


@pytest.mark.parametrize('key,value',[('rows',[1]),('metadata_h2d_bytes',0),('packet_h2d_bytes',8192),
    ('packets',0),('transfer_finished',10.)])
def test_packet_replay_tamper(key,value):
    args=packet_fixture(); args[0][0][key]=value
    with pytest.raises(ValueError): audit_pages(*args)


def test_packet_must_beat_both_controls_including_metadata():
    controls=dict(stream=dict(h2d_bytes=200,wall_seconds=20),sparse=dict(h2d_bytes=100,wall_seconds=10),
                  packet=dict(h2d_bytes=50,wall_seconds=8))
    assert all(decision(controls)['checks'].values())
    controls['packet']['h2d_bytes']=51
    assert not decision(controls)['checks']['Htraffic']

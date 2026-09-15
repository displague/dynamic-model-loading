from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamic_model_loading.early_rows import EarlyRows,completion_rows


def test_forecast_does_not_depend_on_current_activity():
    assert completion_rows([1,3,5],[0,1,7],2)==([1,3],[0,7])
    assert completion_rows([1,3,5],[3],2)==([1,3],[])


@pytest.mark.parametrize('condition',['packet','late','early'])
def test_false_forecasts_and_misses_preserve_full_arithmetic(condition):
    torch.manual_seed(38)
    layers=[SimpleNamespace(fc1=torch.nn.Linear(3,7),fc2=torch.nn.Linear(7,3)) for _ in range(2)]
    weights=[l.fc2.weight.detach().clone() for l in layers]; rows=[]
    bank=EarlyRows(layers,capacity=0,device='cpu',record=rows.append); bank.begin('test',condition)
    for i,selected in [(0,[0,2]),(1,[4]),(0,[1,2]),(1,[]),(0,[5,6])]:
        layers[i].fc1(torch.ones(1,3))
        x=torch.zeros(1,7); x[:,selected]=2.
        actual=layers[i].fc2(x); expected=torch.nn.functional.linear(x,weights[i],layers[i].fc2.bias)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if condition!='packet':
        assert rows[2]['predicted']==[0,2] and rows[2]['misses']==[1]
        assert rows[2]['wasted_prefetched']==1
    bank.begin('reset',condition); assert not any(bank.previous)


def test_unconsumed_packet_cannot_reset_or_reenter():
    layer=SimpleNamespace(fc1=torch.nn.Linear(3,7),fc2=torch.nn.Linear(7,3))
    bank=EarlyRows([layer],capacity=0,device='cpu'); bank.begin('test','early')
    layer.fc1(torch.ones(1,3))
    with pytest.raises(ValueError): bank.begin('bad','early')
    with pytest.raises(ValueError): layer.fc1(torch.ones(1,3))


def test_gate_requires_same_predictions_bytes_and_both_wall_controls():
    from dynamic_model_loading.early_screen import decision
    groups=dict(packet=dict(h2d_bytes=100,wall_seconds=100.),late=dict(h2d_bytes=125,wall_seconds=100.),
        early=dict(h2d_bytes=125,wall_seconds=95.))
    assert all(decision(groups)['checks'].values())
    groups['early']['h2d_bytes']=126; assert not decision(groups)['checks']['Htraffic']
    groups['early']['wall_seconds']=96; assert not decision(groups)['checks']['Htiming']


def test_raw_audit_rejects_current_activity_as_past_prediction():
    from dynamic_model_loading.early_screen import audit_pages
    calls=[dict(input_ids=[5],started=1.,finished=50.)]; pages=[]; arrays={}
    for i in range(24):
        a=np.zeros((1,1024),np.uint8); a[0,0]=128; arrays[f'activity.{i+1}']=a
        pages.append(dict(call=i+1,layer=i,tokens=1,condition='early',active=[0],predicted=[],misses=[0],
            useful_prefetched=0,wasted_prefetched=0,weight_h2d_bytes=8192,metadata_h2d_bytes=8,
            activity_d2h_bytes=8193,copy_ms=0.,overlap_ms=0.,wait_seconds=0.,
            event_intervals=dict(copy_start_ms=0.,copy_end_ms=0.,fc1_start_ms=0.,fc1_end_ms=0.),
            started=1.+i,selection_finished=1.1+i,acquisition_finished=1.2+i,finished=1.3+i))
    assert audit_pages(pages,arrays,calls,'early',0)['weight_h2d_bytes']==24*8192
    pages[0]['predicted']=[0]
    with pytest.raises(ValueError): audit_pages(pages,arrays,calls,'early',0)

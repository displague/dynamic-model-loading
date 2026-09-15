from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamic_model_loading.phase_rows import PhaseRows
from dynamic_model_loading.phase_screen import audit_pages,decision


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_phase_changes_only_transport_and_restores_recording(dtype):
    torch.manual_seed(42)
    layers=[SimpleNamespace(fc2=torch.nn.Linear(7,3,dtype=dtype)) for _ in range(2)]
    weights=[l.fc2.weight.detach().clone() for l in layers]; rows=[]
    bank=PhaseRows(layers,capacity=0,device='cpu',record=rows.append)
    for mode in ('phase','packet','stream','phase'):
        bank.begin('test',mode)
        for i,n in [(0,4),(1,4),(0,1),(1,1)]:
            x=torch.zeros(n,7,dtype=dtype); x[:,[1,4]]=3
            torch.testing.assert_close(layers[i].fc2(x),torch.nn.functional.linear(x,weights[i],layers[i].fc2.bias),rtol=0,atol=0)
            assert bank.condition==mode and rows[-1]['condition']==mode
            if mode=='phase':
                assert rows[-1]['physical_condition']==('stream' if n>1 else 'packet')
                assert (rows[-1]['activity'] is None)==(n>1)


def test_phase_restores_mode_after_invalid_input():
    layer=SimpleNamespace(fc2=torch.nn.Linear(7,3)); bank=PhaseRows([layer],capacity=0,device='cpu')
    callback=bank.record; bank.begin('test','phase')
    with pytest.raises(ValueError): layer.fc2(torch.ones(2,6))
    assert bank.condition=='phase' and bank.record is callback


def test_phase_gate_boundaries_charge_extra_bytes():
    c=dict(packet=dict(h2d_bytes=100,prefill_seconds=10,wall_seconds=20,decode_h2d_bytes=30),
        stream=dict(h2d_bytes=1000,wall_seconds=50,prefill_h2d_bytes=100),
        phase=dict(h2d_bytes=130,prefill_seconds=9,wall_seconds=19,prefill_h2d_bytes=100,decode_h2d_bytes=30))
    assert all(decision(c)['checks'].values())
    c['phase']['h2d_bytes']=131
    assert not decision(c)['checks']['Htraffic_tradeoff']
    c['phase']['wall_seconds']=19.01
    assert not decision(c)['checks']['Hwhole_episode_gain']


def test_phase_raw_audit_counts_prefill_and_decode_once():
    pages=[]; arrays={}; calls=[dict(input_ids=[3]*512,started=0.,finished=100.),dict(input_ids=[4],started=100.,finished=200.)]
    for j in range(64):
        layer=j%32; prefill=j<32; start=float(j if prefill else j+70)
        r=dict(layer=layer,call=j+1,tokens=512 if prefill else 1,condition='phase',
            physical_condition='stream' if prefill else 'packet',started=start,
            selection_finished=start,acquisition_finished=start+.1,finished=start+.2)
        if prefill:
            r.update(direct_contiguous_copy=True,weight_h2d_bytes=52428800,metadata_h2d_bytes=0,activity_d2h_bytes=0)
        else:
            a=np.zeros((1,10240),dtype=np.uint8); a[0,1]=1
            arrays[f'activity.{j+1}']=np.packbits(a,axis=1)
            r.update(active=[1],hits=[],misses=[1],inserts=[],residency=[],
                weight_h2d_bytes=5120,metadata_h2d_bytes=8,activity_d2h_bytes=10241)
        pages.append(r)
    result=audit_pages(pages,arrays,calls,'phase',0)
    assert result['prefill_h2d_bytes']==1677721600
    assert result['decode_h2d_bytes']==32*(5120+8)
    assert result['weight_h2d_bytes']+result['metadata_h2d_bytes']==result['prefill_h2d_bytes']+result['decode_h2d_bytes']
    pages[32]['physical_condition']='stream'
    with pytest.raises(ValueError): audit_pages(pages,arrays,calls,'phase',0)

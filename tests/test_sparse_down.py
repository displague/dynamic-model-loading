from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamic_model_loading.sparse_down import SparseDown, active_extents


def layers():
    torch.manual_seed(91)
    return [SimpleNamespace(fc2=torch.nn.Linear(7,3)) for _ in range(2)]


def test_union_coalescing_and_short_tail():
    a=np.zeros((2,7),dtype=bool)
    a[0,0]=True
    a[1,6]=True
    assert active_extents(a,2)==([0,3],[[0,2],[6,7]])
    a[1,2]=True
    assert active_extents(a,2)==([0,1,3],[[0,4],[6,7]])


def test_empty_activity_requires_no_load():
    assert active_extents(np.zeros((2,7),dtype=bool),2)==([],[])


@pytest.mark.parametrize('condition',['stream','sparse'])
def test_original_bias_and_output_match_with_shared_stale_workspace(condition):
    ls=layers()
    weights=[l.fc2.weight.detach().clone() for l in ls]
    biases=[l.fc2.bias.detach().clone() for l in ls]
    rows=[]
    bank=SparseDown(ls,rows.append,width=2,device='cpu')
    bank.begin('tiny',condition)
    for index in (0,1,0):
        x=torch.tensor([[1.,0,0,0,0,0,2.],[0,0,0,0,0,0,0]])
        expected=torch.nn.functional.linear(x,weights[index],biases[index])
        torch.testing.assert_close(ls[index].fc2(x),expected,rtol=1e-6,atol=1e-6)
    assert bank.allocation()['host_weight_aliases']
    assert all(r['weight_h2d_bytes']==r['activity_d2h_bytes']==0 for r in rows)
    assert bank.restore()==0
    for i,l in enumerate(ls):
        torch.testing.assert_close(l.fc2.weight,weights[i],rtol=0,atol=0)
        assert l.fc2.weight.is_contiguous()


def test_zero_activation_keeps_bias_without_loading():
    ls=layers()
    bank=SparseDown(ls,width=2,device='cpu')
    bank.begin('zero','sparse')
    result=ls[0].fc2(torch.zeros((3,7)))
    torch.testing.assert_close(result,ls[0].fc2.bias.expand(3,-1),rtol=0,atol=0)
    assert bank.loads==0


def test_nonfinite_activation_cannot_be_skipped():
    bank=SparseDown(layers(),width=2,device='cpu')
    bank.begin('bad','sparse')
    with pytest.raises(ValueError): bank.forward(0,torch.full((1,7),float('nan')))


@pytest.mark.parametrize('activity,width',[(np.ones((2,7)),2),(np.ones(7,dtype=bool),2),
                                          (np.ones((2,7),dtype=bool),0)])
def test_invalid_discovery_inputs(activity,width):
    with pytest.raises(ValueError): active_extents(activity,width)


def test_all_active_equals_full_extent():
    assert active_extents(np.ones((2,7),dtype=bool),2)==([0,1,2,3],[[0,7]])

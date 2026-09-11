import json

import pytest
import torch

from dynamic_model_loading.hardware_cost import byte_count, rank, reduce_timings, run
from dynamic_model_loading.hardware_cost import synthetic_inputs, input_receipt


def test_input_recipe_is_rng_isolated_and_receipt_detects_changed_weights():
    pool,order,x=synthetic_inputs(5,2,4)
    expected=input_receipt(pool,order,x)
    torch.manual_seed(92);torch.randn(75)
    assert input_receipt(*synthetic_inputs(5,2,4))==expected
    pool[0,0,0,0]+=1
    assert input_receipt(pool,order,x)['pool_sha256']!=expected['pool_sha256']


def test_group_payload_includes_all_three_width8_projections():
    assert byte_count(1)==147456
    assert byte_count(1120)==3*1536*8960*4
    with pytest.raises(ValueError):byte_count(True)
    with pytest.raises(ValueError):byte_count(0)


def test_ranking_methods_match_unique_scores_and_stable_ties_are_explicit():
    torch.manual_seed(5)
    scores=torch.randperm(1120).float()
    assert torch.equal(rank(scores,'stable_argsort'),rank(scores,'topk'))
    expected=torch.arange(1120)<1008
    assert torch.equal(rank(torch.ones(1120),'stable_argsort'),expected)
    assert rank(torch.ones(1120),'topk').sum()==1008


def test_medians_exclude_all_three_warm_repetitions():
    rows=[{'repetition':i,'warmup':i<3,'cuda_ms':i+1.,'wall_ms':i+2.} for i in range(13)]
    assert reduce_timings(rows)=={'cuda_ms':8.5,'wall_ms':9.5}


@pytest.mark.parametrize('mutation',['missing','order','warmup','nan','negative','boolean'])
def test_invalid_timing_receipts_cannot_validate(mutation):
    rows=[{'repetition':i,'warmup':i<3,'cuda_ms':i+1.,'wall_ms':i+2.} for i in range(13)]
    if mutation=='missing':rows.pop()
    elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='warmup':rows[4]['warmup']=True
    elif mutation=='nan':rows[4]['cuda_ms']=float('nan')
    elif mutation=='negative':rows[4]['wall_ms']=-1
    else:rows[4]['cuda_ms']=True
    with pytest.raises(ValueError):reduce_timings(rows)


def test_cpu_unavailability_is_retained_as_failure_not_timing(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    output=tmp_path/'unavailable'
    with pytest.raises(ValueError,match='CUDA'):run(output,'unused-protocol')
    assert (output/'failure.json').exists() and not (output/'summary.json').exists()
    rows=[json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in rows]==['error']

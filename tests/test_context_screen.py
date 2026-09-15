import copy
import json
from pathlib import Path
import pytest
from dynamic_model_loading.context_screen import stage_decision,audit_workload


def test_context_stage_boundaries_are_independent():
    c=dict(stream=dict(prefill_h2d_bytes=100,decode_h2d_bytes=100,prefill_seconds=2.,decode_seconds=10.),
        packet=dict(prefill_h2d_bytes=50,decode_h2d_bytes=10,prefill_seconds=2.,decode_seconds=5.))
    assert all(stage_decision(c)['checks'].values())
    c['packet']['prefill_h2d_bytes']=51; c['packet']['prefill_seconds']=2.01
    checks=stage_decision(c)['checks']
    assert not checks['Hprefill_traffic'] and not checks['Hprefill_runtime']
    assert checks['Hdecode_traffic'] and checks['Hdecode_runtime']


def test_context_workload_is_exact_archived_concatenation():
    cfg=json.loads(Path('configs/context-screen.json').read_text())
    audit_workload(cfg)
    for key,value in [('prefix_tokens',513),('generation_tokens',17),('warmup_prompt','changed')]:
        bad=copy.deepcopy(cfg); bad[key]=value
        with pytest.raises(ValueError): audit_workload(bad)


def test_context_zero_denominator_fails_closed():
    c=dict(stream=dict(prefill_h2d_bytes=0,decode_h2d_bytes=100,prefill_seconds=2.,decode_seconds=10.),packet={})
    with pytest.raises(ValueError): stage_decision(c)

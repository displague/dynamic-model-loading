from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import stock_placement as study
import analyze_stock_placement as audit


def test_registered_threshold_tie_rule_and_nonfinite_rejection():
    assert audit.choose_threshold({2:10,4:10.05,8:10.09})==8
    assert audit.choose_threshold({2:10,4:10.05,8:10.2})==4
    for bad in ({2:1,4:2},{2:float('nan'),4:1,8:2},{2:-1,4:2,8:3}):
        with pytest.raises(ValueError): audit.choose_threshold(bad)


def test_grid_keeps_draft_and_resources_fixed():
    for t in [2,4,8]:
        cfg=study.configuration('threshold',t,0)
        assert (cfg['k'],cfg['ngl'],cfg['context'],cfg['kv'])==(16,38,18432,'q8_0')
        assert study.configuration('allocation',t,36)['ngl']==65
    for args in [('threshold',32,0),('threshold',2,36),('allocation',8,0),('allocation',8,28)]:
        with pytest.raises(ValueError): study.configuration(*args)


def graph(layers,backend):
    lines=[f'node #{i} (FLASH_ATTN): node_{i} (20K) [{backend}] use=0: cache_k_l{i}' for i in range(layers)]
    for i in range(layers):
        for k in ['up','gate','down']:
            name='out' if k=='down' else k
            lines.append(f'node #{len(lines)} (MUL_MAT): ffn_{name}-{i} (20K) [{backend}] CUDA0#blk.{i}.ffn_{k}.')
    return '\n'.join(lines)


def test_graph_audit_separates_draft_and_target_graphs_and_detects_cpu_attention():
    good=graph(24,'CPU')+'\n'+graph(64,'CUDA0')
    assert audit.graph_evidence(good)['all_target_attention_cuda']
    assert len(audit.graph_evidence(good)['target_graphs'])==1
    assert len(audit.graph_evidence(good)['target_graphs'][0]['ffn'])==192
    assert not audit.graph_evidence(graph(24,'CUDA0'))['all_target_attention_cuda']
    assert not audit.graph_evidence(graph(64,'CPU'))['all_target_attention_cuda']
    assert not audit.graph_evidence(good+'\n'+graph(64,'CPU'))['all_target_attention_cuda']


def test_cpu_specialized_buffers_do_not_pass_pinned_host_check(monkeypatch):
    monkeypatch.setattr(study.stock,'validate_placement',lambda *a:None)
    cfg=study.configuration('allocation',8,36)
    kv='llama_kv_cache: CUDA0 KV buffer size = 2448.00 MiB\nllama_kv_cache: CUDA0 KV buffer size = 114.77 MiB\n'
    rows='\n'.join(f'tensor blk.{i}.ffn_{k}.weight buffer type overridden to CUDA_Host' for i in range(36) for k in ['up','down','gate'])
    assert study.placement_evidence(kv+rows,cfg)['host_override_pass']
    assert not study.placement_evidence(kv+rows.replace('CUDA_Host','CPU_REPACK'),cfg)['host_override_pass']
    assert not study.placement_evidence(kv+rows.replace('blk.0.ffn_up','blk.1.ffn_up'),cfg)['host_override_pass']
    actual='\n'.join(f'load_tensors: {b} model buffer size = 100.00 MiB' for b in ['CUDA0','CUDA_Host','CUDA0','CUDA_Host'])
    evidence=study.placement_evidence(kv+rows+'\n'+actual,cfg)
    import json
    assert json.loads(json.dumps(evidence))==evidence
    assert evidence['actual_host_pass']
    assert not study.placement_evidence(kv+rows+'\n'+actual.replace('CUDA_Host','CPU'),cfg)['actual_host_pass']


def test_allocation_selection_cannot_stop_early(tmp_path,monkeypatch):
    run=tmp_path/audit.identity('allocation',8,36,1); run.mkdir()
    monkeypatch.setattr(audit,'audit',lambda p:{'allocation_eligible':True,'resources':{},'interval':[1,2]})
    with pytest.raises(ValueError,match='boundary'): audit.allocation_decision(tmp_path,8)
    run2=tmp_path/audit.identity('allocation',8,34,1); run2.mkdir()
    study.stock.write_json(run2/'failure.json',{'type':'RuntimeError','message':'allocation'})
    with pytest.raises(ValueError,match='unscored'): audit.allocation_decision(tmp_path,8)


def test_missing_matrix_source_is_rejected_before_reading_data(tmp_path):
    run=tmp_path/audit.identity('threshold',8,0,1); run.mkdir()
    study.stock.write_json(run/'attempt.json',{'kind':'threshold','repeat':1,
        'configuration':study.configuration('threshold',8,0),'source_sha256':{}})
    with pytest.raises(ValueError,match='source coverage'): audit.audit(run)


def test_timing_counters_and_native_step_denominator():
    base={'prompt_n':16384,'cache_n':0,'predicted_n':128,'draft_n':160,'draft_n_accepted':100,'prompt_ms':30000,'predicted_ms':9000}
    audit.validate_native_timing(base,128,39)
    for bad in [dict(base,predicted_n=127),dict(base,predicted_n=128.0),dict(base,prompt_ms=1000000)]:
        with pytest.raises(ValueError): audit.validate_native_timing(bad,128,39)
    value=audit.pooled([{'rows':[{'case':'x','timings':base,'emitted':128,'request_s':39,'reference_match':True}],
                         'resources':{'gpu_peak':1}}])
    assert value['native_steps_per_s']==127/9


def test_serial_order_rejects_overlap_and_reversal():
    audit.serial_order([{'interval':[1,2]},{'interval':[2,3]}])
    for bad in [[{'interval':[1,3]},{'interval':[2,4]}],[{'interval':[3,4]},{'interval':[1,2]}]]:
        with pytest.raises(ValueError): audit.serial_order(bad)


def test_process_bounds_reject_contradictory_request_and_resource_times():
    audit.process_bounds([1,10],[(2,3),(4,6)],[1.5,9.5])
    for bounds,requests,samples in [([1,2],[(100,101)],[]),([1,10],[(5,6),(2,3)],[]),([1,2],[],[3])]:
        with pytest.raises(ValueError): audit.process_bounds(bounds,requests,samples)


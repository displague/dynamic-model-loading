import json
from pathlib import Path
import sys
import io
import urllib.error

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continuing_agent as study
import analyze_continuing as audit


def test_sse_requires_complete_token_coverage_and_one_terminal_event():
    final={'stop':True,'tokens':[],'content':'','tokens_predicted':2}
    parts=[{'stop':False,'tokens':[5],'content':'a'},{'stop':False,'tokens':[7],'content':'b'}]
    assert study.parse_sse(parts+[final])['tokens']==[5,7]
    assert study.parse_sse(parts+[final])['content']=='ab'
    for bad in (parts,parts+[final,final],[parts[0],final],[final,parts[0]]):
        with pytest.raises(ValueError): study.parse_sse(bad)


def test_stopping_does_not_waive_early_eos_or_short_limit():
    base={'tokens':[5,6],'tokens_predicted':2,'truncated':False,'stop_type':'limit'}
    study.check_output(base,2)
    with pytest.raises(ValueError): study.check_output(base,3)
    with pytest.raises(ValueError): study.check_output(dict(base,tokens=[151645,6]),2)
    study.check_output(dict(base,tokens=[5,151645],stop_type='eos'),64)
    with pytest.raises(ValueError): study.check_output(dict(base,stop_type='eos'),64)
    with pytest.raises(ValueError): study.check_output(dict(base,tokens=[7,151645],stop_type='eos'),1)


def test_append_closes_message_exactly_once():
    turn={'text':'next'}
    assert study.suffix_text(turn,False).startswith('<|im_end|>\n')
    assert study.suffix_text(turn,True).startswith('\n<|im_start|>user')


def test_fixture_covers_all_practical_continuations_without_preview_selection():
    fixture=study.read(study.ROOT/'data/committed-replay.json')['cases']
    assert len(fixture)==8
    assert sum(len(c['tokens']) for c in fixture.values())==1792
    for case in fixture.values():
        assert len(case['tokens'])==case['max_tokens']
        assert case['stop_type']=='limit' and not study.EOG.intersection(case['tokens'])
        assert len(case['source_files'])==3


def test_replay_keeps_context_placement_precision_but_omits_draft():
    short=study.settings('replay-short','offload')
    long=study.settings('replay-long','offload')
    assert (short['ngl'],short['context'],short['kv'],short['k'],short['threshold'])==(44,4096,'f16',None,32)
    assert (long['ngl'],long['context'],long['kv'],long['k'],long['threshold'])==(38,18432,'q8_0',None,32)


def test_replay_audit_rejects_missing_positions_before_trusting_compact_flags(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'audit_run',lambda _:({'kind':'replay-long'},{}))
    (tmp_path/'rows.jsonl').write_text('',encoding='utf-8')
    with pytest.raises(ValueError,match='sequence coverage'): audit.replay(tmp_path)


def test_agent_audit_rejects_missing_turns(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'audit_run',lambda _:({'kind':'agent-retained'},{}))
    (tmp_path/'rows.jsonl').write_text('{"turn":0}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='sequence coverage'): audit.agent(tmp_path)


def test_duplicate_run_identity_is_not_complete_matrix_coverage():
    with pytest.raises(ValueError,match='matrix identity'):
        audit.matrix_identity('replay-long',{'kind':'replay-short','condition':'offload','repeat':1})
    with pytest.raises(ValueError,match='matrix identity'):
        audit.matrix_identity('offload-r2-retained',{'kind':'agent-retained','condition':'default','repeat':1})


def test_nonfinite_and_unbounded_request_times_fail():
    for end in (float('inf'),float('nan'),-1,1801):
        with pytest.raises(ValueError): audit.timing_interval(0,end)


def test_startup_only_resource_trace_cannot_cover_later_requests(tmp_path):
    sample={'monotonic':1,'gpu':{'used':5},'host':{'available':10}}
    (tmp_path/'resources.jsonl').write_text(json.dumps(sample)+'\n',encoding='utf-8')
    (tmp_path/'completion.json').write_text(json.dumps({'resources':{'samples':1,'gpu_peak':5,'host_available_min':10}}),encoding='utf-8')
    with pytest.raises(ValueError,match='interval coverage'): audit.resource_coverage(tmp_path,[(2,20)])


def test_sse_http_error_keeps_native_body_and_status(tmp_path,monkeypatch):
    error=urllib.error.HTTPError('http://test/completion',500,'failure',{},io.BytesIO(b'{"error":"native"}'))
    def fail(*a,**kw): raise error
    monkeypatch.setattr(study.urllib.request,'urlopen',fail)
    with pytest.raises(urllib.error.HTTPError): study.streaming_request('http://test',{},tmp_path/'request')
    assert (tmp_path/'request.body').read_bytes()==b'{"error":"native"}'
    meta=study.read(tmp_path/'request.http.json')
    assert meta['status']==500 and not meta['complete']


def test_empty_source_identity_manifest_is_rejected(tmp_path):
    run=tmp_path/'replay-long'; run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'kind':'replay-long','condition':'offload','repeat':1,'source_sha256':{}}),encoding='utf-8')
    (run/'completion.json').write_text('{"complete":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='source identity coverage'): audit.audit_run(run)

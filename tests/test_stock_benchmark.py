"""CPU-only tests for telemetry semantics; no pretrained execution."""
import importlib.util
import json
import io
import os
from pathlib import Path
import socket
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip('psutil')

spec=importlib.util.spec_from_file_location('stock_benchmark',Path(__file__).resolve().parents[1]/'scripts/stock_benchmark.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_survival_counts_zero_acceptance_and_short_attempts():
    log='[1] slot accepted 0/4 draft tokens\n[2] slot accepted 2/4 draft tokens\n[3] slot accepted 1/1 draft tokens\n'
    summary=bench.acceptance_summary(bench.parse_acceptance(log),{'draft_n':9,'draft_n_accepted':3})
    assert summary['consistent']
    assert summary['cycles']==3
    assert summary['survival']==[2/3,1/3,0,0]


def test_missing_events_do_not_fabricate_survival_from_aggregate():
    summary=bench.acceptance_summary([],{'draft_n':8,'draft_n_accepted':5})
    assert not summary['consistent'] and summary['survival'] is None


def test_checkpoint_replay_cannot_double_count_a_cycle():
    events=bench.parse_acceptance('accepted 2/4 draft tokens (restore checkpoint)\naccepted 2/2 draft tokens\n')
    result=bench.acceptance_summary(events,{'draft_n':4,'draft_n_accepted':2})
    assert result['checkpoint_replay_present'] and result['survival'] is None


def test_invalid_acceptance_fails():
    with pytest.raises(ValueError):bench.parse_acceptance('accepted 5/4 draft tokens')


def test_greedy_requests_preserve_eos_and_do_not_cache_previous_prompt():
    payload=bench.completion_payload([1,2,3],64)
    assert payload['prompt']==[1,2,3] and payload['return_tokens']
    assert payload['temperature']==0 and payload['samplers']==['temperature']
    assert not payload['ignore_eos'] and not payload['cache_prompt']


def test_stock_command_uses_fixed_residency_and_maximum_draft():
    cmd=bench.server_command('server','target','draft',42,16,8,8099)
    assert cmd[cmd.index('-ngl')+1]=='42'
    assert cmd[cmd.index('--spec-draft-ngl')+1]=='all'
    assert cmd[cmd.index('--spec-draft-n-max')+1]=='8'
    assert cmd[cmd.index('--fit')+1]=='off'
    assert cmd[cmd.index('--verbosity')+1]=='4'  # Stock loader placement logs are trace-level.
    assert '--spec-synth-len' not in cmd and '--ignore-eos' not in cmd


@pytest.mark.parametrize('bad_resource',[False,True])
def test_measurement_preserves_raw_response_before_resource_failure(tmp_path,monkeypatch,bad_resource):
    class Monitor:
        def __init__(self,*args):
            self.rows=[];self.errors=[]
            state['monitor']=self
        def sample(self,bad=False):
            self.rows.append({'monotonic':time.perf_counter(),'gpu':{'used':16000*2**20 if bad else 1000},
                              'host':{'available':4*2**30}})
        def __enter__(self):self.sample();return self
        def __exit__(self,*args):pass
    class Process:
        pid=123
        def __init__(self,*args,**kwargs):
            state['process']=self;self.stopped=False
            kwargs['stdout'].write(b'load_tensors: offloaded 40/65 layers to GPU\n');kwargs['stdout'].flush()
        def poll(self):return None
        def terminate(self):self.stopped=True
        def wait(self,**kwargs):return 0
    state={}
    @contextmanager
    def launch(*args,**kwargs):yield Process(*args,**kwargs)
    def api(base,endpoint,payload=None,**kwargs):
        if endpoint=='/health':return {'status':'ok'}
        if endpoint=='/props':return {}
        if endpoint=='/apply-template':return {'prompt':'test template'}
        if endpoint=='/tokenize':return {'tokens':[1,2,3]}
        assert endpoint=='/completion' and not payload['cache_prompt']
        state['monitor'].sample(bad_resource)
        return {'tokens':[4,5],'timings':{'predicted_ms':10,'predicted_n':2}}
    monkeypatch.setattr(bench,'Resources',Monitor)
    monkeypatch.setattr(bench,'managed_process',launch)
    monkeypatch.setattr(bench.subprocess,'run',lambda *a,**kw:None)
    monkeypatch.setattr(bench.subprocess,'check_output',lambda cmd,**kw:'0'*40 if 'rev-parse' in cmd else '')
    monkeypatch.setattr(bench,'verify_catalog',lambda *a,**kw:{'target':['fixture.gguf']})
    monkeypatch.setattr(bench,'request',api)
    monkeypatch.setattr(bench,'metrics',lambda base,**kwargs:'fixture metrics\n')
    monkeypatch.setattr(bench.time,'sleep',lambda seconds:None)
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1',0));port=reserved.getsockname()[1]
    output=tmp_path/'measurement'
    args=SimpleNamespace(ngl=40,mode='calibration',inputs=None,reference=False,draft=None,
                         models=tmp_path,binary=tmp_path,output=output,order_seed=None,
                         threads=8,k=16,port=port)
    if bad_resource:
        with pytest.raises(RuntimeError,match='resource bound'):bench.measure(args)
    else:bench.measure(args)
    rows=[json.loads(line) for line in (output/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[0]['response']['tokens']==[4,5]
    assert (output/'response-calibration-explanation.json').exists()
    assert (output/'request-calibration-explanation.json').exists()
    assert (output/'inputs.json').exists()
    assert state['process'].stopped
    assert (output/'failure.json').exists()==bad_resource
    assert (output/'completion.json').exists()!=bad_resource


def test_empty_reference_and_wrong_gpu_placement_fail_closed():
    with pytest.raises(ValueError,match='expected cases'):
        bench.validate_inputs({},[{'id':'x','max_tokens':2}])
    with pytest.raises(ValueError,match='GPU placement'):
        bench.validate_placement('warning: GPU unavailable; using CPU',40,'draft05')
    assert bench.validate_placement('offloaded 40/65 layers to GPU\noffloaded 25/25 layers to GPU',40,'draft05')==[(40,65),(25,25)]


def test_unpinned_backend_is_rejected(tmp_path):
    (tmp_path/'ggml-cpu.dll').write_bytes(b'known')
    (tmp_path/'ggml-unpinned.dll').write_bytes(b'unknown')
    catalogue={'models':{},'runtime':{'files':[{'name':'ggml-cpu.dll','bytes':5,'sha256':bench.digest(tmp_path/'ggml-cpu.dll')}]}}
    with pytest.raises(ValueError,match='unpinned'):
        bench.verify_catalog(catalogue,tmp_path,tmp_path,[])


@pytest.mark.parametrize('malformed',[False,True])
def test_http_raw_body_survives_status_or_json_failure(tmp_path,monkeypatch,malformed):
    error=bench.urllib.error.HTTPError('http://127.0.0.1/test',200 if malformed else 500,
                                      'fixture',{},io.BytesIO(b'not valid JSON'))
    def get(*args,**kwargs):
        if malformed:return error
        raise error
    monkeypatch.setattr(bench.urllib.request,'urlopen',get)
    receipt=tmp_path/'http'
    with pytest.raises((bench.urllib.error.HTTPError,json.JSONDecodeError)):
        bench.request('http://127.0.0.1','/test',receipt=receipt)
    assert Path(str(receipt)+'.body').read_bytes()==b'not valid JSON'
    assert json.loads(Path(str(receipt)+'.http.json').read_text(encoding='utf-8'))['status']==(200 if malformed else 500)


@pytest.mark.skipif(os.name!='nt',reason='Windows native job lifecycle')
def test_job_close_kills_descendant_without_runner_finally(tmp_path):
    program="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    with bench.managed_process([sys.executable,'-c',program],stdout=bench.subprocess.PIPE,text=True) as proc:
        child=int(proc.stdout.readline())
        assert bench.psutil.pid_exists(child)
        # Kill the Python owner directly; its child still belongs to our enclosing job.
        proc.kill();proc.wait()
    for _ in range(100):
        if not bench.psutil.pid_exists(child):break
        time.sleep(0.01)
    assert not bench.psutil.pid_exists(child)


def test_runner_error_takes_precedence_over_allocation_warning():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import stock_study as study
    warning='failed to allocate pinned buffer; continuing with pageable memory'
    assert not study.is_resource_failure({'type':'ValueError','message':'wrong inputs'},warning,[])
    assert not study.is_resource_failure({'type':'HTTPError','message':'500'},warning,[])
    assert study.is_resource_failure({'type':'RuntimeError','message':'server exited 1'},'CUDA: out of memory',[])
    assert not study.is_resource_failure({'type':'RuntimeError','message':'server exited 1'},'CUDA: out of memory',[{'error':'NVML error 15','error_type':'RuntimeError'}])
    assert study.is_resource_failure({'type':'RuntimeError','message':'server exited 1'},'CUDA: out of memory',[{'error':'native exited','error_type':'NoSuchProcess'}])


def test_stale_calibration_cannot_launch_evaluation(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import stock_study as study
    path=tmp_path/'selection.json'
    path.write_text(json.dumps({'source':'old','evaluation_started':False}),encoding='utf-8')
    monkeypatch.setattr(study.subprocess,'check_output',lambda *a,**k:'current')
    with pytest.raises(ValueError,match='current preregistered source'):
        study.validate_selection(path,tmp_path)


def test_partial_http_body_and_status_survive_disconnect(tmp_path,monkeypatch):
    class Partial:
        status=200
        headers={'Content-Length':'100'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):raise bench.http.client.IncompleteRead(b'{"tokens":[',89)
    monkeypatch.setattr(bench.urllib.request,'urlopen',lambda *a,**kw:Partial())
    receipt=tmp_path/'partial'
    with pytest.raises(bench.http.client.IncompleteRead):
        bench.request('http://127.0.0.1','/completion',receipt=receipt)
    assert Path(str(receipt)+'.body').read_bytes()==b'{"tokens":['
    metadata=json.loads(Path(str(receipt)+'.http.json').read_text(encoding='utf-8'))
    assert metadata['status']==200 and not metadata['complete'] and metadata['bytes']==11


def test_selection_must_match_configuration_in_ledger(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import stock_study as study
    root=tmp_path/'source';(root/'configs').mkdir(parents=True);(root/'data').mkdir()
    for relative in ['configs/stock-speculation-artifacts.json','data/stock-speculation-workloads.json']:
        (root/relative).write_text('{}',encoding='utf-8')
    evidence=tmp_path/'evidence';run=evidence/'target';run.mkdir(parents=True)
    manifest={'head':'current','mode':'calibration','ngl':40,'threads':8,'draft':None,'k':None,
              'catalog_sha256':bench.digest(root/'configs/stock-speculation-artifacts.json'),
              'workloads_sha256':bench.digest(root/'data/stock-speculation-workloads.json')}
    bench.write_json(run/'manifest.json',manifest);bench.write_json(run/'completion.json',{})
    rows=[{'resource_pass':True,'warmup':i==0,'sampled_gpu_peak':1000,
           'response':{'timings':{'predicted_ms':10}}} for i in range(3)]
    (run/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    ledger=study.summarize(run)|{'ngl':40,'threads':8,'draft':None,'k':None,'returncode':0}
    (evidence/'driver-ledger.jsonl').write_text(json.dumps(ledger)+'\n',encoding='utf-8')
    bench.write_json(evidence/'driver-complete.json',{})
    chosen={'feasible':True,'ngl':40,'threads':8,'draft':None,'calibration_result':str(run),'median_decode_ms':10}
    selection={'source':'current','evaluation_started':False,'driver_ledger_sha256':bench.digest(evidence/'driver-ledger.jsonl'),
               'selected':{'target':chosen,**{k:{'feasible':False} for k in ['draft05','draft15','draft32']}}}
    path=evidence/'selection.json';bench.write_json(path,selection)
    monkeypatch.setattr(study.subprocess,'check_output',lambda *a,**kw:'current')
    study.validate_selection(path,root)
    chosen['ngl']=41;manifest['ngl']=41
    bench.write_json(path,selection);bench.write_json(run/'manifest.json',manifest)
    with pytest.raises(ValueError,match='unsupported'):
        study.validate_selection(path,root)

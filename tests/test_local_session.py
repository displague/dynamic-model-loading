from pathlib import Path
import json
import io
from http.client import HTTPResponse
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import local_session as local
import local_session_smoke as smoke
import attention_agent


def test_interactive_measured_flags_preserve_measured_constructor():
    checked = {'target': ['target.gguf'], 'draft05': ['draft.gguf']}
    cfg = attention_agent.settings('attention')
    reference = local.placement.command(Path('bin'), checked, cfg, 8080)
    cmd = local.server_command(Path('bin'), checked, 'measured', 8080)
    assert cmd[:len(reference)] == reference
    assert cmd[len(reference):] == ['--alias', local.ALIAS, '--jinja', '--temp', '0', '--seed', '0']


def test_named_controls_do_not_accidentally_keep_draft_or_host_override():
    checked = {'target': ['target.gguf'], 'draft05': ['draft.gguf']}
    cmd = local.server_command(Path('bin'), checked, 'target-only', 8080)
    assert '-md' not in cmd and '--n-cpu-ffn' not in cmd
    assert cmd[cmd.index('--spec-type')+1] == 'none'
    cfg = local.configuration('stock-speculative')
    assert (cfg['threshold'], cfg['k'], cfg['ngl']) == (32, 4, 38)
    with pytest.raises(ValueError): local.configuration('32k')


def test_child_settings_do_not_inherit_cloud_credentials_or_mutate_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(local, 'executable', lambda name: name+'.exe')
    parent = {'PATH': 'x', 'ANTHROPIC_API_KEY': 'secret', 'OPENAI_API_KEY': 'secret2',
              'CODEX_HOME': 'real-user-home', 'CLAUDE_CODE_USE_BEDROCK': '1'}
    for client in ['codex', 'claude']:
        cmd, env, effective = local.client_settings(client, 'http://127.0.0.1:8080', tmp_path, tmp_path, parent)
        assert 'secret' not in str(env) and 'secret2' not in str(env)
        assert 'CLAUDE_CODE_USE_BEDROCK' not in env
        assert 'bypassPermissions' not in cmd and '--dangerously-bypass-approvals-and-sandbox' not in cmd
    assert parent['CODEX_HOME'] == 'real-user-home' and parent['ANTHROPIC_API_KEY'] == 'secret'
    with pytest.raises(ValueError):
        local.client_settings('claude', 'https://external.invalid', tmp_path, tmp_path, parent)


def test_file_success_requires_exact_semantics_not_a_claim_or_extra_statements(tmp_path):
    p = tmp_path/'calculator.py'
    p.write_text(smoke.ORIGINAL)
    assert not smoke.correct_file(p)
    p.write_text(smoke.EXPECTED)
    assert smoke.correct_file(p)
    p.write_text(smoke.EXPECTED+'print("also did extra work")\n')
    assert not smoke.correct_file(p)


def test_stream_parser_distinguishes_actual_text_deltas():
    raw = 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"READY_LOCAL"}\n\ndata: [DONE]\n'
    assert smoke.text_output('responses', raw, True) == 'READY_LOCAL'
    raw = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"PROBE_4821"}}\n\n'
    assert smoke.text_output('messages', raw, True) == 'PROBE_4821'


@pytest.mark.parametrize('kind', ['chat', 'responses', 'messages'])
def test_expected_text_does_not_pass_an_errored_or_unfinished_stream(kind):
    assert not smoke.stream_complete(kind, 'data: {"type":"error","error":{"message":"failed"}}\n\n')
    assert not smoke.stream_complete(kind, 'data: {"type":"response.output_text.delta","delta":"READY_LOCAL"}\n\n')


def test_stream_completion_requires_the_routes_terminal_event():
    def sse(*es): return ''.join('data: '+json.dumps(e)+'\n\n' for e in es)
    chat = sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})+'data: [DONE]\n'
    responses = sse({'type': 'response.completed', 'response': {'status': 'completed'}})
    messages = sse({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}}, {'type': 'message_stop'})
    for kind, raw in [('chat', chat), ('responses', responses), ('messages', messages)]:
        assert smoke.stream_complete(kind, raw)
        assert not smoke.stream_complete(kind, raw+sse({'type': 'error', 'error': 'later failure'}))


def test_interrupted_http_preserves_status_and_partial_body(tmp_path, monkeypatch):
    class Response:
        status = 200
        headers = {'content-type': 'text/event-stream'}
        count = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read1(self, n):
            self.count += 1
            if self.count == 1: return b'data: first\n\n'
            raise smoke.IncompleteRead(b'partial', 8)
    monkeypatch.setattr(smoke.urllib.request, 'urlopen', lambda *a, **kw: Response())
    prefix = tmp_path/'response'
    with pytest.raises(smoke.IncompleteRead):
        smoke.http('http://127.0.0.1:8080', '/v1/responses', {'stream': True}, prefix)
    assert prefix.with_suffix('.response.txt').read_bytes() == b'data: first\n\npartial'
    assert json.loads(prefix.with_suffix('.http.json').read_text())['status'] == 200
    assert prefix.with_suffix('.transport-error.json').exists()


def test_aggregate_verdict_rejects_missing_failed_and_server_error_rows():
    good = {p: {'chat': {'text_stream': True}} for p in local.PROFILES}
    good['measured'] = {k: {'text_stream': True, 'tool_call': True, 'tool_result': True}
                        for k in ['chat', 'responses', 'messages']}
    good['measured'].update(codex={'passed': True}, claude={'passed': True})
    assert smoke.all_passed(good)
    good['target-only']['server_error'] = 'native exit'
    assert not smoke.all_passed(good)
    assert not smoke.all_passed({})


def test_real_httpresponse_premature_content_length_eof_is_not_a_pass(tmp_path, monkeypatch):
    payload = b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
    class Socket:
        def makefile(self, *args):
            return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(payload)+20).encode()+b'\r\n\r\n'+payload)
    response = HTTPResponse(Socket())
    response.begin()
    monkeypatch.setattr(smoke.urllib.request, 'urlopen', lambda *a, **kw: response)
    prefix = tmp_path/'truncated'
    with pytest.raises(smoke.IncompleteRead):
        smoke.http('http://127.0.0.1:8080', '/v1/responses', {'stream': True}, prefix)
    assert prefix.with_suffix('.response.txt').read_bytes() == payload
    assert json.loads(prefix.with_suffix('.transport-error.json').read_text())['type'] == 'IncompleteRead'


def test_claude_diagnostic_string_is_not_an_assistant_message():
    rows = [{'type': 'system', 'message': 'permission denied'},
            {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'name': 'Read'}]}},
            {'type': 'result', 'result': 'not a tool'}]
    assert smoke.tool_events('claude', rows) == [{'type': 'tool_use', 'name': 'Read'}]


def test_exec_sandbox_flag_belongs_to_the_exec_subcommand(tmp_path, monkeypatch):
    monkeypatch.setattr(local, 'executable', lambda name: name+'.exe')
    cmd, _, _ = local.client_settings('codex', 'http://127.0.0.1:8080', tmp_path, tmp_path, {},
                                     prompt='edit the fixture', result=tmp_path/'result.txt')
    assert cmd[cmd.index('exec')+1:cmd.index('exec')+3] == ['-s', 'workspace-write']
    assert '-s' not in cmd[:cmd.index('exec')]

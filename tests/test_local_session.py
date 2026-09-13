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


def test_small_context_manual_mode_keeps_true_window_and_manual_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(local, 'executable', lambda name: name+'.exe')
    parent = {'DISABLE_COMPACT': '1', 'disable_auto_compact': '1',
              'CLAUDE_CODE_MAX_CONTEXT_TOKENS': '200000',
              'CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE': '200000'}
    cmd, env, _ = local.client_settings('claude', 'http://127.0.0.1:8080', tmp_path, tmp_path, parent)
    assert '--disable-slash-commands' not in cmd
    assert '--safe-mode' in cmd and '--bare' not in cmd and '--restricted' in cmd
    assert env['DISABLE_AUTO_COMPACT'] == '1'
    assert 'DISABLE_COMPACT' not in env and 'disable_auto_compact' not in env
    assert env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == '18432'
    assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '2048'
    assert env['CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS'] == '2048'
    assert 'CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE' not in env
    _, auto, _ = local.client_settings('claude', 'http://127.0.0.1:8080', tmp_path, tmp_path, parent,
                                       claude_compaction='auto')
    assert 'DISABLE_AUTO_COMPACT' not in auto and 'DISABLE_COMPACT' not in auto
    assert parent['DISABLE_COMPACT'] == '1'
    with pytest.raises(ValueError):
        local.client_settings('claude', 'http://127.0.0.1:8080', tmp_path, tmp_path, parent,
                              claude_compaction='pretend-larger')


def test_scripted_read_evidence_rejects_denials_and_deduplicates_history():
    from claude_context_check import successful_fixture_reads
    def req(*blocks):
        return {'body': {'messages': [{'role': 'user', 'content': list(blocks)}]}}
    denied = {'type': 'tool_result', 'tool_use_id': 'tool_1', 'is_error': True,
              'content': 'Permission denied: A small fixture.'}
    good = dict(denied, is_error=False, content='1: A small fixture.')
    assert successful_fixture_reads([req(denied)]) == set()
    assert successful_fixture_reads([req(good), req(good)]) == {'tool_1'}
    assert successful_fixture_reads([req(dict(good, content='unrelated text'))]) == set()


def test_trimmed_full_read_cannot_substitute_for_denied_targeted_read():
    from claude_context_check import read_budget_evidence
    full = {'type': 'tool_result', 'tool_use_id': 'tool_1', 'content': 'row 0: a\nrow 1: b'}
    target = dict(full, tool_use_id='tool_2', is_error=True, content='Permission denied')
    rows = [{'type': 'user', 'message': {'content': [full]}, 'tool_use_result':
             {'file': {'truncatedByTokenCap': True, 'numLines': 2, 'totalLines': 1501}}}]
    def requests(t):
        return [{'body': {'messages': [{'content': [full, t]}]}}]
    _, trimmed, bounded = read_budget_evidence(rows, requests(target))
    assert trimmed and not bounded
    _, _, bounded = read_budget_evidence(rows, requests(dict(full, tool_use_id='tool_2')))
    assert bounded


def test_compactor_can_retain_tools_without_turning_tool_output_into_instructions():
    from claude_context_check import compaction_request
    instruction = 'CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Produce a summary.'
    body = {'tools': [{'name': 'Read'}], 'messages': [{'role': 'user', 'content':
            [{'type': 'text', 'text': instruction}]}]}
    assert compaction_request(body)
    body['messages'].append({'role': 'system', 'content': [{'type': 'text', 'text': '<total_tokens>15000000 tokens left</total_tokens>'}]})
    assert compaction_request(body)
    body['messages'].append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'A later reply'}]})
    assert not compaction_request(body)
    body['messages'] = body['messages'][:1]
    body['messages'][0]['content'] = [{'type': 'tool_result', 'content': instruction}]
    assert not compaction_request(body)


def test_file_tool_verdict_rejects_denied_reads_and_missing_write():
    from claude_tools_check import verdict, ORIGINAL, EXPECTED
    import copy
    blocks = [{'type': 'tool_result', 'tool_use_id': f'tool_{n}', 'content': 'ok'} for n in range(1, 9)]
    blocks[0]['content'] = ORIGINAL
    blocks[6]['content'] = '1: created\n'
    for n, text in [(2, 'file already exists'), (3, 'Found 2 matches'), (4, 'String to replace not found')]:
        blocks[n-1].update(is_error=True, content=text)
    requests = [{'body': {'tools': [{'name': name} for name in ['Read', 'Edit', 'Write']],
                         'messages': [{'content': blocks}]}, 'files': {'existing.txt': ORIGINAL}} for _ in range(9)]
    requests[6]['files'] = {'existing.txt': EXPECTED, 'new.txt': 'created\n'}
    final = {'existing.txt': EXPECTED, 'new.txt': 'updated\n'}
    assert verdict(requests, False, final, 0)['passed']
    bad = copy.deepcopy(requests)
    for r in bad:
        r['body']['messages'][0]['content'][0]['is_error'] = True
    assert not verdict(bad, False, final, 0)['passed']
    bad = copy.deepcopy(requests)
    bad[0]['body']['tools'].pop()
    assert not verdict(bad, False, final, 0)['passed']
    bad = copy.deepcopy(requests)
    bad[3]['files']['existing.txt'] = 'damaged'
    assert not verdict(bad, False, final, 0)['passed']
    assert not verdict([], False, final, 0)['passed']
    bad = copy.deepcopy(requests)
    bad[6]['files']['new.txt'] = ''
    assert not verdict(bad, False, final, 0)['passed']
    bad = copy.deepcopy(requests)
    for r in bad:
        r['body']['messages'][0]['content'][6]['content'] = 'Empty file'
    assert not verdict(bad, False, final, 0)['passed']

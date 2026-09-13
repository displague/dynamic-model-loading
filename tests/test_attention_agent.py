from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import attention_agent as study
import analyze_attention_agent as audit


def test_explicit_conversation_adapter_does_not_launch_historical_configuration(tmp_path):
    cfg=study.settings('attention')
    @contextmanager
    def checked_server(args,actual):
        assert actual==cfg
        assert (actual['ngl'],actual['cold_ffns'],actual['threshold'],actual['k'])==(65,32,2,16)
        raise RuntimeError('checked before native launch')
        yield
    args=SimpleNamespace(kind='agent-retained',condition='attention',output=tmp_path)
    with pytest.raises(RuntimeError,match='checked before native launch'):
        study.conversation.agent(args,configuration=cfg,server_factory=checked_server)


def test_registered_pairs_and_no_extra_context_or_confidence_conditions():
    order=study.matrix()
    assert len(order)==8 and len(set(order))==8
    for i in range(0,8,2):
        assert order[i][:2]==order[i+1][:2]
        assert (order[i][2],order[i+1][2])==('retained','reset')
    for c in ['whole','attention']:
        cfg=study.settings(c)
        assert (cfg['context'],cfg['kv'],cfg['scheduler_debug'])==(18432,'q8_0',0)
    with pytest.raises(ValueError): study.settings('32k')


def test_incomplete_matrix_is_rejected_before_partial_scoring(tmp_path):
    (tmp_path/'attention-r1-retained').mkdir()
    with pytest.raises(ValueError,match='exact native matrix'): audit.analyze(tmp_path)


def test_explicit_audit_callback_still_checks_conversation_coverage(tmp_path):
    (tmp_path/'rows.jsonl').write_text('{"turn":0}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='sequence coverage'):
        audit.prior.agent(tmp_path,audit=lambda _:({'kind':'agent-retained'},{}))

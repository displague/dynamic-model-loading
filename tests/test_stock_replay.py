import importlib.util
from pathlib import Path
import sys
import json
import socket
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import stock_replay as replay


def test_replays_deduplicate_prefixes_without_discarding_origins():
    config = {'draft': 'draft05', 'ngl': 44, 'threads': 16, 'k': 8}
    item = {'configuration': config, 'tokens': [1, 2, 3], 'run': 'repeat1-draft05-k8',
            'id': 'code', 'first_difference': {'reference_next': 8, 'candidate_next': 9}}
    second = dict(item, run='repeat2-draft05-k8')
    analysis = {'first_divergence_diagnostics': [item, second]}
    selection = {'selected': {'target': {'ngl': 45, 'threads': 16}}}
    plan = replay.replay_plan(analysis, selection)
    assert len(plan) == 3
    assert {g['configuration']['ngl'] for g in plan.values()} == {44, 45}
    for group in plan.values():
        assert len(group['cases']) == 1
        case = next(iter(group['cases'].values()))
        assert case['tokens'] == [1, 2, 3] and len(case['origins']) == 2


def test_replays_reject_prefixes_that_leave_no_prediction_slot():
    item = {'tokens': [1]*4096}
    with pytest.raises(ValueError, match='common-prefix'):
        replay.replay_plan({'first_divergence_diagnostics': [item]}, {'selected': {'target': {}}})


def test_diagnostic_resource_failure_remains_visible():
    resources = SimpleNamespace(rows=[{'gpu': {'used': 15001*2**20},
                                       'host': {'available': 4*2**30}}], errors=[])
    with pytest.raises(RuntimeError, match='resource bound'):
        replay.check_resources(resources)


def test_changed_candidate_or_reference_receipt_rejects_stale_analysis(tmp_path):
    files = ['reference-sustained/rows.jsonl', 'reference-smoke/inputs.json', 'repeat1-target/rows.jsonl']
    for name in files:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('original', encoding='utf-8')
    data={'evidence_sha256': {name: replay.bench.digest(tmp_path/name) for name in files}}
    replay.validate_analysis_receipts(data, tmp_path)
    for name in files:
        (tmp_path/name).write_text('changed', encoding='utf-8')
        with pytest.raises(ValueError, match='receipt changed'):
            replay.validate_analysis_receipts(data, tmp_path)
        (tmp_path/name).write_text('original', encoding='utf-8')


def test_diagnostic_groups_never_reuse_a_port_or_wrap_the_range():
    groups = dict.fromkeys(['target', 'matched', 'draft'])
    ports = replay.replay_ports(groups, 65533)
    assert list(ports) == list(groups)
    assert len(set(ports.values())) == len(groups)
    assert min(ports.values()) == 65533 and max(ports.values()) == 65535
    for first in [0, -1, 65534, 65536]:
        with pytest.raises(ValueError, match='port range'):
            replay.replay_ports(groups, first)


def test_occupied_port_preserves_failure_receipt(tmp_path):
    with socket.socket() as held:
        held.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        held.bind(('127.0.0.1', 0))
        port = held.getsockname()[1]
        with pytest.raises(OSError):
            replay.check_port(port, tmp_path)
    receipt = json.loads((tmp_path/'failure.json').read_text())
    assert receipt['phase'] == 'port-probe' and receipt['port'] == port

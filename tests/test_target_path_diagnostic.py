from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import target_path_diagnostic as diagnostic
from analyze_target_path import compare, analyze


def response(entries, chosen=1):
    return {'tokens': [chosen], 'tokens_predicted': 1,
            'completion_probabilities': [{'top_logprobs': entries}]}


def test_encoded_probability_vector_retains_ids_and_reports_a_wrong_argmax(monkeypatch):
    monkeypatch.setattr(diagnostic, 'VOCAB', 4)
    entries = [{'id': 2, 'logprob': -4.0}, {'id': 0, 'logprob': -2.0},
               {'id': 3, 'logprob': float(np.finfo(np.float32).min)}, {'id': 1, 'logprob': -0.3}]
    vector, summary = diagnostic.distribution(response(entries))
    assert vector[1] == np.float32(-0.3)
    assert summary['encoded_argmax_matches']
    assert summary['top_two_ids'] == [1, 0]
    assert summary['zero_probability_sentinels'] == 1
    _, mismatch = diagnostic.distribution(response(entries, chosen=0))
    assert not mismatch['encoded_argmax_matches']


def test_full_vocabulary_claim_rejects_duplicate_or_missing_ids(monkeypatch):
    monkeypatch.setattr(diagnostic, 'VOCAB', 3)
    with pytest.raises(ValueError, match='coverage'):
        diagnostic.distribution(response([{'id': 0, 'logprob': -1}, {'id': 0, 'logprob': -2},
                                          {'id': 2, 'logprob': -3}]))


def test_factor_comparisons_change_one_declared_setting():
    base = diagnostic.FACTORS['incremental-reference']
    for key in ('threads', 'placement', 'prefill'):
        assert sum(a != b for a, b in zip(base, diagnostic.FACTORS[key])) == 1
    assert sum(a != b for a, b in zip(diagnostic.FACTORS['prefill'], diagnostic.FACTORS['microbatch'])) == 1


def test_probability_margin_bound_and_near_tie_are_distinguished():
    left = np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    right = np.log(np.array([0.65, 0.25, 0.1], dtype=np.float32))
    stable = compare(left, right, 0, 0)
    assert stable['margin_exceeds_twice_distance'] and not stable['token_flip']
    tied = np.log(np.array([0.451, 0.449, 0.1], dtype=np.float32))
    flipped = np.log(np.array([0.448, 0.452, 0.1], dtype=np.float32))
    changed = compare(tied, flipped, 0, 1)
    assert changed['token_flip'] and not changed['argmax_bound_violation']
    assert changed['raw_logit_linf'] is None


def test_missing_intermediate_requests_cannot_pass_on_final_case_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'FACTORS', {'incremental-reference': (45, 16, 256, True)})
    fixture = tmp_path/'fixture.json'
    fixture.write_text(json.dumps({'cases': {'case': {'tokens': [1, 2, 3], 'original_prompt_tokens': 2}}}), encoding='utf-8')
    run = tmp_path/'incremental-reference'
    run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'factor': 'incremental-reference',
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest()}), encoding='utf-8')
    (run/'resources.jsonl').write_text(json.dumps({'gpu': {'used': 100}, 'host': {'available': 3*2**30}})+'\n', encoding='utf-8')
    (run/'completion.json').write_text('{"complete": true}', encoding='utf-8')
    (run/'cases.jsonl').write_text('{"case": "case"}\n', encoding='utf-8')
    (run/'requests.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='sequence coverage'):
        analyze(tmp_path, fixture)

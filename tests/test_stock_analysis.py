import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('analyze_stock', Path(__file__).resolve().parents[1]/'scripts/analyze_stock.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(tokens, seconds, decode_ms, events=()):
    return {'request': {'prompt': [5, 6], 'n_predict': 8}, 'seconds': seconds,
            'response': {'tokens': tokens, 'stop_type': 'limit', 'stopping_word': '', 'truncated': False,
                         'timings': {'predicted_n': len(tokens), 'predicted_ms': decode_ms, 'prompt_ms': 100,
                                     'draft_n': sum(e['attempted'] for e in events),
                                     'draft_n_accepted': sum(e['accepted'] for e in events)}},
            'acceptance': events, 'acceptance_summary': {'consistent': True},
            'sampled_gpu_peak': 100, 'sampled_host_available_min': 1000}


def test_time_ratios_are_weighted_and_first_token_is_not_a_decode_step():
    result = analysis.aggregate([row([1, 2, 3], 1, 900), row([1, 2], 3, 2900)])
    assert result['emitted_per_request_second'] == 1.25
    assert result['emitted_per_decode_second'] == pytest.approx(5/3.8)
    assert result['native_decode_steps_per_second'] == pytest.approx(3/3.8)
    assert result['native_decode_steps'] == 3


def test_first_divergence_keeps_reference_prefix_and_stop_contract():
    ref = row([1, 2, 3], 1, 900)
    candidate = row([1, 8, 3], 1, 900)
    result = analysis.compare(candidate, ref)
    assert not result['token_match'] and result['common_generated_prefix'] == 1
    assert result['reference_next'] == 2 and result['candidate_next'] == 8
    candidate = copy.deepcopy(ref)
    candidate['response']['stop_type'] = 'eos'
    result = analysis.compare(candidate, ref)
    assert result['token_match'] and not result['stop_match']


def test_mismatched_input_cannot_be_called_output_drift():
    ref = row([1, 2], 1, 900)
    candidate = copy.deepcopy(ref)
    candidate['request']['prompt'].append(99)
    with pytest.raises(ValueError, match='request differs'):
        analysis.compare(candidate, ref)


def test_survival_uses_real_cycles_and_withholds_unreconciled_curve():
    events = [{'accepted': 2, 'attempted': 4, 'checkpoint_replay': False},
              {'accepted': 0, 'attempted': 2, 'checkpoint_replay': False}]
    sample = row([1, 2, 3, 4, 5], 1, 900, events)
    result = analysis.aggregate([sample])
    assert result['accepted_prefix_survival'] == [0.5, 0.5, 0, 0]
    assert result['attempted_length_counts'] == {'2': 1, '4': 1}
    assert result['emitted_decode_steps_per_cycle'] == 2
    sample['acceptance_summary']['consistent'] = False
    assert analysis.aggregate([sample])['accepted_prefix_survival'] is None


def test_incomplete_evaluation_is_not_a_final_frontier(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        analysis.analyze(tmp_path)


def test_frozen_selection_requires_all_drafts_lengths_and_matched_controls():
    selected = {'target': {'feasible': True, 'ngl': 45, 'threads': 16},
                'draft05': {'feasible': True, 'ngl': 44, 'threads': 24},
                'draft15': {'feasible': False}, 'draft32': {'feasible': False}}
    schedule = analysis.evaluation_schedule(selected, 'selection-digest')
    assert {c['name'] for c in schedule['configs']} == {
        'target', 'draft05-k4', 'draft05-k8', 'draft05-k16', 'matched-ngl44-t24'}
    analysis.validate_schedule(schedule, {'selected': selected}, 'selection-digest')
    schedule['configs'] = schedule['configs'][:1]
    with pytest.raises(ValueError, match='matrix'):
        analysis.validate_schedule(schedule, {'selected': selected}, 'selection-digest')


def test_reference_requires_selected_target_only_placement_and_workload():
    target = {'ngl': 45, 'threads': 16}
    manifest = {'draft': None, 'ngl': 45, 'threads': 16, 'k': None, 'mode': 'sustained',
                'order_seed': None, 'inputs_sha256': None}
    analysis.validate_reference(manifest, target, 'sustained')
    for changes in [{'draft': 'draft32'}, {'ngl': 11}, {'threads': 8}, {'mode': 'smoke'}]:
        with pytest.raises(ValueError, match='reference'):
            analysis.validate_reference(manifest|changes, target, 'sustained')

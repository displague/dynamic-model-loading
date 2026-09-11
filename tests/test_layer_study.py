import argparse
import json
from pathlib import Path

import pytest
import torch

from dynamic_model_loading import experiment, layer_plot, layer_study
from test_experiment import tiny_run


@pytest.fixture
def prepared_study(tiny_run, monkeypatch):
    config_path = Path(tiny_run.config)
    config = json.loads(config_path.read_text())
    config.update(layouts=["popularity", "coactivation"], group_widths=[4],
                  coactivation_reservoir=3, coactivation_sketch_dim=4,
                  coactivation_group_width=4, coactivation_iterations=3)
    config_path.write_text(json.dumps(config))
    Path(tiny_run.corpus).with_name("provenance.json").write_text(json.dumps({
        "corpus_sha256": experiment.digest(Path(tiny_run.corpus))}))
    experiment.run(tiny_run)
    monkeypatch.setattr(layer_study, "snapshot_download", experiment.snapshot_download)
    cfg = {"purpose": "test_fixture", "parent_results_sha256": experiment.digest(Path(tiny_run.output) / "results.jsonl"),
           "parent_manifest_sha256": experiment.digest(Path(tiny_run.output) / "manifest.json"),
           "parent_summary_sha256": experiment.digest(Path(tiny_run.output) / "summary.json"),
           "expected_torch": str(torch.__version__), "group_width": 4, "keep_fraction": 0.5,
           "layouts": ["popularity", "coactivation"], "blocks": [[0], [1]],
           "protocol": "docs/layer-study-protocol.md"}
    study_config = config_path.with_name("study-config.json")
    study_config.write_text(json.dumps(cfg))
    return argparse.Namespace(parent=tiny_run.output, config=str(study_config),
                              output=str(config_path.parent / "study"), device="cpu")


def test_layer_interventions_account_for_unmasked_layers(prepared_study):
    summary = layer_study.run(prepared_study)
    assert summary["status"] == "layer_study_completed"
    assert len(summary["probes"]) == 10
    for row in summary["probes"]:
        expected = 0.5 if row["condition_kind"] == "all" else 0.75
        assert row["hypothetical_whole_ffn_weight_fraction"] == expected
    assert summary["layer_analysis"]["popularity"]["joint_kl_minus_parent"] == 0
    output = Path(prepared_study.output)
    assert layer_plot.validate_results(output) == summary
    for alteration in ("drop", "relabel", "volume"):
        changed = json.loads(json.dumps(summary))
        if alteration == "drop":
            changed["probes"].pop(0)
        elif alteration == "relabel":
            changed["probes"][0]["layers"] = [999]
        else:
            changed["probes"][0]["hypothetical_whole_ffn_weight_fraction"] = 0.01
        (output / "summary.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            layer_plot.validate_results(output)
    (output / "summary.json").write_text(json.dumps(summary))
    raw_path = Path(prepared_study.output) / "results.jsonl"
    before = raw_path.read_bytes()
    with pytest.raises(FileExistsError):
        layer_study.run(prepared_study)
    assert raw_path.read_bytes() == before


def test_numerical_failure_blocks_every_omission(prepared_study, monkeypatch):
    monkeypatch.setattr(layer_study, "grouped_forward", lambda m, x, w: m(x) * 0)
    summary = layer_study.run(prepared_study)
    assert summary["status"] == "correctness_gate_failed"
    assert not summary["probes"]
    assert "omission_document" not in (Path(prepared_study.output) / "results.jsonl").read_text()


def test_parent_tampering_is_rejected_before_measurement(prepared_study):
    p = Path(prepared_study.parent) / "corpus.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        layer_study.run(prepared_study)
    assert not Path(prepared_study.output).exists()


@pytest.mark.parametrize("filename", ["manifest.json", "summary.json"])
def test_parent_metadata_is_anchored_before_model_loading(prepared_study, filename):
    p = Path(prepared_study.parent) / filename
    data = json.loads(p.read_text())
    if filename == "manifest.json":
        data["checkpoint"]["files"] = {}
    else:
        data["probes"][0]["mean_kl_dense_to_candidate"] = 12345
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        layer_study.run(prepared_study)
    assert not Path(prepared_study.output).exists()


def test_fixed_blocks_must_partition_every_layer_once():
    with pytest.raises(ValueError, match="partition"):
        layer_study.conditions(4, [[0, 1], [1, 2, 3]])
    with pytest.raises(ValueError, match="partition"):
        layer_study.conditions(4, [[0, 1], [3]])


def test_interrupt_preserves_raw_failure_receipt(prepared_study, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("test interruption")
    monkeypatch.setattr(layer_study, "snapshot_download", interrupt)
    with pytest.raises(KeyboardInterrupt):
        layer_study.run(prepared_study)
    root = Path(prepared_study.output)
    assert json.loads((root / "failure.json").read_text())["error_type"] == "KeyboardInterrupt"
    assert json.loads((root / "results.jsonl").read_text().splitlines()[-1])["kind"] == "error"

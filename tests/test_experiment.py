import argparse
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from dynamic_model_loading import experiment


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1))
    model.save_pretrained(checkpoint)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(checkpoint)
    monkeypatch.setattr(experiment, "snapshot_download", lambda *a, **kw: str(checkpoint))
    config = json.loads(Path("configs/smoke.json").read_text())
    config.update(dtype="float32", group_widths=[1, 4], keep_fractions=[0.5],
                  reconstruction_group_width=7, timing_repeats=1, decode_tokens=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('\n'.join(json.dumps(dict(id=name, split=split, domain="fixture", text=text))
                              for name, split, text in [("cal", "calibration", "a b c d"),
                                                        ("diag", "diagnostic", "d c b a")]))
    return argparse.Namespace(config=str(config_path), corpus=str(corpus), output=str(tmp_path / "run"),
                              device="cpu", allow_download=False, diagnostics=True)


def test_end_to_end_run_and_exclusive_receipts(tiny_run):
    result = experiment.run(tiny_run)
    assert result["status"] == "smoke_diagnostics_completed"
    assert len(result["probes"]) == 6
    output = Path(tiny_run.output)
    raw = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len([r for r in raw if r["kind"] == "hindsight_document"]) == 6
    assert all(r["passed"] for r in raw if "passed" in r)
    before = (output / "results.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        experiment.run(tiny_run)
    assert (output / "results.jsonl").read_bytes() == before


def test_failed_correctness_blocks_probes(tiny_run, monkeypatch):
    monkeypatch.setattr(experiment, "grouped_forward", lambda m, x, w: m(x) * 0)
    result = experiment.run(tiny_run)
    assert result["status"] == "correctness_gate_failed"
    assert not result["probes"]
    raw = (Path(tiny_run.output) / "results.jsonl").read_text()
    assert "hindsight_document" not in raw


def test_duplicate_corpus_is_rejected(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    row = dict(id="x", split="calibration", domain="x", text="same text")
    path.write_text(json.dumps(row) + '\n' + json.dumps({**row, "id": "y", "split": "diagnostic"}))
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.load_corpus(path)


def test_coactivation_pilot_with_disk_references(tiny_run):
    path = Path(tiny_run.config)
    config = json.loads(path.read_text())
    config.update(layouts=["native", "random", "popularity", "coactivation"],
                  coactivation_reservoir=3, coactivation_sketch_dim=4,
                  coactivation_group_width=4, coactivation_iterations=3,
                  spill_reference_logits=True)
    path.write_text(json.dumps(config))
    summary = experiment.run(tiny_run)
    assert summary["status"] == "packing_pilot_completed"
    assert len(summary["probes"]) == 8
    output = Path(tiny_run.output)
    assert len(list((output / "reference_logits").glob("*.safetensors"))) == 1
    assert (output / "source" / "packing.py").is_file()
    reservoir = json.loads((output / "reservoir.json").read_text())
    assert reservoir["samples_per_layer"] == [3, 3]

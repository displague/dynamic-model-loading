"""Pinned OPT ReLU control with exact-zero and approximate group diagnostics."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .experiment import digest, environment, load_corpus, validate_config, write_json
from .metrics import aggregate, compare_logits, relative_l2
from . import relu


@torch.inference_mode()
def run(config_path, corpus_path, output_path, device="cuda"):
    output, config_path, corpus_path = Path(output_path), Path(config_path), Path(corpus_path)
    output.mkdir(parents=True, exist_ok=False)
    raw = (output / "results.jsonl").open("x", encoding="utf-8")
    def record(kind, **values):
        raw.write(json.dumps({"kind": kind, **values}, allow_nan=False) + "\n")
        raw.flush()
        os.fsync(raw.fileno())
    try:
        cfg = json.loads(config_path.read_text())
        validate_config(cfg)
        if cfg["keep_fractions"] != [.75] or cfg["layouts"] != ["native", "random", "popularity"]:
            raise ValueError("ReLU protocol requires its declared probe/layout grid")
        if (cfg["reconstruction_relative_l2_max"] != .01 or cfg["logit_relative_l2_max"] != .01 or
                cfg["logit_mean_kl_max"] != .001):
            raise ValueError("ReLU protocol requires original numerical limits")
        if cfg["dtype"] != "float32" or str(torch.__version__) != cfg["expected_torch"]:
            raise ValueError("ReLU protocol requires pinned FP32 environment")
        if digest(corpus_path) != cfg["expected_corpus_sha256"]:
            raise ValueError("ReLU corpus hash mismatch")
        corpus = load_corpus(corpus_path)
        device = torch.device(device)
        if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested execution device unavailable")
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for source, name in ((config_path, "config.json"), (corpus_path, "corpus.jsonl"),
                             (Path(cfg["protocol"]), "protocol.md")):
            shutil.copyfile(source, output / name)
        shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
        started = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(device),
                   "config_sha256": digest(config_path), "corpus_sha256": digest(corpus_path),
                   "protocol_sha256": digest(output / "protocol.md"),
                   "sources": {p.name: digest(p) for p in (output / "source").glob("*.py")}}
        write_json(output / "started.json", started)
        print("Loading pinned OPT checkpoint", flush=True)
        snapshot = Path(snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True))
        files = {p.name: digest(p) for p in snapshot.iterdir() if p.is_file()}
        if files["pytorch_model.bin"] != cfg["expected_weights_sha256"]:
            raise ValueError("OPT weight hash mismatch")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32, attn_implementation="sdpa",
            local_files_only=True, trust_remote_code=False, use_safetensors=False, weights_only=True).to(device).eval()
        layers = relu.extract(model)
        shapes = [list(relu.dimensions(layer)) for layer in layers]
        if shapes != cfg["expected_shapes"]:
            raise ValueError("OPT shape mismatch")
        tokens = {r["id"]: tokenizer(r["text"], return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=cfg["max_tokens"])["input_ids"].to(device) for r in corpus}
        if any(ids.shape[1] < 2 for ids in tokens.values()):
            raise ValueError("Document needs at least two tokens")
        exact_ids = {key: ids.cpu().tolist() for key, ids in tokens.items()}
        write_json(output / "token-ids.json", exact_ids)
        groupable = sum(layer.fc1.weight.numel()+layer.fc1.bias.numel()+layer.fc2.weight.numel() for layer in layers)*4
        bias_bytes = sum(layer.fc2.bias.numel() for layer in layers)*4
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        manifest = {**started, "model": cfg["model"], "revision": cfg["revision"], "checkpoint_files": files,
            "ffn_shapes": shapes, "model_parameter_bytes": parameter_bytes,
            "groupable_ffn_parameter_bytes": groupable, "fixed_ffn_output_bias_bytes": bias_bytes,
            "fixed_non_groupable_parameter_bytes": parameter_bytes-groupable,
            "diagnostic_layout_backup_bytes": groupable,
            "largest_layer_transaction_payload_bytes": max(
                layer.fc1.weight.numel()+layer.fc1.bias.numel()+layer.fc2.weight.numel() for layer in layers)*4*2,
            "token_ids_file_sha256": digest(output / "token-ids.json"),
            "token_ids_sha256": hashlib.sha256(json.dumps(exact_ids, sort_keys=True).encode()).hexdigest(),
            "attention_implementation": model.config._attn_implementation,
            "tokenization": "plain text, no added special tokens, OPT truncation of existing Qwen-prefix texts"}
        write_json(output / "manifest.json", manifest)
        diagnostics = [r for r in corpus if r["split"] == "diagnostic"]
        calibration = [r for r in corpus if r["split"] == "calibration"]
        def forward(ids):
            return model(input_ids=ids, use_cache=False).logits
        (output / "reference_logits").mkdir()
        references = {}
        print("Dense reference", flush=True)
        for index, row in enumerate(diagnostics):
            path = output / "reference_logits" / f"{index:03d}.safetensors"
            value = forward(tokens[row["id"]]).cpu().contiguous()
            save_file({"logits": value}, path)
            references[row["id"]] = path
            record("dense_reference", document=row["id"], input_tokens=tokens[row["id"]].shape[1],
                   logits_sha256=digest(path))
            del value
        def reference(name):
            return load_file(references[name])["logits"]
        print("Calibration-only popularity order", flush=True)
        collectors = [relu.ImportanceCollector(layer) for layer in layers]
        with relu.hooks(layers, collectors):
            for row in calibration:
                forward(tokens[row["id"]])
        generator = torch.Generator().manual_seed(cfg["seed"])
        orders = {"native": [torch.arange(shape[1]) for shape in shapes],
                  "random": [torch.randperm(shape[1], generator=generator) for shape in shapes],
                  "popularity": [collector.order() for collector in collectors]}
        write_json(output / "layouts.json", {name: [order.tolist() for order in values] for name, values in orders.items()})
        record("calibration", documents=[r["id"] for r in calibration],
               token_observations_per_layer=[c.tokens for c in collectors], layouts_sha256=digest(output / "layouts.json"))
        del collectors
        inputs = []
        handles = []
        try:
            for layer in layers:
                handles.append(layer.fc1.register_forward_pre_hook(lambda module, args: inputs.append(args[0][:2].clone())))
            forward(tokens[diagnostics[0]["id"]])
        finally:
            for handle in handles:
                handle.remove()
        passed = True
        for name in cfg["layouts"]:
            print("Numerical gates: " + name, flush=True)
            with relu.layout(layers, orders[name]):
                for index, (layer, x) in enumerate(zip(layers, inputs, strict=True)):
                    error = relative_l2(relu.dense_forward(layer, x), relu.grouped_forward(layer, x, cfg["reconstruction_group_width"]))
                    ok = error <= cfg["reconstruction_relative_l2_max"]
                    record("group_reconstruction", layout=name, layer=index, relative_l2=error, passed=ok)
                    passed &= ok
                for row in diagnostics:
                    metrics = compare_logits(reference(row["id"]), forward(tokens[row["id"]]).cpu(), tokens[row["id"]].cpu())
                    ok = metrics["logit_relative_l2"] <= cfg["logit_relative_l2_max"] and metrics["mean_kl_dense_to_candidate"] <= cfg["logit_mean_kl_max"]
                    record("layout_correctness", layout=name, document=row["id"], passed=ok, **metrics)
                    passed &= ok
        del inputs
        summary = {"status": "correctness_gate_failed", "correctness_passed": bool(passed),
                   "exact_zero_passed": None, "probes": []}
        if passed:
            zero_passed = True
            for mode in ("exact_zero", "approximate_75"):
                if mode == "approximate_75" and not zero_passed:
                    break
                for name in cfg["layouts"]:
                    with relu.layout(layers, orders[name]):
                        for width in cfg["group_widths"]:
                            probes = [relu.Probe(layer, width, mode) for layer in layers]
                            measurements, accounts = [], []
                            with relu.hooks(layers, probes):
                                for row in diagnostics:
                                    candidate = forward(tokens[row["id"]]).cpu()
                                    account = [probe.take() for probe in probes]
                                    metrics = compare_logits(reference(row["id"]), candidate, tokens[row["id"]].cpu())
                                    extra = {}
                                    if mode == "exact_zero":
                                        ok = metrics["logit_relative_l2"] <= cfg["logit_relative_l2_max"] and metrics["mean_kl_dense_to_candidate"] <= cfg["logit_mean_kl_max"]
                                        extra["passed"] = ok
                                        zero_passed &= ok
                                    record("relu_document", layout=name, group_width=width, mode=mode,
                                           document=row["id"], layer_accounting=account, **metrics, **extra)
                                    measurements.append(metrics)
                                    accounts.extend(account)
                                    del candidate
                            totals = {key: sum(a[key] for a in accounts) for key in accounts[0]}
                            result = {"layout": name, "group_width": width, "mode": mode, **aggregate(measurements),
                                "accounting": totals, "hypothetical_ffn_fraction": totals["hypothetical_selected_ffn_bytes"]/totals["full_ffn_bytes"],
                                "zero_neuron_fraction": totals["zero_neurons"]/totals["neuron_observations"],
                                "zero_group_fraction": totals["zero_groups"]/totals["group_observations"]}
                            record("relu_aggregate", **result)
                            summary["probes"].append(result)
                            print(f"{mode} {name} width={width}: fraction={result['hypothetical_ffn_fraction']:.6f} PPL={result['relative_perplexity']:.6f}", flush=True)
            summary["exact_zero_passed"] = bool(zero_passed)
            summary["status"] = "relu_control_completed" if zero_passed else "exact_zero_gate_failed"
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        record("error", error_type=type(error).__name__, message=str(error))
        write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.config, args.corpus, args.output, args.device)
    print(json.dumps({"status": result["status"]}))
    return 0 if result["status"] == "relu_control_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

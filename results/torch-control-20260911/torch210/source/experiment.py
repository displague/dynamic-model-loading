"""Run a reproducible correctness gate followed by optional analytical probes."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

from .adapters import extract_ffns
from .ffn import HindsightMask, ImportanceCollector, dimensions, grouped_forward, repack_
from .metrics import aggregate, compare_logits, relative_l2
from .packing import CoactivationCollector
from .trace import TraceRecorder, save_trace


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_corpus(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids, seen_texts = set(), set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "split", "domain", "text")):
            raise ValueError("Each corpus row needs nonempty id, split, domain, text")
        text_key = " ".join(row["text"].split())
        if row["id"] in seen_ids or text_key in seen_texts:
            raise ValueError("Duplicate document ID or text across corpus")
        seen_ids.add(row["id"])
        seen_texts.add(text_key)
        if row["split"] not in ("calibration", "diagnostic"):
            raise ValueError("This apparatus accepts calibration and diagnostic splits only")
    if {r["split"] for r in rows} != {"calibration", "diagnostic"}:
        raise ValueError("Both calibration and diagnostic documents are required")
    return rows


def validate_config(cfg: dict) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["revision"]):
        raise ValueError("Use an immutable 40-character checkpoint commit")
    if cfg["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Only float32 and bfloat16 are supported")
    for key in ("group_widths", "keep_fractions", "layouts"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"Empty or duplicated {key}")
    if any(type(x) is not int or x <= 0 for x in cfg["group_widths"]):
        raise ValueError("Group widths must be positive integers")
    if any(not 0 < x <= 1 for x in cfg["keep_fractions"]):
        raise ValueError("Keep fractions must be in (0, 1]")
    if not set(cfg["layouts"]) <= {"native", "random", "popularity", "coactivation"}:
        raise ValueError("Unsupported layout")
    if "coactivation" in cfg["layouts"]:
        for key in ("coactivation_reservoir", "coactivation_sketch_dim", "coactivation_group_width", "coactivation_iterations"):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
    for key in ("max_tokens", "cpu_threads", "reconstruction_group_width", "timing_repeats", "decode_tokens"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_tokens"] < 2:
        raise ValueError("Need at least two tokens")
    for key in ("reconstruction_relative_l2_max", "logit_relative_l2_max", "logit_mean_kl_max"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"Invalid tolerance {key}")


@contextmanager
def hooks(mlps, observers):
    handles = []
    try:
        for mlp, observer in zip(mlps, observers, strict=True):
            handles.append(mlp.down_proj.register_forward_pre_hook(observer))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def layout(mlps, orders):
    applied = []
    try:
        for mlp, order in zip(mlps, orders, strict=True):
            repack_(mlp, order)
            applied.append((mlp, order))
        yield
    finally:
        for mlp, order in reversed(applied):
            repack_(mlp, order.argsort())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device) -> dict:
    value = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("transformers", "safetensors", "huggingface-hub", "numpy")},
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cpu_threads": torch.get_num_threads(),
        "process_id": os.getpid(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        value["gpu"] = {"name": props.name, "total_bytes": props.total_memory,
                        "compute_capability": list(torch.cuda.get_device_capability(device))}
        try:
            value["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader"], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            value["nvidia_smi"] = "unavailable"
    return value


def cuda_memory(device) -> dict:
    if device.type != "cuda":
        return {}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def render_report(summary: dict) -> str:
    text = ["# Apparatus run", "", f"Status: **{summary['status']}**.", "",
            f"Purpose: `{summary.get('purpose', 'apparatus_smoke_test')}`.", "",
            "This is an exploratory diagnostic. It is not a confirmatory held-out quality benchmark,",
            "a bounded-memory runtime, or evidence of an inference speedup.", "",
            "## Correctness", "", f"All-group reconstruction and layout checks passed: {summary['correctness_passed']}.", "",
            "Tolerances were copied into this run before measurement. Raw per-document and per-layer",
            "results are in `results.jsonl`; model, source, corpus, and environment provenance are in `manifest.json`.", ""]
    if summary["probes"]:
        text += ["## Hindsight diagnostics", "",
                 "All FFNs are masked together. Gate/up activations are computed densely before selection.",
                 "Selected bytes are a hypothetical no-cache weight volume, not measured transfers.", "",
                 "| Layout | Neurons/group | Groups retained | Weight fraction | KL | Relative perplexity | Top-1 agreement |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary["probes"]:
            text.append(f"| {row['layout']} | {row['group_width']} | {row['keep_fraction']:.0%} | "
                        f"{row['hypothetical_weight_fraction']:.3f} | {row['mean_kl_dense_to_candidate']:.5g} | "
                        f"{row['relative_perplexity']:.4f} | {row['top1_agreement']:.3%} |")
    text += ["", "No selection predictor, cache simulation, weight transfer runtime,",
             "or omission-error detector is implemented. These measurements do not establish their feasibility.", ""]
    return "\n".join(text)


@torch.inference_mode()
def run(args) -> dict:
    config_path, corpus_path = Path(args.config), Path(args.corpus)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    corpus = load_corpus(corpus_path)
    if cfg.get("capture_traces"):
        for key, actual in (("expected_corpus_sha256", digest(corpus_path)),
                            ("expected_torch", str(torch.__version__))):
            if cfg[key] != actual:
                raise ValueError(f"Trace protocol input mismatch: {key}")
        if cfg["dtype"] != "float32":
            raise ValueError("Trace protocol requires FP32")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    if device.type == "cuda" and cfg["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is unsupported")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "config.json")
    shutil.copyfile(corpus_path, output / "corpus.jsonl")
    protocol = Path(cfg.get("protocol", Path(__file__).resolve().parents[2] / "docs" / "protocol.md"))
    if "protocol" in cfg and not protocol.is_file():
        raise ValueError("Configured protocol file does not exist")
    if protocol.exists():
        shutil.copyfile(protocol, output / "protocol.md")
    source_root = Path(__file__).parent
    shutil.copytree(source_root, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": cfg["purpose"], "mode": "diagnostics" if args.diagnostics else "stage0",
        "config_sha256": digest(config_path), "corpus_sha256": digest(corpus_path),
        "sources": {p.name: digest(p) for p in sorted(source_root.glob("*.py"))},
        "environment": environment(device), "checkpoint": {"model": cfg["model"], "revision": cfg["revision"]},
    }
    if protocol.is_file():
        manifest["protocol_sha256"] = digest(output / "protocol.md")
    provenance = corpus_path.parent / "provenance.json"
    if provenance.is_file():
        if json.loads(provenance.read_text())["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("Corpus provenance does not match the input corpus")
        shutil.copyfile(provenance, output / "corpus-provenance.json")
        manifest["corpus_provenance_sha256"] = digest(output / "corpus-provenance.json")
    write_json(output / "started.json", manifest)
    raw = (output / "results.jsonl").open("x", encoding="utf-8")

    def record(kind, **values):
        raw.write(json.dumps({"kind": kind, **values}, allow_nan=False) + "\n")
        raw.flush()
        os.fsync(raw.fileno())

    summary = {"status": "running", "purpose": cfg["purpose"], "correctness_passed": False, "probes": []}
    try:
        print("Loading pinned checkpoint", flush=True)
        start = time.perf_counter()
        snapshot = Path(snapshot_download(cfg["model"], revision=cfg["revision"],
            local_files_only=not args.allow_download,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
        manifest["checkpoint"]["files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(snapshot.iterdir()) if p.is_file()
        }
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=getattr(torch, cfg["dtype"]),
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to(device).eval()
        mlps = extract_ffns(model)
        dims = [dimensions(mlp) for mlp in mlps]
        tokens = {}
        for row in corpus:
            # Plain text, no chat template: this is corpus NLL, not chat-response accuracy.
            ids = tokenizer(row["text"], return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=cfg["max_tokens"])["input_ids"]
            if ids.shape[1] < 2:
                raise ValueError(f"Document {row['id']} has fewer than two tokens")
            tokens[row["id"]] = ids.to(device)
        diagnostics = [row for row in corpus if row["split"] == "diagnostic"]
        calibration = [row for row in corpus if row["split"] == "calibration"]
        manifest["tokenization"] = {"chat_template_applied": False, "special_tokens_added": False,
            "document_tokens": {key: ids.shape[1] for key, ids in tokens.items()},
            "token_ids_sha256": hashlib.sha256(json.dumps({key: ids.cpu().tolist() for key, ids in tokens.items()},
                                                        sort_keys=True).encode()).hexdigest()}
        manifest["model_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
        manifest["ffn_parameter_bytes"] = sum(p.numel() * p.element_size() for m in mlps for p in m.parameters())
        manifest["ffn_shapes"] = dims
        manifest["attention_implementation"] = model.config._attn_implementation
        if cfg.get("capture_traces"):
            if len(set(dims)) != 1:
                raise ValueError("Trace protocol requires homogeneous FFN dimensions")
            if manifest["tokenization"]["token_ids_sha256"] != cfg["expected_token_ids_sha256"]:
                raise ValueError("Trace tokenization mismatch")
            write_json(output / "token-ids.json", {key: ids.cpu().tolist() for key, ids in tokens.items()})
            (output / "traces").mkdir()
        synchronize(device)
        record("load", seconds=time.perf_counter() - start, memory=cuda_memory(device))
        write_json(output / "manifest.json", manifest)

        def forward(ids):
            return model(input_ids=ids, use_cache=False).logits

        print("Dense reference and synchronized timing", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        references = {}
        if cfg.get("spill_reference_logits", False):
            (output / "reference_logits").mkdir()

        def reference(document):
            value = references[document]
            return load_file(value)["logits"] if isinstance(value, Path) else value

        for row in diagnostics:
            ids = tokens[row["id"]]
            del_output = forward(ids)  # Shape-specific warmup is excluded.
            del del_output
            timings = []
            for _ in range(cfg["timing_repeats"]):
                synchronize(device)
                start = time.perf_counter()
                logits = forward(ids)
                synchronize(device)
                timings.append(time.perf_counter() - start)
                if cfg.get("spill_reference_logits", False):
                    if len(timings) == cfg["timing_repeats"]:
                        ref_path = output / "reference_logits" / f"{len(references):04d}.safetensors"
                        save_file({"logits": logits.cpu().contiguous()}, ref_path)
                        references[row["id"]] = ref_path
                else:
                    references[row["id"]] = logits.cpu()
                del logits
            record("dense_prefill", document=row["id"], input_tokens=ids.shape[1],
                   seconds=timings, median_seconds=statistics.median(timings), memory=cuda_memory(device))
        # An explicit incremental decode baseline includes KV; EOS does not shorten the loop.
        decode_prompt = tokens[diagnostics[0]["id"]]
        prefill = model(input_ids=decode_prompt, use_cache=True)
        cache = prefill.past_key_values
        next_id = prefill.logits[:, -1:].argmax(-1)
        del prefill
        generated, decode_times = [], []
        for _ in range(cfg["decode_tokens"]):
            generated.append(next_id.item())
            synchronize(device)
            start = time.perf_counter()
            step = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id, cache = step.logits[:, -1:].argmax(-1), step.past_key_values
            synchronize(device)
            decode_times.append(time.perf_counter() - start)
            del step
        record("dense_decode", prompt_document=diagnostics[0]["id"], steps=len(decode_times),
               seconds=decode_times, median_seconds=statistics.median(decode_times),
               output_token_ids=generated, memory=cuda_memory(device))
        del cache

        print("Calibration-only layouts", flush=True)
        calibration_start = time.perf_counter()
        collectors = ([CoactivationCollector(m, cfg["coactivation_reservoir"], cfg["seed"]) for m in mlps]
                      if "coactivation" in cfg["layouts"] else [ImportanceCollector(m) for m in mlps])
        with hooks(mlps, collectors):
            for row in calibration:
                forward(tokens[row["id"]])
        popularity = [c.order() for c in collectors]
        calibration_counts = [c.tokens for c in collectors]
        generator = torch.Generator().manual_seed(cfg["seed"])
        orders = {"native": [torch.arange(width) for _, width in dims],
                  "random": [torch.randperm(width, generator=generator) for _, width in dims],
                  "popularity": popularity}
        if "coactivation" in cfg["layouts"]:
            orders["coactivation"] = []
            for index, collector in enumerate(collectors):
                orders["coactivation"].append(collector.coactivation_order(
                    cfg["coactivation_group_width"], cfg["coactivation_sketch_dim"],
                    cfg["coactivation_iterations"], cfg["seed"] + index))
            write_json(output / "reservoir.json", {"token_ordinals_per_layer": [c.ordinals.tolist() for c in collectors],
                                                   "samples_per_layer": [len(c.ordinals) for c in collectors]})
        importance_receipt = {}
        if cfg.get("capture_traces"):
            save_file({str(i): (c.total / c.tokens).cpu().contiguous()
                       for i, c in enumerate(collectors)}, output / "calibration-importance.safetensors")
            importance_receipt["importance_sha256"] = digest(output / "calibration-importance.safetensors")
        del collectors
        write_json(output / "layouts.json", {key: [p.tolist() for p in value] for key, value in orders.items()})
        if cfg.get("capture_traces") and digest(output / "layouts.json") != cfg["expected_layouts_sha256"]:
            raise ValueError("Regenerated trace layouts mismatch")
        record("calibration", token_observations_per_layer=calibration_counts,
               documents=[r["id"] for r in calibration], layouts_sha256=digest(output / "layouts.json"),
               seconds=time.perf_counter() - calibration_start, **importance_receipt)

        inputs = []
        handles = []
        try:
            for mlp in mlps:
                handles.append(mlp.register_forward_pre_hook(lambda module, args: inputs.append(args[0][:, :2].clone())))
            forward(tokens[diagnostics[0]["id"]])
        finally:
            for handle in handles:
                handle.remove()
        passed = True
        for name in cfg["layouts"]:
            print(f"Correctness gate: {name}", flush=True)
            with layout(mlps, orders[name]):
                for index, (mlp, x) in enumerate(zip(mlps, inputs, strict=True)):
                    error = relative_l2(mlp(x), grouped_forward(mlp, x, cfg["reconstruction_group_width"]))
                    ok = error <= cfg["reconstruction_relative_l2_max"]
                    record("group_reconstruction", layout=name, layer=index, relative_l2=error, passed=ok)
                    passed &= ok
                for row in diagnostics:
                    candidate = forward(tokens[row["id"]]).cpu()
                    metrics = compare_logits(reference(row["id"]), candidate, tokens[row["id"]].cpu())
                    ok = (metrics["logit_relative_l2"] <= cfg["logit_relative_l2_max"] and
                          metrics["mean_kl_dense_to_candidate"] <= cfg["logit_mean_kl_max"])
                    record("layout_correctness", layout=name, document=row["id"], passed=ok, **metrics)
                    passed &= ok
                    del candidate
        del inputs
        summary["correctness_passed"] = bool(passed)
        if not passed:
            summary["status"] = "correctness_gate_failed"
            print("Correctness gate failed; sparsity probes are blocked", flush=True)
        elif args.diagnostics:
            print("Correctness passed; beginning hindsight diagnostics (not timing sparse execution)", flush=True)
            for name in cfg["layouts"]:
                with layout(mlps, orders[name]):
                    for width in cfg["group_widths"]:
                        for keep in cfg["keep_fractions"]:
                            observers = ([TraceRecorder(width, keep) for _ in mlps]
                                         if cfg.get("capture_traces") else [None] * len(mlps))
                            masks = [HindsightMask(m, width, keep, observer)
                                     for m, observer in zip(mlps, observers, strict=True)]
                            measurements = []
                            with hooks(mlps, masks):
                                for row in diagnostics:
                                    candidate = forward(tokens[row["id"]]).cpu()
                                    metrics = compare_logits(reference(row["id"]), candidate, tokens[row["id"]].cpu())
                                    extra = {}
                                    if cfg.get("capture_traces"):
                                        trace_name = f"{name}-{width}-{keep}-{len(measurements):03d}.npz"
                                        extra["trace"] = save_trace(output / "traces" / trace_name, observers,
                                            {"document": row["id"], "layout": name, "group_width": width,
                                             "keep_fraction": keep, "neurons": dims[0][1],
                                             "bytes_per_neuron": masks[0].bytes_per_neuron})
                                    record("hindsight_document", layout=name, group_width=width, keep_fraction=keep,
                                           document=row["id"], domain=row["domain"], **metrics, **extra)
                                    measurements.append(metrics)
                                    del candidate
                            accounts = [mask.accounting() for mask in masks]
                            selected_bytes = sum(a["hypothetical_selected_weight_bytes"] for a in accounts)
                            dense_bytes = sum(a["full_weight_bytes_for_same_observations"] for a in accounts)
                            result = {"layout": name, "group_width": width, "keep_fraction": keep,
                                      **aggregate(measurements),
                                      "hypothetical_selected_weight_bytes": selected_bytes,
                                      "hypothetical_weight_fraction": selected_bytes / dense_bytes}
                            record("hindsight_aggregate", **result, layer_accounting=accounts)
                            summary["probes"].append(result)
                            print(f"{name} group={width} keep={keep}: KL={result['mean_kl_dense_to_candidate']:.5g}", flush=True)
                            del masks
            summary["status"] = "packing_pilot_completed" if "coactivation" in cfg["layouts"] else "smoke_diagnostics_completed"
        else:
            summary["status"] = "stage0_passed"
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
        with (output / "report.md").open("x", encoding="utf-8") as report:
            report.write(render_report(summary))
        return summary
    except (Exception, KeyboardInterrupt) as error:
        record("error", error_type=type(error).__name__, message=str(error))
        write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--corpus", default="data/smoke.jsonl")
    parser.add_argument("--output", required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true", help="Fetch the pinned checkpoint if not cached")
    parser.add_argument("--diagnostics", action="store_true", help="Run hindsight probes only after correctness passes")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output": str(Path(args.output).resolve())}))
    return 0 if summary["correctness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

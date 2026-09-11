"""Frozen-layout, per-layer and fixed-block omission measurements."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import extract_ffns
from .analysis import paired_kl_interval
from .experiment import digest, environment, hooks, layout, load_corpus, write_json
from .ffn import HindsightMask, dimensions, grouped_forward, validate_order
from .metrics import aggregate, compare_logits, relative_l2


def load_parent(root, cfg):
    for filename, key in (("manifest.json", "parent_manifest_sha256"), ("summary.json", "parent_summary_sha256")):
        if digest(root / filename) != cfg[key]:
            raise ValueError(f"Parent {filename} digest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    parent = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "packing_pilot_completed" or not summary["correctness_passed"]:
        raise ValueError("Parent must be a completed pilot with passing correctness")
    for name, key in (("config.json", "config_sha256"), ("corpus.jsonl", "corpus_sha256"),
                      ("corpus-provenance.json", "corpus_provenance_sha256")):
        if digest(root / name) != manifest[key]:
            raise ValueError(f"Parent {name} digest mismatch")
    if digest(root / "results.jsonl") != cfg["parent_results_sha256"]:
        raise ValueError("Parent raw ledger digest mismatch")
    raw = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    expected_documents = {r["id"] for r in load_corpus(root / "corpus.jsonl") if r["split"] == "diagnostic"}
    for name in cfg["layouts"]:
        match = lambda r: (r["layout"] == name and r["group_width"] == cfg["group_width"]
                           and r["keep_fraction"] == cfg["keep_fraction"])
        probes = [p for p in summary["probes"] if match(p)]
        rows = [r for r in raw if r["kind"] == "hindsight_document" and match(r)]
        if len(probes) != 1 or len(rows) != len(expected_documents) or {r["document"] for r in rows} != expected_documents:
            raise ValueError("Parent control document coverage mismatch")
        if any(probes[0][key] != value for key, value in aggregate(rows).items()):
            raise ValueError("Parent summary does not reproduce from its raw ledger")
    calibration = [r for r in raw if r["kind"] == "calibration"]
    if len(calibration) != 1:
        raise ValueError("Expected exactly one parent calibration receipt")
    p = root / "layouts.json"
    data = p.read_bytes() if p.exists() else gzip.decompress((root / "layouts.json.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != calibration[0]["layouts_sha256"]:
        raise ValueError("Parent layouts digest mismatch")
    if parent["dtype"] != "float32":
        raise ValueError("This omission study requires the FP32 parent control")
    return manifest, parent, summary, json.loads(data), data


def conditions(layers, blocks):
    flattened = [i for block in blocks for i in block]
    if not blocks or any(not block for block in blocks) or sorted(flattened) != list(range(layers)):
        raise ValueError("Fixed blocks must partition all layers exactly once")
    if any(type(i) is not int for i in flattened):
        raise ValueError("Layer indices must be integers")
    return ([{"name": f"layer-{i:02d}", "kind": "single", "layers": [i]} for i in range(layers)]
            + [{"name": "all", "kind": "all", "layers": list(range(layers))}]
            + [{"name": f"block-{i}", "kind": "block", "layers": block} for i, block in enumerate(blocks)])


def summarize(probes, documents, parent_summary):
    names = sorted({p["layout"] for p in probes})
    analysis = {}
    for name in names:
        singles = [p for p in probes if p["layout"] == name and p["condition_kind"] == "single"]
        ranked = sorted(singles, key=lambda p: (-p["mean_kl_dense_to_candidate"], p["layers"][0]))
        total = sum(p["mean_kl_dense_to_candidate"] for p in singles)
        joint = next(p for p in probes if p["layout"] == name and p["condition_kind"] == "all")
        parent = next(p for p in parent_summary["probes"] if p["layout"] == name
                      and p["group_width"] == joint["group_width"] and p["keep_fraction"] == joint["keep_fraction"])
        analysis[name] = {"sum_single_layer_kl": total, "joint_layer_kl": joint["mean_kl_dense_to_candidate"],
                          "joint_to_sum_ratio": joint["mean_kl_dense_to_candidate"] / total if total else None,
                          "largest_four_layers": [p["layers"][0] for p in ranked[:4]],
                          "largest_four_share_of_single_kl_sum": sum(p["mean_kl_dense_to_candidate"] for p in ranked[:4]) / total if total else None,
                          "parent_joint_kl": parent["mean_kl_dense_to_candidate"],
                          "joint_kl_minus_parent": joint["mean_kl_dense_to_candidate"] - parent["mean_kl_dense_to_candidate"]}
    paired = [{**row, "kind": "hindsight_document"} for row in documents if row["condition_kind"] == "all"]
    first = probes[0]
    interval = paired_kl_interval(paired, first["group_width"], first["keep_fraction"])
    return {"layer_analysis": analysis, "all_layer_paired_comparison": interval}


def render_report(summary):
    lines = ["# Per-layer omission study", "", f"Status: **{summary['status']}**.", "",
             "Frozen calibration layouts; teacher-forced development articles; dense masked execution.",
             "Selected FFN volume is hypothetical, with unmasked layers charged in full. No paging speedup is measured.", "",
             "| Layout | Condition | Layers | Mean KL | Relative perplexity | Top-1 | Whole-FFN weight fraction |",
             "|---|---|---|---:|---:|---:|---:|"]
    for p in summary["probes"]:
        layer_text = str(p["layers"][0]) if len(p["layers"]) == 1 else f"{min(p['layers'])}-{max(p['layers'])}"
        lines.append(f"| {p['layout']} | {p['condition']} | {layer_text} | {p['mean_kl_dense_to_candidate']:.6g} | "
                     f"{p['relative_perplexity']:.6f} | {p['top1_agreement']:.2%} | {p['hypothetical_whole_ffn_weight_fraction']:.4%} |")
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def run(args):
    root, output = Path(args.parent), Path(args.output)
    cfg = json.loads(Path(args.config).read_text())
    manifest, parent, parent_summary, arrays, layout_bytes = load_parent(root, cfg)
    if str(torch.__version__) != cfg["expected_torch"]:
        raise ValueError("Unexpected torch version; use a separately declared environment control")
    if cfg["layouts"] != ["popularity", "coactivation"]:
        raise ValueError("The paired study requires popularity and coactivation layouts")
    if type(cfg["group_width"]) is not int or cfg["group_width"] <= 0 or not 0 < cfg["keep_fraction"] <= 1:
        raise ValueError("Invalid omission settings")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True, exist_ok=False)
    raw = (output / "results.jsonl").open("x", encoding="utf-8")

    def record(kind, **values):
        row = {"kind": kind, **values}
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        raw.flush()
        os.fsync(raw.fileno())
        return row

    try:
        torch.set_num_threads(parent["cpu_threads"])
        torch.manual_seed(parent["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        shutil.copyfile(args.config, output / "config.json")
        shutil.copyfile(cfg["protocol"], output / "protocol.md")
        for filename in ("corpus.jsonl", "corpus-provenance.json"):
            shutil.copyfile(root / filename, output / filename)
        shutil.copyfile(root / "manifest.json", output / "parent-manifest.json")
        shutil.copyfile(root / "summary.json", output / "parent-summary.json")
        (output / "layouts.json.gz").write_bytes(gzip.compress(layout_bytes, mtime=0))
        shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
        checkpoint = Path(snapshot_download(parent["model"], revision=parent["revision"], local_files_only=True))
        for filename, receipt in manifest["checkpoint"]["files"].items():
            if digest(checkpoint / filename) != receipt["sha256"]:
                raise ValueError(f"Checkpoint digest mismatch: {filename}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        corpus = load_corpus(output / "corpus.jsonl")
        tokens = {r["id"]: tokenizer(r["text"], return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=parent["max_tokens"])["input_ids"].to(device) for r in corpus}
        ids = {key: value.cpu().tolist() for key, value in tokens.items()}
        token_hash = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
        if token_hash != manifest["tokenization"]["token_ids_sha256"]:
            raise ValueError("Token IDs differ from the frozen parent")
        write_json(output / "token-ids.json", ids)
        diagnostics = [r for r in corpus if r["split"] == "diagnostic"]
        mlps = extract_ffns(model)
        cases = conditions(len(mlps), cfg["blocks"])
        orders = {name: [torch.tensor(p, dtype=torch.long) for p in arrays[name]] for name in cfg["layouts"]}
        for values in orders.values():
            if len(values) != len(mlps):
                raise ValueError("Layout layer count mismatch")
            for mlp, order in zip(mlps, values, strict=True):
                validate_order(order, dimensions(mlp)[1])
        write_json(output / "manifest.json", {"purpose": cfg["purpose"], "started_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment(device), "parent_results_sha256": cfg["parent_results_sha256"],
            "parent_manifest_sha256": digest(output / "parent-manifest.json"),
            "parent_summary_sha256": digest(output / "parent-summary.json"),
            "config_sha256": digest(output / "config.json"), "protocol_sha256": digest(output / "protocol.md"),
            "corpus_sha256": digest(output / "corpus.jsonl"), "corpus_provenance_sha256": digest(output / "corpus-provenance.json"),
            "layouts_sha256": hashlib.sha256(layout_bytes).hexdigest(), "token_ids_sha256": token_hash,
            "token_ids_file_sha256": digest(output / "token-ids.json"), "checkpoint": manifest["checkpoint"],
            "sources": {p.name: digest(p) for p in (output / "source").glob("*.py")},
            "conditions": cases, "diagnostic_documents": [r["id"] for r in diagnostics]})

        def forward(document):
            return model(input_ids=tokens[document], use_cache=False).logits.cpu()

        (output / "reference_logits").mkdir()
        reference_paths = {}
        for i, row in enumerate(diagnostics):
            p = output / "reference_logits" / f"{i:04d}.safetensors"
            save_file({"logits": forward(row["id"]).contiguous()}, p)
            reference_paths[row["id"]] = p
            record("dense_reference", document=row["id"], logits_sha256=digest(p))
        inputs, handles = [], []
        try:
            for mlp in mlps:
                handles.append(mlp.register_forward_pre_hook(lambda module, args: inputs.append(args[0][:, :2].clone())))
            forward(diagnostics[0]["id"])
        finally:
            for handle in handles:
                handle.remove()
        passed = True
        for name in cfg["layouts"]:
            with layout(mlps, orders[name]):
                for i, (mlp, x) in enumerate(zip(mlps, inputs, strict=True)):
                    error = relative_l2(mlp(x), grouped_forward(mlp, x, parent["reconstruction_group_width"]))
                    ok = error <= parent["reconstruction_relative_l2_max"]
                    record("group_reconstruction", layout=name, layer=i, relative_l2=error, passed=ok)
                    passed &= ok
                for row in diagnostics:
                    key = row["id"]
                    metrics = compare_logits(load_file(reference_paths[key])["logits"], forward(key), tokens[key].cpu())
                    ok = metrics["logit_relative_l2"] <= parent["logit_relative_l2_max"] and metrics["mean_kl_dense_to_candidate"] <= parent["logit_mean_kl_max"]
                    record("layout_correctness", layout=name, document=key, passed=ok, **metrics)
                    passed &= ok
        del inputs
        summary = {"status": "correctness_gate_failed", "correctness_passed": bool(passed), "probes": []}
        documents = []
        if passed:
            total_tokens = sum(tokens[r["id"]].shape[1] for r in diagnostics)
            layer_bytes = [sum(p.numel() * p.element_size() for p in m.parameters()) for m in mlps]
            dense_bytes = total_tokens * sum(layer_bytes)
            for name in cfg["layouts"]:
                with layout(mlps, orders[name]):
                    for case in cases:
                        active = [mlps[i] for i in case["layers"]]
                        masks = [HindsightMask(m, cfg["group_width"], cfg["keep_fraction"]) for m in active]
                        common = {"layout": name, "condition": case["name"], "condition_kind": case["kind"],
                                  "layers": case["layers"], "group_width": cfg["group_width"], "keep_fraction": cfg["keep_fraction"]}
                        measurements = []
                        with hooks(active, masks):
                            for row in diagnostics:
                                key = row["id"]
                                metrics = compare_logits(load_file(reference_paths[key])["logits"], forward(key), tokens[key].cpu())
                                documents.append(record("omission_document", document=key, **common, **metrics))
                                measurements.append(metrics)
                        accounts = [m.accounting() for m in masks]
                        selected_bytes = sum(a["hypothetical_selected_weight_bytes"] for a in accounts)
                        unmasked_bytes = total_tokens * sum(b for i, b in enumerate(layer_bytes) if i not in case["layers"])
                        result = {**common, **aggregate(measurements), "layer_accounting": accounts,
                                  "hypothetical_masked_layer_selected_bytes": selected_bytes,
                                  "unmasked_layer_weight_bytes": unmasked_bytes,
                                  "full_ffn_weight_bytes_for_same_observations": dense_bytes,
                                  "hypothetical_whole_ffn_weight_fraction": (selected_bytes + unmasked_bytes) / dense_bytes}
                        record("omission_aggregate", **result)
                        summary["probes"].append(result)
                        print(f"{name} {case['name']}: KL={result['mean_kl_dense_to_candidate']:.6g}", flush=True)
            summary["status"] = "layer_study_completed"
            summary.update(summarize(summary["probes"], documents, parent_summary))
        summary["raw_results_sha256"] = digest(output / "results.jsonl")
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(render_report(summary), encoding="utf-8")
        return summary
    except (Exception, KeyboardInterrupt) as error:
        try:
            record("error", error_type=type(error).__name__, message=str(error))
            write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        except OSError:
            pass  # Preserve the original failure if receipt storage itself is unavailable.
        raise
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="results/packing-pilot-20260911/pilot")
    parser.add_argument("--config", default="configs/layer-study.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    summary = run(parser.parse_args())
    return 0 if summary["correctness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose arithmetic-order sensitivity without relaxing the original gates."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import importlib.metadata

import torch
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

from .adapters import extract_ffns
from .experiment import digest, environment, layout, load_corpus, validate_config, write_json
from .ffn import dimensions
from .metrics import compare_logits


MODES = ("bf16", "reduction_off", "fp32_down", "fp32_ffn", "canonical_down")


def inventory():
    names = sorted({d.metadata["Name"].lower().replace("_", "-")
                    for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    return {"python": sys.version, "executable": sys.executable,
            "versions": {name: importlib.metadata.version(name) for name in names}}


def validate_control(cfg, values, corpus_path):
    if cfg.get("environment_control"):
        if cfg["logit_relative_l2_max"] != 0.01 or cfg["logit_mean_kl_max"] != 0.001:
            raise ValueError("Environment control requires the original numerical limits")
        if values["python"] != cfg["expected_python"]:
            raise ValueError("Environment control interpreter mismatch")
        actual = hashlib.sha256(json.dumps(values["versions"], sort_keys=True).encode()).hexdigest()
        if actual != cfg["expected_versions_sha256"]:
            raise ValueError("Environment control distribution inventory mismatch")
        if digest(Path(corpus_path)) != cfg["expected_corpus_sha256"]:
            raise ValueError("Environment control corpus mismatch")


def save_inputs(output, tokens, native, random_order):
    """Persist exact token IDs and permutations before any numerical measurement."""
    ids = {key: value.cpu().tolist() for key, value in tokens.items()}
    write_json(output / "token-ids.json", ids)
    write_json(output / "layouts.json", {"native": [p.tolist() for p in native],
                                         "random": [p.tolist() for p in random_order]})
    return {"tokenization": {"chat_template_applied": False, "special_tokens_added": False,
                            "document_tokens": {key: value.shape[1] for key, value in tokens.items()},
                            "token_ids_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()},
            "token_ids_file_sha256": digest(output / "token-ids.json"),
            "layouts_sha256": digest(output / "layouts.json")}


@contextmanager
def arithmetic(mlps, mode, orders):
    """Temporary diagnostic forwards, restored even when a comparison fails."""
    if mode not in MODES:
        raise ValueError(f"Unknown arithmetic policy: {mode}")
    reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    saved = []
    absent = object()
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = mode != "reduction_off"
        for mlp, order in zip(mlps, orders, strict=True):
            if mode == "fp32_ffn":
                target = mlp.source

                def forward(x, m=mlp):
                    z = F.silu(F.linear(x.float(), m.gate_proj.weight.float()))
                    z = z * F.linear(x.float(), m.up_proj.weight.float())
                    return F.linear(z, m.down_proj.weight.float()).to(x.dtype)
            elif mode in ("fp32_down", "canonical_down"):
                target = mlp.down_proj
                inverse = order.argsort().to(target.weight.device)

                def forward(z, m=mlp, inverse=inverse, policy=mode):
                    if policy == "fp32_down":
                        return F.linear(z.float(), m.down_proj.weight.float()).to(z.dtype)
                    return F.linear(z.index_select(-1, inverse), m.down_proj.weight.index_select(1, inverse))
            else:
                continue
            saved.append((target, target.__dict__.get("forward", absent)))
            target.forward = forward
        yield
    finally:
        for target, previous in reversed(saved):
            if previous is absent:
                del target.forward
            else:
                target.forward = previous
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced


def run_precision(config_path, corpus_path, output_path):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run_precision(config_path, corpus_path, output_path)
    except (Exception, KeyboardInterrupt) as error:
        if output.is_dir() and not (output / "failure.json").exists():
            write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise


@torch.inference_mode()
def _run_precision(config_path, corpus_path, output_path):
    cfg = json.loads(Path(config_path).read_text())
    validate_config(cfg)
    if cfg["dtype"] != "bfloat16":
        raise ValueError("Precision diagnosis requires a BF16 checkpoint configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("Precision diagnosis requires CUDA")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(output_path)
    shutil.copyfile(config_path, output / "config.json")
    shutil.copyfile(corpus_path, output / "corpus.jsonl")
    shutil.copyfile(cfg.get("protocol", "docs/packing-pilot-protocol.md"), output / "protocol.md")
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    values = inventory()
    write_json(output / "environment-inventory.json", values)
    validate_control(cfg, values, corpus_path)
    corpus = [r for r in load_corpus(Path(corpus_path)) if r["split"] == "diagnostic"]
    checkpoint = Path(snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True))
    manifest = {"purpose": "numerical_diagnosis_not_sparse_quality_evaluation", "modes": MODES,
                "config_sha256": digest(Path(config_path)), "corpus_sha256": digest(Path(corpus_path)),
                "protocol_sha256": digest(output / "protocol.md"),
                "model": cfg["model"], "revision": cfg["revision"],
                "environment": environment(torch.device("cuda")),
                "inventory_sha256": digest(output / "environment-inventory.json"),
                "sources": {p.name: digest(p) for p in (output / "source").glob("*.py")},
                "checkpoint_files": {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}}
    if cfg.get("environment_control") and manifest["checkpoint_files"] != cfg["expected_checkpoint_files"]:
        raise ValueError("Environment control checkpoint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).cuda().eval()
    mlps = extract_ffns(model)
    tokens = {r["id"]: tokenizer(r["text"], return_tensors="pt", add_special_tokens=False,
                               truncation=True, max_length=cfg["max_tokens"])["input_ids"].cuda() for r in corpus}
    generator = torch.Generator().manual_seed(cfg["seed"])
    native = [torch.arange(dimensions(m)[1]) for m in mlps]
    random_order = [torch.randperm(dimensions(m)[1], generator=generator) for m in mlps]
    manifest.update(save_inputs(output, tokens, native, random_order))
    write_json(output / "manifest.json", manifest)
    if cfg.get("environment_control"):
        if (manifest["tokenization"]["token_ids_sha256"] != cfg["expected_token_ids_sha256"] or
                manifest["layouts_sha256"] != cfg["expected_layouts_sha256"]):
            raise ValueError("Environment control token/layout mismatch")

    def logits(ids):
        return model(input_ids=ids, use_cache=False).logits.cpu()

    with arithmetic(mlps, "bf16", native):
        original = {key: logits(ids) for key, ids in tokens.items()}
    if cfg.get("environment_control"):
        save_file({key: value.contiguous() for key, value in original.items()}, output / "ordinary-logits.safetensors",
                  metadata={"manifest_sha256": digest(output / "manifest.json"), "comparison": "ordinary_native_bf16"})
        write_json(output / "ordinary-logits-receipt.json", {"sha256": digest(output / "ordinary-logits.safetensors"),
                   "documents": list(original), "comparison": "ordinary_native_bf16"})
    results = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as raw:
        for mode in MODES:
            print(f"Arithmetic diagnosis: {mode}", flush=True)
            with arithmetic(mlps, mode, native):
                references = {key: logits(ids) for key, ids in tokens.items()}
            with layout(mlps, random_order), arithmetic(mlps, mode, random_order):
                for key, ids in tokens.items():
                    candidate = logits(ids)
                    comparisons = {
                        "permutation_within_policy": compare_logits(references[key], candidate, ids.cpu()),
                        "policy_vs_original_bf16": compare_logits(original[key], references[key], ids.cpu()),
                    }
                    for comparison, metrics in comparisons.items():
                        row = {"mode": mode, "document": key, "comparison": comparison, **metrics,
                               "within_original_limits": metrics["logit_relative_l2"] <= cfg["logit_relative_l2_max"]
                               and metrics["mean_kl_dense_to_candidate"] <= cfg["logit_mean_kl_max"]}
                        raw.write(json.dumps(row, allow_nan=False) + "\n")
                        raw.flush()
                        os.fsync(raw.fileno())
                        results.append(row)
    summary = []
    for mode in MODES:
        for comparison in ("permutation_within_policy", "policy_vs_original_bf16"):
            rows = [r for r in results if r["mode"] == mode and r["comparison"] == comparison]
            summary.append({"mode": mode, "comparison": comparison,
                            "all_within_original_limits": all(r["within_original_limits"] for r in rows),
                            "max_relative_l2": max(r["logit_relative_l2"] for r in rows),
                            "max_mean_kl": max(r["mean_kl_dense_to_candidate"] for r in rows)})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--corpus", default="data/smoke.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_precision(args.config, args.corpus, args.output)


if __name__ == "__main__":
    main()

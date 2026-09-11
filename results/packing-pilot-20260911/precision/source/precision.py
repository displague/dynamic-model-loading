"""Diagnose arithmetic-order sensitivity without relaxing the original gates."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import extract_ffns
from .experiment import digest, environment, layout, load_corpus, validate_config, write_json
from .ffn import dimensions
from .metrics import compare_logits


MODES = ("bf16", "reduction_off", "fp32_down", "fp32_ffn", "canonical_down")


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


@torch.inference_mode()
def run_precision(config_path, corpus_path, output_path):
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
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "config.json")
    shutil.copyfile(corpus_path, output / "corpus.jsonl")
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    corpus = [r for r in load_corpus(Path(corpus_path)) if r["split"] == "diagnostic"]
    checkpoint = Path(snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True))
    manifest = {"purpose": "numerical_diagnosis_not_sparse_quality_evaluation", "modes": MODES,
                "config_sha256": digest(Path(config_path)), "corpus_sha256": digest(Path(corpus_path)),
                "model": cfg["model"], "revision": cfg["revision"],
                "environment": environment(torch.device("cuda")),
                "sources": {p.name: digest(p) for p in (output / "source").glob("*.py")},
                "checkpoint_files": {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).cuda().eval()
    mlps = extract_ffns(model)
    tokens = {r["id"]: tokenizer(r["text"], return_tensors="pt", add_special_tokens=False,
                               truncation=True, max_length=cfg["max_tokens"])["input_ids"].cuda() for r in corpus}
    generator = torch.Generator().manual_seed(cfg["seed"])
    native = [torch.arange(dimensions(m)[1]) for m in mlps]
    random_order = [torch.randperm(dimensions(m)[1], generator=generator) for m in mlps]

    def logits(ids):
        return model(input_ids=ids, use_cache=False).logits.cpu()

    with arithmetic(mlps, "bf16", native):
        original = {key: logits(ids) for key, ids in tokens.items()}
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

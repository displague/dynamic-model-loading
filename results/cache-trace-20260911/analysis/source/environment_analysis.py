"""Validate and compare the pinned PyTorch arithmetic controls; no timing inference."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from safetensors.torch import load_file
from safetensors import safe_open
import torch

from .experiment import digest, write_json
from .metrics import compare_logits
from .precision import MODES, validate_control


COMPARISONS = ("permutation_within_policy", "policy_vs_original_bf16")


def validate_pair(cfg_a, cfg_b, env_a, env_b):
    without_inventory = lambda cfg: {k: v for k, v in cfg.items() if k != "expected_versions_sha256"}
    if without_inventory(cfg_a) != without_inventory(cfg_b):
        raise ValueError("Cross-environment experimental config drift")
    stable = ("python", "platform", "device", "cuda_build", "versions", "tf32_matmul", "tf32_cudnn",
              "bf16_reduced_precision_reduction", "fp16_reduced_precision_reduction", "cpu_threads", "gpu")
    if any(env_a[k] != env_b[k] for k in stable):
        raise ValueError("Cross-environment execution settings drift")
    for cfg, env in ((cfg_a, env_a), (cfg_b, env_b)):
        if env["cpu_threads"] != cfg["cpu_threads"] or env["tf32_matmul"] or env["tf32_cudnn"]:
            raise ValueError("Control execution differs from declared settings")


def validate_rows(rows, summary, cfg, tokens):
    if cfg["logit_relative_l2_max"] != 0.01 or cfg["logit_mean_kl_max"] != 0.001:
        raise ValueError("Arithmetic comparisons require the original numerical limits")
    expected = set(itertools.product(MODES, COMPARISONS, tokens))
    key = lambda r: (r["mode"], r["comparison"], r["document"])
    if len(rows) != len(expected) or {key(r) for r in rows} != expected:
        raise ValueError("Incomplete or duplicate arithmetic comparisons")
    derived = []
    for row in rows:
        if row["predicted_tokens"] != len(tokens[row["document"]][0])-1:
            raise ValueError("Arithmetic prediction count mismatch")
        for k, value in row.items():
            if type(value) in (int, float) and not math.isfinite(value):
                raise ValueError("Nonfinite arithmetic metric")
        passed = (0 <= row["logit_relative_l2"] <= cfg["logit_relative_l2_max"] and
                  0 <= row["mean_kl_dense_to_candidate"] <= cfg["logit_mean_kl_max"])
        if passed != row["within_original_limits"]:
            raise ValueError("Arithmetic pass flag mismatch")
    for mode, comparison in itertools.product(MODES, COMPARISONS):
        selected = [r for r in rows if (r["mode"], r["comparison"]) == (mode, comparison)]
        derived.append({"mode": mode, "comparison": comparison,
            "all_within_original_limits": all(r["within_original_limits"] for r in selected),
            "max_relative_l2": max(r["logit_relative_l2"] for r in selected),
            "max_mean_kl": max(r["mean_kl_dense_to_candidate"] for r in selected)})
    if derived != summary:
        raise ValueError("Arithmetic aggregate mismatch")


def load_ordinary_logits(root, tokens):
    root = Path(root)
    logits_receipt = json.loads((root / "ordinary-logits-receipt.json").read_text())
    if digest(root / "ordinary-logits.safetensors") != logits_receipt["sha256"]:
        raise ValueError("Ordinary logits hash mismatch")
    with safe_open(root / "ordinary-logits.safetensors", framework="pt") as stored:
        if stored.metadata() != {"manifest_sha256": digest(root / "manifest.json"), "comparison": "ordinary_native_bf16"}:
            raise ValueError("Ordinary logits provenance mismatch")
    logits = load_file(root / "ordinary-logits.safetensors")
    if set(logits) != set(tokens) or set(logits_receipt["documents"]) != set(tokens):
        raise ValueError("Ordinary logits document mismatch")
    return logits


def load_run(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    cfg, manifest, tokens = read("config.json"), read("manifest.json"), read("token-ids.json")
    if (root / "failure.json").exists() or not cfg.get("environment_control"):
        raise ValueError("Completed environment control required")
    for filename, key in (("config.json", "config_sha256"), ("corpus.jsonl", "corpus_sha256"),
                          ("protocol.md", "protocol_sha256"), ("environment-inventory.json", "inventory_sha256"),
                          ("token-ids.json", "token_ids_file_sha256"), ("layouts.json", "layouts_sha256")):
        if digest(root / filename) != manifest[key]:
            raise ValueError(f"Control hash mismatch: {filename}")
    for name, expected in manifest["sources"].items():
        if digest(root / "source" / name) != expected:
            raise ValueError("Control source mismatch")
    validate_control(cfg, read("environment-inventory.json"), root / "corpus.jsonl")
    if manifest["checkpoint_files"] != cfg["expected_checkpoint_files"] or manifest["layouts_sha256"] != cfg["expected_layouts_sha256"]:
        raise ValueError("Control checkpoint/layout mismatch")
    if hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest() != cfg["expected_token_ids_sha256"]:
        raise ValueError("Control tokens mismatch")
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    validate_rows(rows, read("summary.json"), cfg, tokens)
    logits = load_ordinary_logits(root, tokens)
    return cfg, manifest, tokens, rows, logits


def run(baseline, candidate, parent, output):
    baseline, candidate, parent, output = map(Path, (baseline, candidate, parent, output))
    a, b = load_run(baseline), load_run(candidate)
    validate_pair(a[0], b[0], a[1]["environment"], b[1]["environment"])
    if a[2] != b[2] or any(a[1][k] != b[1][k] for k in ("sources", "checkpoint_files", "layouts_sha256", "protocol_sha256")):
        raise ValueError("Cross-environment input/source mismatch")
    if a[1]["environment"]["torch"] != "2.10.0+cu130" or b[1]["environment"]["torch"] != "2.12.0+cu130":
        raise ValueError("Declared torch control pair required")
    inv_a = json.loads((baseline / "environment-inventory.json").read_text())
    inv_b = json.loads((candidate / "environment-inventory.json").read_text())
    if inv_a["python"] != inv_b["python"]:
        raise ValueError("Cross-environment interpreter drift")
    differences = {k: [inv_a["versions"].get(k), inv_b["versions"].get(k)] for k in
        inv_a["versions"].keys() | inv_b["versions"].keys() if inv_a["versions"].get(k) != inv_b["versions"].get(k)}
    if set(differences) != {"torch", "torchvision"}:
        raise ValueError("Unexpected distribution drift")
    if digest(parent / "results.jsonl") != a[0]["expected_parent_results_sha256"]:
        raise ValueError("Historical raw control anchor mismatch")
    if digest(parent / "config.json") != a[0]["expected_parent_config_sha256"]:
        raise ValueError("Historical configuration anchor mismatch")
    historical_cfg = json.loads((parent / "config.json").read_text())
    if any(a[0][k] != v for k, v in historical_cfg.items() if k != "purpose"):
        raise ValueError("Control settings differ from archived original configuration")
    previous = [json.loads(line) for line in (parent / "results.jsonl").read_text().splitlines()]
    key = lambda r: (r["mode"], r["comparison"], r["document"])
    if len(previous) != len(a[3]) or {key(r) for r in previous} != {key(r) for r in a[3]}:
        raise ValueError("Historical control coverage mismatch")
    repeat_differences = []
    for row in a[3]:
        old = next(r for r in previous if key(r) == key(row))
        if old != row:
            repeat_differences.append({"mode": row["mode"], "comparison": row["comparison"],
                "document": row["document"], "changed_fields": {k: [old.get(k), v] for k, v in row.items() if old.get(k) != v}})
    cross = [{"document": name, **compare_logits(a[4][name], b[4][name], torch.tensor(a[2][name]))} for name in a[2]]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"inputs": {label: {p.name: digest(p) for p in root.iterdir() if p.is_file()}
        for label, root in (("baseline", baseline), ("candidate", candidate), ("parent", parent))},
        "analysis_source_sha256": digest(Path(__file__))})
    result = {"status": "complete", "distribution_differences": differences,
        "baseline_repeat_exact": not repeat_differences, "baseline_repeat_differences": repeat_differences,
        "cross_environment_ordinary_bf16": cross,
        "baseline_policies": json.loads((baseline / "summary.json").read_text()),
        "candidate_policies": json.loads((candidate / "summary.json").read_text())}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "parent", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.parent, args.output)


if __name__ == "__main__":
    main()

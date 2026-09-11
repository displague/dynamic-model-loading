"""Validate complete applied traces, then publish same-budget traffic simulations."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
from safetensors.torch import load_file

from .cache import POLICIES, dense_static_frontier, group_arrays, replay
from .experiment import digest, load_corpus, write_json
from .metrics import aggregate
from .trace import load_trace


def validate_gates(rows, cfg, docs, layers):
    for kind, dimension, expected in (
        ("group_reconstruction", "layer", set(itertools.product(cfg["layouts"], range(layers)))),
        ("layout_correctness", "document", set(itertools.product(cfg["layouts"], docs))),
    ):
        checks = [r for r in rows if r["kind"] == kind]
        if len(checks) != len(expected) or {(r["layout"], r[dimension]) for r in checks} != expected:
            raise ValueError("Incomplete or duplicate numerical gates")
        for row in checks:
            limits = (("relative_l2", "reconstruction_relative_l2_max"),) if kind == "group_reconstruction" else (
                ("logit_relative_l2", "logit_relative_l2_max"), ("mean_kl_dense_to_candidate", "logit_mean_kl_max"))
            if row["passed"] is not True or any(not math.isfinite(row[k]) or not 0 <= row[k] <= cfg[limit] for k, limit in limits):
                raise ValueError("Numerical gate measurement exceeds tolerance")


def run(root, output):
    root, output = Path(root), Path(output)
    cfg = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (not cfg.get("capture_traces") or not summary["correctness_passed"] or
            summary["status"] not in ("smoke_diagnostics_completed", "packing_pilot_completed")):
        raise ValueError("Complete, correct trace run required")
    for filename, key in (("config.json", "config_sha256"), ("corpus.jsonl", "corpus_sha256"),
                          ("protocol.md", "protocol_sha256")):
        if digest(root / filename) != manifest[key]:
            raise ValueError(f"Input hash mismatch: {filename}")
    for name, expected in manifest["sources"].items():
        if digest(root / "source" / name) != expected:
            raise ValueError("Source snapshot mismatch")
    if cfg["dtype"] != "float32" or manifest["corpus_sha256"] != cfg["expected_corpus_sha256"] or manifest["environment"]["torch"] != cfg["expected_torch"]:
        raise ValueError("Pinned environment/corpus mismatch")
    token_ids = json.loads((root / "token-ids.json").read_text())
    if hashlib.sha256(json.dumps(token_ids, sort_keys=True).encode()).hexdigest() != cfg["expected_token_ids_sha256"]:
        raise ValueError("Exact token IDs mismatch")
    corpus = load_corpus(root / "corpus.jsonl")
    docs = [r["id"] for r in corpus if r["split"] == "diagnostic"]
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    if any(r["kind"] == "error" or ("passed" in r and not r["passed"]) for r in rows):
        raise ValueError("Failed raw run")
    calibration = [r for r in rows if r["kind"] == "calibration"]
    if len(calibration) != 1:
        raise ValueError("Missing or duplicate calibration receipt")
    for filename, expected in (("layouts.json", cfg["expected_layouts_sha256"]),
                               ("calibration-importance.safetensors", calibration[0]["importance_sha256"])):
        if digest(root / filename) != expected:
            raise ValueError(f"Calibration hash mismatch: {filename}")
    layouts = json.loads((root / "layouts.json").read_text())
    shapes = manifest["ffn_shapes"]
    if len({tuple(shape) for shape in shapes}) != 1:
        raise ValueError("Homogeneous FFNs required")
    layers, (hidden, neurons) = len(shapes), shapes[0]
    validate_gates(rows, cfg, docs, layers)
    means_file = load_file(root / "calibration-importance.safetensors")
    if set(means_file) != {str(i) for i in range(layers)}:
        raise ValueError("Calibration layer mismatch")
    means = np.stack([means_file[str(i)].numpy() for i in range(layers)])
    if means.shape != (layers, neurons):
        raise ValueError("Calibration neuron mismatch")
    dense_frontier = dense_static_frontier(means, {name: layouts[name] for name in cfg["layouts"]},
        cfg["group_widths"], cfg["cache_budgets_bytes"], 3*hidden*4, [len(token_ids[d][0]) for d in docs])
    conditions = list(itertools.product(cfg["layouts"], cfg["group_widths"], cfg["keep_fractions"]))
    key = lambda r: (r["layout"], r["group_width"], r["keep_fraction"])
    probes = [r for r in rows if r["kind"] == "hindsight_document"]
    aggregates = [r for r in rows if r["kind"] == "hindsight_aggregate"]
    if len(probes) != len(conditions)*len(docs) or {(key(r), r["document"]) for r in probes} != set(itertools.product(conditions, docs)):
        raise ValueError("Missing, duplicate, or unexpected document conditions")
    if len(aggregates) != len(conditions) or {key(r) for r in aggregates} != set(conditions):
        raise ValueError("Missing or duplicate aggregate conditions")
    if len(summary["probes"]) != len(conditions) or {key(r) for r in summary["probes"]} != set(conditions):
        raise ValueError("Incomplete summary conditions")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    write_json(output / "manifest.json", {"input_files": {p.name: digest(p) for p in root.iterdir() if p.is_file()},
        "sources": {p.name: digest(p) for p in (output / "source").glob("*.py")},
        "semantics": "teacher_forced_prefill_masks_in_hypothetical_token_major_order_no_physical_transfers"})
    records = (output / "replay.jsonl").open("x", encoding="utf-8")
    results = []
    try:
        for condition in conditions:
            name, width, keep = condition
            metadata = dict(layout=name, group_width=width, keep_fraction=keep)
            documents = [r for r in probes if key(r) == condition]
            quality = aggregate(documents)
            parent = next(r for r in aggregates if key(r) == condition)
            compact = next(r for r in summary["probes"] if key(r) == condition)
            if any(parent[k] != v or compact[k] != v for k, v in quality.items()):
                raise ValueError("Raw quality aggregate mismatch")
            sizes, importance = group_arrays(means, np.asarray(layouts[name]), width, 3*hidden*4)
            totals = {}
            applied_bytes = 0
            geometry = {k: 0 for k in ("top_individual_neurons", "top_individual_neurons_in_applied_mask",
                                      "neurons_in_individual_cover_groups", "applied_neurons")}
            for row in documents:
                receipt = row["trace"]
                trace_path = root / receipt["path"]
                if trace_path.resolve().parent != (root / "traces").resolve():
                    raise ValueError("Trace path outside trace directory")
                trace = load_trace(trace_path, receipt)
                for k, v in {**metadata, "document": row["document"], "neurons": neurons,
                             "bytes_per_neuron": 3*hidden*4, "bitorder": "little"}.items():
                    if receipt[k] != v:
                        raise ValueError("Applied trace condition mismatch")
                if trace.shape != (len(token_ids[row["document"]][0]), layers, math.ceil(neurons/width)):
                    raise ValueError("Trace token/layer/group shape mismatch")
                if not (trace.sum(axis=2) == math.ceil(trace.shape[2]*keep)).all():
                    raise ValueError("Trace selection count mismatch")
                trace_bytes = int((trace.sum(axis=0) * sizes).sum())
                applied_bytes += trace_bytes
                if len(receipt["geometry_per_layer"]) != layers:
                    raise ValueError("Trace geometry layer mismatch")
                if sum(g["applied_neurons"] for g in receipt["geometry_per_layer"]) * 3*hidden*4 != trace_bytes:
                    raise ValueError("Applied trace bytes mismatch geometry")
                for g in receipt["geometry_per_layer"]:
                    for k in geometry:
                        geometry[k] += g[k]
                dense = np.ones_like(trace)
                for budget in cfg["cache_budgets_bytes"]:
                    for policy in POLICIES:
                        sparse_rows = replay(trace, sizes, importance, budget, policy)
                        dense_rows = replay(dense, sizes, importance, budget, policy)
                        for sparse, baseline in zip(sparse_rows, dense_rows, strict=True):
                            record = {**metadata, "document": row["document"], "trace_sha256": receipt["sha256"],
                                      **sparse, "dense_baseline": baseline}
                            records.write(json.dumps(record, allow_nan=False) + "\n")
                            total_key = (budget, policy, sparse["state"])
                            accumulator = totals.setdefault(total_key, {k: 0 for k in
                                ("demand_hits", "demand_misses", "demand_bytes", "preload_bytes", "total_bytes", "dense_total_bytes")})
                            for k in accumulator:
                                accumulator[k] += baseline["total_bytes"] if k == "dense_total_bytes" else sparse[k]
                records.flush()
            if applied_bytes != parent["hypothetical_selected_weight_bytes"]:
                raise ValueError("Trace geometry and quality accounting mismatch")
            for (budget, policy, state), total in totals.items():
                dense_best = dense_frontier[(budget, state)]["total_bytes"]
                savings = 1-total["total_bytes"]/dense_best if dense_best else None
                result = {**metadata, "budget_bytes": budget, "policy": policy, "state": state, **total,
                          **quality, "strongest_dense_static_bytes": dense_best,
                          "strongest_dense_static_configuration": dense_frontier[(budget, state)],
                          "savings_vs_dense_static": savings,
                          "development_screen_passed": state == "warm" and quality["relative_perplexity"] <= cfg["screen_relative_ppl_max"] and savings is not None and savings >= cfg["screen_warm_savings_min"],
                          "individual_top_neuron_coverage": geometry["top_individual_neurons_in_applied_mask"]/geometry["top_individual_neurons"],
                          "individual_cover_amplification": geometry["neurons_in_individual_cover_groups"]/geometry["top_individual_neurons"]}
                results.append(result)
            print(f"Replayed {name} width={width} keep={keep}", flush=True)
        passing = sorted((r for r in results if r["development_screen_passed"]), key=lambda r:
                         (r["total_bytes"], r["mean_kl_dense_to_candidate"], r["layout"], r["group_width"],
                          r["keep_fraction"], r["budget_bytes"], r["policy"]))
        write_json(output / "summary.json", {"status": "complete", "conditions": len(conditions),
                   "documents": len(docs), "fixed_non_ffn_parameter_bytes": manifest["model_parameter_bytes"]-manifest["ffn_parameter_bytes"],
                   "passing_warm_rows": len(passing), "prototype_choice": passing[0] if passing else None,
                   "rows": results})
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        records.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)


if __name__ == "__main__":
    main()

"""Summarize the predeclared packing comparison from completed raw receipts."""

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import digest, write_json


def paired_kl_interval(rows, group_width=32, keep_fraction=0.75, seed=1729, replicates=2000):
    pairs = {}
    for name in ("popularity", "coactivation"):
        selected = [r for r in rows if r.get("kind") == "hindsight_document" and r["layout"] == name
                    and r["group_width"] == group_width and r["keep_fraction"] == keep_fraction]
        pairs[name] = {r["document"]: r for r in selected}
        if len(selected) != len(pairs[name]):
            raise ValueError("Duplicate document measurements")
    if not pairs["popularity"] or pairs["popularity"].keys() != pairs["coactivation"].keys():
        raise ValueError("Paired document sets must be nonempty and identical")
    differences, weights = [], []
    for name in sorted(pairs["popularity"]):
        p, c = pairs["popularity"][name], pairs["coactivation"][name]
        if p["predicted_tokens"] != c["predicted_tokens"] or p["predicted_tokens"] <= 0:
            raise ValueError("Mismatched or empty token denominators")
        differences.append(c["mean_kl_dense_to_candidate"] - p["mean_kl_dense_to_candidate"])
        weights.append(p["predicted_tokens"])
    differences, weights = np.asarray(differences), np.asarray(weights)
    if not np.isfinite(differences).all():
        raise ValueError("Nonfinite paired measurements")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(weights), size=(replicates, len(weights)))
    resampled = (differences[indices] * weights[indices]).sum(1) / weights[indices].sum(1)
    return {"comparison": "coactivation_minus_popularity_mean_KL", "group_width": group_width,
            "keep_fraction": keep_fraction, "documents": len(weights), "bootstrap_replicates": replicates,
            "bootstrap_seed": seed, "delta": float(np.average(differences, weights=weights)),
            "percentile_95_interval": np.quantile(resampled, [0.025, 0.975]).tolist(),
            "interpretation": "Exploratory paired whole-document interval, not confirmatory inference"}


def analyze(run_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(run_path), Path(output_path)
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "packing_pilot_completed" or not summary["correctness_passed"]:
        raise ValueError("A completed pilot with passing correctness checks is required")
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    interval = paired_kl_interval(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", {**interval, "raw_results_sha256": digest(root / "results.jsonl"),
                                          "numpy_version": np.__version__})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    colors = {"native": "#8b98a7", "random": "#d58b52", "popularity": "#3677b1", "coactivation": "#239074"}
    for ax, fraction in zip(axes, (0.5, 0.75)):
        for name, color in colors.items():
            points = sorted([p for p in summary["probes"] if p["layout"] == name and p["keep_fraction"] == fraction],
                            key=lambda p: p["group_width"])
            ax.plot([p["group_width"] for p in points], [p["mean_kl_dense_to_candidate"] for p in points],
                    "o-", label=name, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 32, 128], ["1", "32", "128"])
        ax.set_xlabel("Neurons per group")
        ax.set_title(f"Retain {fraction:.0%} of groups")
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("Mean KL(dense || masked), log scale")
    axes[1].legend(frameon=False)
    fig.suptitle("WikiText packing pilot: FP32, 16 development articles\nHindsight masks; no measured paging speedup", fontsize=12)
    fig.savefig(output / "packing.png", dpi=180)
    fig.savefig(output / "packing.svg")
    plt.close(fig)
    print(json.dumps(interval, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)


if __name__ == "__main__":
    main()

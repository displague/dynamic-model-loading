"""Standalone quality/traffic figure from the complete cache replay summary."""

import argparse
import json
from pathlib import Path

from .experiment import digest, write_json


def plot(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, output = Path(root), Path(output)
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Complete cache replay required")
    output.mkdir(parents=True, exist_ok=False)
    rows = summary["rows"]
    budget = max(r["budget_bytes"] for r in rows)
    layouts = sorted({r["layout"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout="constrained")
    colors = dict(zip(layouts, ("#4477aa", "#66a0a0", "#228833", "#cc6677")))
    for width, marker in ((8, "o"), (32, "s"), (128, "^")):
        for name in layouts:
            values = [r for r in rows if r["state"] == "warm" and r["budget_bytes"] == budget
                      and r["layout"] == name and r["group_width"] == width and r["policy"].startswith("static_")]
            chosen = [min((r for r in values if r["keep_fraction"] == keep), key=lambda r: r["total_bytes"])
                      for keep in sorted({r["keep_fraction"] for r in values})]
            for axis in axes:
                axis.plot([100*r["total_bytes"]/r["strongest_dense_static_bytes"] for r in chosen],
                          [100*(r["relative_perplexity"]-1) for r in chosen], color=colors[name], marker=marker,
                          markersize=4, linewidth=1, label=f"{name}, {width}")
    for axis in axes:
        axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        axis.axvline(90, color="#555555", linestyle=":", linewidth=1)
        axis.set_xlabel("Warm simulated bytes / strongest dense static (%)")
        axis.set_ylabel("Relative perplexity change (%)")
        axis.grid(alpha=.15)
    axes[0].set_title(f"All {summary['conditions']} development conditions; best of two static caches")
    axes[1].set_title("Detail around the prospective development screen")
    axes[1].set_ylim(-.15, 2.5)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(f"Qwen2.5-1.5B FP32, {budget/2**30:g} GiB FFN budget: analytical replay, no measured transfers", fontsize=11)
    fig.savefig(output / "cache-frontier.png", dpi=180)
    fig.savefig(output / "cache-frontier.svg")
    plt.close(fig)
    write_json(output / "provenance.json", {"summary_sha256": digest(root / "summary.json"),
               "plot_source_sha256": digest(Path(__file__)), "budget_bytes": budget,
               "note": "Per-condition lower traffic of two declared static policies; full grid at left, labeled quality zoom at right"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()

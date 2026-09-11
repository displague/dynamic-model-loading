"""Plot the complete declared layer and fixed-block omission results."""

import argparse
import json
from pathlib import Path

from .experiment import digest, write_json
from .metrics import aggregate


def validate_results(root):
    """Require every planned condition and document, including its exact labels."""
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = json.loads((root / "config.json").read_text())
    if digest(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Configuration digest mismatch")
    if summary["status"] != "layer_study_completed" or not summary["correctness_passed"]:
        raise ValueError("Expected a completed passing layer study")
    if digest(root / "results.jsonl") != summary["raw_results_sha256"]:
        raise ValueError("Raw ledger digest mismatch")
    raw = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    expected = {(name, case["name"]): {"layout": name, "condition": case["name"],
                 "condition_kind": case["kind"], "layers": case["layers"],
                 "group_width": cfg["group_width"], "keep_fraction": cfg["keep_fraction"]}
                for name in cfg["layouts"] for case in manifest["conditions"]}
    probes = {(p["layout"], p["condition"]): p for p in summary["probes"]}
    aggregates = {(p["layout"], p["condition"]): p for p in raw if p["kind"] == "omission_aggregate"}
    docs = [p for p in raw if p["kind"] == "omission_document"]
    if (len(probes) != len(summary["probes"]) or probes.keys() != expected.keys()
            or aggregates.keys() != expected.keys()
            or sum(r["kind"] == "omission_aggregate" for r in raw) != len(expected)
            or {(r["layout"], r["condition"]) for r in docs} != expected.keys()):
        raise ValueError("Missing, duplicated, or unexpected condition")
    for key, common in expected.items():
        p = probes[key]
        rows = [r for r in docs if (r["layout"], r["condition"]) == key]
        if (len(rows) != len(manifest["diagnostic_documents"])
                or {r["document"] for r in rows} != set(manifest["diagnostic_documents"])):
            raise ValueError("Missing, duplicated, or unexpected document")
        if any(any(row[k] != v for k, v in common.items()) for row in [p, *rows]):
            raise ValueError("Condition metadata differs from the declared plan")
        if {k: v for k, v in aggregates[key].items() if k != "kind"} != p:
            raise ValueError("Summary differs from its raw aggregate receipt")
        if any(p[k] != v for k, v in aggregate(rows).items()):
            raise ValueError("Summary does not reproduce from raw measurements")
    return summary


def plot(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(run), Path(output)
    summary = validate_results(root)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    colors = {"popularity": "#3677b1", "coactivation": "#239074"}
    for offset, (name, color) in enumerate(colors.items()):
        points = sorted([p for p in summary["probes"] if p["layout"] == name and p["condition_kind"] == "single"],
                        key=lambda p: p["layers"][0])
        axes[0].plot([p["layers"][0] for p in points], [p["mean_kl_dense_to_candidate"] for p in points],
                     "o-", color=color, label=name, markersize=3)
        blocks = [p for p in summary["probes"] if p["layout"] == name and p["condition_kind"] in ("block", "all")]
        axes[1].bar([i + (offset - 0.5) * 0.35 for i in range(len(blocks))],
                    [100 * (p["relative_perplexity"] - 1) for p in blocks], width=0.35, color=color, label=name)
    axes[0].set(xlabel="Masked layer (zero-based)", ylabel="Mean KL(dense || masked)",
                title="One layer masked at a time")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(len(blocks)), ["All" if p["condition_kind"] == "all"
                         else f"{min(p['layers'])}-{max(p['layers'])}" for p in blocks])
    axes[1].set(xlabel="Masked layers", ylabel="Perplexity change versus dense (%)", title="Declared joint interventions")
    axes[1].axhline(0, color="#8b98a7", linewidth=0.7)
    fig.suptitle("FFN omission: 32 neurons/group, 75% retained in masked layers\nFP32 development study; hypothetical weight volume only", fontsize=12)
    fig.savefig(output / "layers.png", dpi=180)
    fig.savefig(output / "layers.svg")
    plt.close(fig)
    write_json(output / "provenance.json", {"raw_sha256": digest(root / "results.jsonl"),
                "summary_sha256": digest(root / "summary.json"), "source_sha256": digest(Path(__file__)),
                "matplotlib_version": matplotlib.__version__, "all_aggregate_metrics_verified": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.run, args.output)


if __name__ == "__main__":
    main()

"""Validate and summarize a complete ReLU ledger without loading the model."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file

from .experiment import digest, validate_config, write_json
from .metrics import aggregate, compare_logits


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_metrics(row, predicted):
    keys = ('mean_kl_dense_to_candidate','top1_agreement','dense_nll','candidate_nll',
            'relative_perplexity','logit_relative_l2','logit_max_abs_error')
    require(all(k in row and isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0
                for k in keys), 'Invalid document metrics')
    require(row['predicted_tokens'] == predicted and 0 <= row['top1_agreement'] <= 1,
            'Invalid metric denominator or agreement')
    delta = row['candidate_nll']-row['dense_nll']
    require(row['relative_perplexity'] > 0 and
            math.isclose(math.log(row['relative_perplexity']),delta,rel_tol=1e-10,abs_tol=1e-12),
            'Document perplexity mismatch')


def analyze(path):
    path = Path(path)
    read = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    cfg, manifest, summary = (read(name) for name in ("config.json", "manifest.json", "summary.json"))
    validate_config(cfg)
    require(summary["status"] == "relu_control_completed" and summary["correctness_passed"] is True
            and summary["exact_zero_passed"] is True, "Incomplete or failed ReLU run")
    require((cfg["reconstruction_relative_l2_max"], cfg["logit_relative_l2_max"],
             cfg["logit_mean_kl_max"]) == (.01, .01, .001), "Changed numerical limits")
    require(cfg["dtype"] == "float32" and cfg["layouts"] == ["native", "random", "popularity"]
            and cfg["keep_fractions"] == [.75], "Changed ReLU policy")
    for file, key in (("config.json", "config_sha256"), ("corpus.jsonl", "corpus_sha256"),
                      ("protocol.md", "protocol_sha256"), ("token-ids.json", "token_ids_file_sha256")):
        require(digest(path / file) == manifest[key], f"Hash mismatch: {file}")
    require(manifest["corpus_sha256"] == cfg["expected_corpus_sha256"], "Unpinned corpus")
    require(manifest["checkpoint_files"]["pytorch_model.bin"] == cfg["expected_weights_sha256"], "Unpinned weights")
    require(manifest["model"] == cfg["model"] and manifest["revision"] == cfg["revision"], "Model mismatch")
    required_sources = {'__init__.py','relu_study.py','relu.py','experiment.py','ffn.py','metrics.py',
                        'adapters.py','packing.py','trace.py'}
    require(required_sources <= set(manifest['sources']), 'Incomplete producer source inventory')
    for name, value in manifest["sources"].items():
        require(digest(path / "source" / name) == value, f"Source mismatch: {name}")
    startup = read('started.json')
    require(set(startup) == {'started_utc','environment','config_sha256','corpus_sha256','protocol_sha256','sources'},
            'Incomplete startup receipt')
    require(all(manifest[k] == v for k, v in startup.items()), "Started receipt mismatch")
    env = manifest['environment']
    require({'python','executable','platform','device','torch','cuda_build','versions','tf32_matmul','tf32_cudnn',
             'bf16_reduced_precision_reduction','fp16_reduced_precision_reduction','cpu_threads','process_id'} <= set(env),
            'Incomplete environment inventory')
    require(env['python'].split()[0] == '3.14.3' and env['versions'] ==
            {'transformers':'5.13.1','safetensors':'0.8.0','huggingface-hub':'1.19.0','numpy':'2.3.5'},
            'Declared baseline version mismatch')
    device = torch.device(env['device'])
    require(env['torch'] == cfg['expected_torch'] and env['cpu_threads'] == cfg['cpu_threads']
            and env['tf32_matmul'] is False and env['tf32_cudnn'] is False
            and device.type in ('cpu','cuda') and manifest['attention_implementation'] == 'sdpa',
            'Execution policy mismatch')
    if device.type == 'cuda':
        require(env['cuda_build'] == '13.0' and {'gpu','nvidia_smi'} <= set(env)
                and {'name','total_bytes','compute_capability'} <= set(env['gpu']), 'Incomplete CUDA provenance')
    shapes = cfg["expected_shapes"]
    require(manifest["ffn_shapes"] == shapes, "FFN shape mismatch")
    groupable = sum((2*h+1)*n*4 for h, n in shapes)
    biases = sum(h*4 for h, n in shapes)
    require(manifest["groupable_ffn_parameter_bytes"] == groupable
            and manifest["fixed_ffn_output_bias_bytes"] == biases
            and manifest["diagnostic_layout_backup_bytes"] == groupable
            and manifest["largest_layer_transaction_payload_bytes"] == max((2*h+1)*n*8 for h,n in shapes)
            and manifest["fixed_non_groupable_parameter_bytes"] == manifest["model_parameter_bytes"]-groupable,
            "Parameter accounting mismatch")
    corpus = [json.loads(line) for line in (path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    tokens = read("token-ids.json")
    require(set(tokens) == {r["id"] for r in corpus}, "Token document mismatch")
    require(hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest() == manifest["token_ids_sha256"],
            "Token digest mismatch")
    require(all(len(ids) == 1 and 2 <= len(ids[0]) <= cfg["max_tokens"] for ids in tokens.values()), "Token shape mismatch")
    docs = [r["id"] for r in corpus if r["split"] == "diagnostic"]
    cal = [r["id"] for r in corpus if r["split"] == "calibration"]
    require(len(docs)+len(cal) == len(tokens) and len(set(docs+cal)) == len(corpus), "Corpus split mismatch")
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    conditions = [(mode, layout, width) for mode in ("exact_zero", "approximate_75")
                  for layout in cfg["layouts"] for width in cfg["group_widths"]]
    expected_counts = {"dense_reference": len(docs), "calibration": 1,
        "group_reconstruction": len(shapes)*len(cfg["layouts"]), "layout_correctness": len(docs)*len(cfg["layouts"]),
        "relu_document": len(conditions)*len(docs), "relu_aggregate": len(conditions)}
    require(Counter(r["kind"] for r in rows) == expected_counts, "Incomplete ledger counts")
    by_kind = {kind: [r for r in rows if r["kind"] == kind] for kind in expected_counts}
    require([r["document"] for r in by_kind["dense_reference"]] == docs, "Dense documents mismatch")
    dense_nll = {}
    for index, row in enumerate(by_kind["dense_reference"]):
        require(row["input_tokens"] == len(tokens[row["document"]][0]), "Dense token count mismatch")
        require(digest(path/'reference_logits'/f'{index:03d}.safetensors') == row.get('logits_sha256'),
                'Dense reference artifact mismatch')
        logits = load_file(path/'reference_logits'/f'{index:03d}.safetensors')['logits']
        dense_nll[row['document']] = compare_logits(logits,logits,torch.tensor(tokens[row['document']]))['dense_nll']
    calibration = by_kind["calibration"][0]
    require(calibration["documents"] == cal and calibration["token_observations_per_layer"] ==
            [sum(len(tokens[d][0]) for d in cal)]*len(shapes), "Calibration split mismatch")
    require(digest(path / "layouts.json") == calibration["layouts_sha256"], "Layout hash mismatch")
    layouts = read("layouts.json")
    require(set(layouts) == set(cfg["layouts"]), "Layout grid mismatch")
    for name, orders in layouts.items():
        require(len(orders) == len(shapes), "Layer count mismatch")
        for order, (h,n) in zip(orders, shapes, strict=True):
            require(sorted(order) == list(range(n)), "Invalid permutation")
            if name == "native":
                require(order == list(range(n)), "Native layout changed")
    local = by_kind["group_reconstruction"]
    full = by_kind["layout_correctness"]
    for row in full:
        validate_metrics(row,len(tokens[row['document']][0])-1)
        require(row['dense_nll'] == dense_nll[row['document']], 'Dense NLL differs from reference artifact')
    require([(r["layout"],r["layer"]) for r in local] ==
            [(l,i) for l in cfg["layouts"] for i in range(len(shapes))], "Local gate grid mismatch")
    require([(r["layout"],r["document"]) for r in full] ==
            [(l,d) for l in cfg["layouts"] for d in docs], "Full gate grid mismatch")
    require(all(r["passed"] is True and 0 <= r["relative_l2"] <= .01 for r in local), "Local numerical failure")
    require(all(r["passed"] is True and 0 <= r["logit_relative_l2"] <= .01 and
                0 <= r["mean_kl_dense_to_candidate"] <= .001 for r in full), "Full numerical failure")
    expected_order = ["dense_reference"]*len(docs)+["calibration"]
    for layout in cfg["layouts"]:
        expected_order += ["group_reconstruction"]*len(shapes)+["layout_correctness"]*len(docs)
    for condition in conditions:
        expected_order += ["relu_document"]*len(docs)+["relu_aggregate"]
    require([r["kind"] for r in rows] == expected_order, "Gate/probe ordering mismatch")
    require([(r["mode"],r["layout"],r["group_width"],r["document"]) for r in by_kind["relu_document"]] ==
            [(*c,d) for c in conditions for d in docs], "Probe grid mismatch")
    recomputed = []
    all_docs = by_kind["relu_document"]
    for index, (mode, layout, width) in enumerate(conditions):
        measurements = all_docs[index*len(docs):(index+1)*len(docs)]
        accounts = []
        for row in measurements:
            count = len(tokens[row["document"]][0])
            validate_metrics(row,count-1)
            require(row['dense_nll'] == dense_nll[row['document']], 'Dense NLL differs from reference artifact')
            require(row["predicted_tokens"] == count-1, "Prediction denominator mismatch")
            require(len(row["layer_accounting"]) == len(shapes), "Missing layer accounting")
            if mode == "exact_zero":
                require(row["passed"] is True and 0 <= row["logit_relative_l2"] <= .01 and
                        0 <= row["mean_kl_dense_to_candidate"] <= .001, "Exact-zero numerical failure")
            for a, (h,n) in zip(row["layer_accounting"], shapes, strict=True):
                require(all(isinstance(v,int) and v >= 0 for v in a.values()), "Invalid integer accounting")
                require(a["token_observations"] == count and a["neuron_observations"] == count*n
                        and a["group_observations"] == count*math.ceil(n/width), "Observation mismatch")
                require(a["zero_neurons"] <= count*n and a["zero_groups"] <= a["group_observations"]
                        and a["selected_neurons"] <= count*n, "Impossible selection count")
                active_groups = a['group_observations']-a['zero_groups']
                active_neurons = count*n-a['zero_neurons']
                tail = width*math.ceil(n/width)-n
                max_active_neurons = active_groups*width-max(0,count-a['zero_groups'])*tail
                require(active_groups <= active_neurons <= max_active_neurons,
                        'Zero neuron/group counts inconsistent')
                require(a["selected_group_weight_bytes"] == a["selected_neurons"]*(2*h+1)*4
                        and a["fixed_output_bias_bytes"] == count*h*4
                        and a["hypothetical_selected_ffn_bytes"] == a["selected_group_weight_bytes"]+count*h*4
                        and a["full_ffn_bytes"] == count*((2*h+1)*n+h)*4, "Biased FFN bytes mismatch")
                if mode == "exact_zero":
                    selected_groups = active_groups
                    require(a["selected_neurons"] >= count*n-a["zero_neurons"], "Active neuron omitted")
                else:
                    selected_groups = count*math.ceil(math.ceil(n/width)*.75)
                difference = selected_groups*width-a['selected_neurons']
                if tail:
                    max_tails = min(count,selected_groups)
                    if mode == 'approximate_75' and selected_groups < count*math.ceil(n/width):
                        # Last-index zero-score tails lose every tie to earlier groups.
                        # A selected tail must then contain a nonzero activation.
                        max_tails = min(max_tails,active_groups)
                    require(difference % tail == 0 and
                            max(0,selected_groups-count*(math.ceil(n/width)-1)) <= difference//tail <= max_tails,
                            'Impossible selected short-tail count')
                    if mode == 'approximate_75' and selected_groups < count*math.ceil(n/width):
                        require(active_neurons <= active_groups*width-(difference//tail)*tail,
                                'Selected short-tail capacity contradicts active neurons')
                else:
                    require(difference == 0, 'Selected group count mismatch')
                accounts.append(a)
        totals = {key: sum(a[key] for a in accounts) for key in accounts[0]}
        result = {"layout": layout, "group_width": width, "mode": mode, **aggregate(measurements), "accounting": totals,
                  "hypothetical_ffn_fraction": totals["hypothetical_selected_ffn_bytes"]/totals["full_ffn_bytes"],
                  "zero_neuron_fraction": totals["zero_neurons"]/totals["neuron_observations"],
                  "zero_group_fraction": totals["zero_groups"]/totals["group_observations"]}
        require(by_kind["relu_aggregate"][index] == {"kind":"relu_aggregate", **result}, "Raw aggregate mismatch")
        recomputed.append(result)
    require(recomputed == summary["probes"], "Summary mismatch")
    zero = [r for r in all_docs if r["mode"] == "exact_zero"]
    return {"status": "validated", "input_hashes": {name: digest(path/name) for name in
            ("manifest.json", "results.jsonl", "summary.json", "config.json", "protocol.md", "corpus.jsonl", "token-ids.json", "layouts.json")},
        "row_counts": expected_counts, "calibration_documents": len(cal), "diagnostic_documents": len(docs),
        "calibration_input_tokens": sum(len(tokens[d][0]) for d in cal),
        "diagnostic_input_tokens": sum(len(tokens[d][0]) for d in docs),
        "diagnostic_predicted_tokens": sum(len(tokens[d][0])-1 for d in docs),
        "maximum_local_l2": max(r["relative_l2"] for r in local),
        "maximum_layout_l2": max(r["logit_relative_l2"] for r in full),
        "maximum_layout_kl": max(r["mean_kl_dense_to_candidate"] for r in full),
        "maximum_exact_zero_l2": max(r["logit_relative_l2"] for r in zero),
        "maximum_exact_zero_kl": max(r["mean_kl_dense_to_candidate"] for r in zero),
        "conditions": recomputed,
        "limitations": "Archived receipt validation; no new model scoring, cache replay, transfer, or runtime claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "relu_analysis.py")
    try:
        write_json(output / "summary.json", analyze(args.run))
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"type":type(error).__name__, "message":str(error)})
        raise


if __name__ == "__main__":
    main()

"""Build a reproducible article-disjoint WikiText development corpus."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from .experiment import digest, write_json


DATASET = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def articles(lines):
    lines = list(lines)
    title, body = None, []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"= ([^=]+) =", line.strip())
        # WikiText also splits some inline equations/table legends into rows
        # that resemble headings. Article titles have blank rows on both sides.
        blank_before = index == 0 or not lines[index - 1].strip()
        blank_after = index + 1 == len(lines) or not lines[index + 1].strip()
        if match and blank_before and blank_after:
            if title is not None:
                yield title, "\n".join(body).strip()
            title, body = match.group(1).strip(), []
        elif title is not None and line.strip():
            body.append(line.strip())
    if title is not None:
        yield title, "\n".join(body).strip()


def build_corpus(config_path, output_dir):
    import pyarrow.parquet as pq

    cfg = json.loads(Path(config_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    all_articles, files = {}, {}
    for split in ("train", "validation"):
        path = Path(hf_hub_download(DATASET, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                                   repo_type="dataset", revision=REVISION, local_files_only=True))
        files[split] = {"filename": path.name, "sha256": digest(path)}
        all_articles[split] = list(articles(pq.read_table(path, columns=["text"])["text"].to_pylist()))
    train_titles = {title.casefold() for title, _ in all_articles["train"]}
    validation_titles = {title.casefold() for title, _ in all_articles["validation"]}
    overlap = train_titles & validation_titles
    seen_text, rows, selection = set(), [], []
    for split, label, count in (("train", "calibration", 80), ("validation", "diagnostic", 16)):
        ranked = sorted(all_articles[split], key=lambda item: hashlib.sha256(
            f"{cfg['seed']}|{split}|{item[0]}".encode()).hexdigest())
        chosen = 0
        seen_titles = set()
        for title, body in ranked:
            if title.casefold() in overlap or title.casefold() in seen_titles:
                continue
            ids = tokenizer.encode(body, add_special_tokens=False)
            if len(ids) < cfg["max_tokens"]:
                continue
            prefix = ids[:cfg["max_tokens"]]
            text = tokenizer.decode(prefix, clean_up_tokenization_spaces=False)
            if tokenizer.encode(text, add_special_tokens=False) != prefix:
                continue  # Exclude a prefix ending inside an undecodable token sequence.
            key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if key in seen_text:
                continue
            seen_text.add(key)
            seen_titles.add(title.casefold())
            document_id = f"wikitext-{split}-{hashlib.sha256(title.encode()).hexdigest()[:16]}"
            rows.append({"id": document_id, "split": label, "domain": "wikipedia", "text": text})
            selection.append({"id": document_id, "title": title, "source_split": split,
                              "tokens": len(prefix), "text_sha256": key})
            chosen += 1
            if chosen == count:
                break
        if chosen != count:
            raise ValueError(f"Only {chosen} eligible {split} articles, expected {count}")
    with (output / "corpus.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset": DATASET, "dataset_revision": REVISION,
                "source": "https://huggingface.co/datasets/Salesforce/wikitext",
                "licenses": ["CC-BY-SA-3.0", "GFDL"], "files": files,
                "excluded_overlapping_titles": sorted(overlap), "selection": selection,
                "model": cfg["model"], "tokenizer_revision": cfg["revision"],
                "seed": cfg["seed"], "max_tokens": cfg["max_tokens"],
                "corpus_sha256": digest(output / "corpus.jsonl"),
                "test_split_accessed": False}
    write_json(output / "provenance.json", manifest)
    print(json.dumps({"documents": len(rows), "calibration_tokens": 80 * cfg["max_tokens"],
                      "diagnostic_tokens": 16 * cfg["max_tokens"], "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/packing-pilot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_corpus(args.config, args.output)


if __name__ == "__main__":
    main()

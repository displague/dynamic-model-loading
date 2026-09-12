# Stock speculation receipts, 2026-09-12

The original native output precedes all compact analysis. The five release
assets below are lossless archives with verified per-file SHA-256 inventories
in [archives.json](archives.json). Original local runs were retained.

| Archive | Compressed bytes | SHA-256 |
|---|---:|---|
| [stock-calibration-complete-20260912.zip](https://github.com/displague/dynamic-model-loading/releases/download/v0.14.0/stock-calibration-complete-20260912.zip) | 4,831,576 | `dfbbe466d82aa917e8a389fd32ab835af36ccec8680327c5b8587968c2c14317` |
| [stock-evaluation-20260912.zip](https://github.com/displague/dynamic-model-loading/releases/download/v0.14.0/stock-evaluation-20260912.zip) | 15,705,515 | `2f077f5ec687bfe68be8159c459e88e84eb1f3f521ee4334b639d2480eb0a096` |
| [stock-replay-20260912.zip](https://github.com/displague/dynamic-model-loading/releases/download/v0.14.0/stock-replay-20260912.zip) | 1,617,887 | `5b88c721b9b7eec09effe6e5e6c2235988a461952c2d049bc33c27a19f5473f9` |
| [stock-startup-attempts-20260912.zip](https://github.com/displague/dynamic-model-loading/releases/download/v0.14.0/stock-startup-attempts-20260912.zip) | 1,296,089 | `3f239d4c6cbb439dc975956fb8eb525c36927255c8f25d5ae9746dc5ba96f6e7` |
| [stock-calibration-interrupted-20260912.zip](https://github.com/displague/dynamic-model-loading/releases/download/v0.14.0/stock-calibration-interrupted-20260912.zip) | 4,658,996 | `5710bcbd6912c83f2f699dd1a3f7fd636786f47363ed0302282d553ab91adc59` |

The startup archive retains the allocation-log and mmap attempts, with zero
generated tokens. The interrupted archive retains the first 39-case non-mmap
calibration, including its 63 requests and terminal unclassified native abort.
The complete-calibration archive contains verified copies of those cases plus two
new startup-only failures. Copied cases are not additional inference. The evaluation
archive contains all 54 groups, 574 requests, original analysis and driver log.
The replay archive contains both the interrupted and complete common-prefix
diagnostics and their driver logs. The eleven repeated baseline requests are an
explicit replication check, not replacements. Archives include compact response
files, native logs, commands, source/artifact hashes and full resource samples.

`analysis.json` is a byte-identical copy of the original frozen-source analyzer
output (SHA-256 `b4244b9c34047e774fcb342701bb45fa05d9da2beb7b45531d66a067686fa3c7`).
It binds the raw evaluation receipts and preserves per-request/per-prompt records,
full accepted-prefix arrays, output differences and first-divergence prefixes.
`summary.json` is a separately derived delivery view, with whole-run resource
audits, paired prompt ratios, native allocation lines, cycle accounting and
diagnostic receipt hashes. Its builder is [build_summary.py](build_summary.py).
No model inference is needed to reproduce either analysis.

## Restore and reproduce without inference

Download all five release ZIPs and verify their archive hashes against
`archives.json`. Extract into a **fresh directory**, preserving each member's exact
relative path. The three final archives contain a `runs/...` hierarchy; the two
earlier startup/interrupted archives retain their original directories at the
extraction root. Verify every member against its bytes/SHA-256 entry; do not
merge into original measurement directories. The reported validation asset records
an actual restore and reanalysis, not merely a ZIP integrity check.

To regenerate the original analysis byte-for-byte, use a clean checkout at
`5b4a902b0ccff26c5313aeb5eef7a5a35a46e49a` and the recorded Python 3.14.3 runtime:

```powershell
python scripts/analyze_stock.py --evaluation C:/restored/runs/stock-evaluation-20260912 --output C:/restored/analysis-reproduced.json
```

The source commit, interpreter string and script hash are part of that output;
changing them changes provenance even when numerical aggregates agree. The script
requires a clean source checkout. Use a new output filename each time.

From a clean v0.14.0 checkout, reproduce the delivery summary against the original
analysis and extracted diagnostics:

```powershell
python results/stock-speculation-20260912/build_summary.py --analysis C:/restored/runs/stock-analysis-20260912.json --evaluation C:/restored/runs/stock-evaluation-20260912 --replay C:/restored/runs/stock-replay-20260912-v2 --original-replay C:/restored/runs/stock-replay-20260912 --output C:/restored/summary-reproduced.json
```

Run Python normally, without `-O`: the delivery builder uses assertions for
receipt and case reconciliation. The two replay driver logs must sit beside their
directories as archived. Existing result files are byte-preserved by the repository's
`.gitattributes`, including on Windows. Full SHA inventories include raw timestamps
and historical machine paths as provenance; those paths are not rewritten.

For fresh native measurements, follow the runner and frozen protocol rather than
these read-only analysis commands. Model/runtime downloads are separate, pinned in
`configs/stock-speculation-artifacts.json`; GGUFs and binaries are not in these
archives. All prior failures and this release's unresolved output identity remain
part of the result. See [the detailed report](../../docs/stock-speculation-results.md)
and [release notes](../../docs/releases/v0.14.0.md).

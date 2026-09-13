# Attention-resident continuing-conversation receipts

The reviewed and pushed source was `5b7dcbd8ace3357b263536f0a38b88badfec1050`.
Eight serial native processes, 48 scored requests and eight excluded warmups;
no failed attempts or replacement runs. `analysis.json` reconstructs raw SSE,
prompt binding, native acceptance/timing, startup allocations and resource traces.
`summary.json` supplies descriptive aggregates; it cannot turn a capped answer
into task completion or a differing-input comparison into identical-input speedup.

The v0.18.0 raw release asset contains all 1,418 inventoried files, including
measured source, native source excerpts and protocol review receipts. All members
were restored with exact hashes and the analyzer reproduced byte-identical output
without loading models. See `archive-inventory.json` and `restoration.json`.

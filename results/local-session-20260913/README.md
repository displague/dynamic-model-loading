# Local integration receipts

`analysis.json` recomputes API/client outcomes from both raw attempts.
`archive-inventory.json` hashes all210 archive members; `restoration.json` records
actual extraction and byte-identical model-free analysis. `environment.json`
records the baseline interpreter/packages and current device/driver.

Raw ZIP and validation assets accompany v0.19.0. The initial client failures and
corrected Claude success remain separate; the aggregate all-client verdict fails
because Codex does not complete the edit. This is not a throughput experiment.

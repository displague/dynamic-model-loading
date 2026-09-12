# Target-scale artifact acquisition

The storage blocker recorded in v0.13 was resolved when the project owner freed
space. Thirteen merged, fully readable worktrees containing no unique untracked
artifacts were removed. Worktrees with untracked fixtures or incomplete scans were
preserved; the audit retains those exclusions. Git history and original measurement
directories remain intact.

All eight GGUF files for the four pinned target/draft representations were acquired
and checked against the published catalogue's SHA-256 and byte counts. The initial
HF CLI attempt made no progress in its destination files; its owned process tree
was stopped. Public HTTP range downloads from the same pinned Hugging Face
revisions completed successfully. `downloads.json` records the acquired files.

`tokenizers.json` was produced using the pinned llama.cpp GGUF metadata reader,
without model inference. It fingerprints all tokenizer fields and compares shared
token IDs. The target/IQ2_XS vocabulary has 152,064 entries; the two small drafts
have 151,936. No shared token ID differs. The stock compatibility check permits
a size difference up to 128; runtime validation is still required.

These are acquisition and metadata receipts, not measurements of acceptance,
inference quality, speed or memory feasibility. Follow the prospective stock
protocol and review/commit the harness before target inference.

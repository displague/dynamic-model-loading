### P2 — Final-tip validation is claimed without an attachment

[docs/releases/v0.23.0.md:67](/C:/Users/displ/Documents/dynamic-model-loading/docs/releases/v0.23.0.md:67) says final-tip validation “is attached separately.” The reviewed receipts contain only `source-freeze-tests.xml`; the archive contains no final-tip validation receipt.

**Fix:** Mark this as pending, or attach and link the intended receipt before publication.

Otherwise, the numbers, failed hypotheses, and limitations reconcile. Verified source `3e0f585`, all 162 raw-file hashes, the archive’s size/SHA256, and all 169 decompressed member hashes.

Disposition: final-tip validation is a publication prerequisite, not a completed
pre-commit claim. Release notes now link separate test and commit-bound validation
assets. Run and verify these on the final clean candidate before publishing.

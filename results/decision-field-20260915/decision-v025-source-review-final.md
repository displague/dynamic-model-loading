## BLOCKED — research fixes verified; two release-helper blockers remain

Both original blockers are resolved: the numerical reference is pinned, shapes/dtypes are checked, and phase accounting now rejects missing or incorrect fields. The added tests and documentation match those fixes.

1. **A filtered test run can qualify as “full” validation.**  
   [validate_release_tip.py:31](/C:/Users/displ/Documents/dynamic-model-loading/scripts/validate_release_tip.py:31) inherits `PYTEST_ADDOPTS`. For example, `PYTEST_ADDOPTS="-k decision_field"` runs only matching tests, yet the positive-count check accepts the receipt.  
   **Fix:** Clear or reject selection-altering pytest environment options and verify the intended full collection before issuing the receipt.

2. **The release tag is not pinned to the validated commit.**  
   [publish_research_release.py:49](/C:/Users/displ/Documents/dynamic-model-loading/scripts/publish_research_release.py:49) tags the current `HEAD`, whereas validation checks the earlier `tip`. If HEAD advances during publication, a different commit can be tagged while the receipt still names the validated commit.  
   **Fix:** Pass `tip` explicitly to `git tag`, verify the peeled tag equals it, and bind the notes to that commit’s file bytes.

One focused static pass completed. No pytest, inference, writes, or publication actions performed.
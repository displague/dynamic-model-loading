# Preserve fitted matrix layout in causal timing replay

This prospective timing-only correction follows the failed initial cost run for
the v0.12 causal evidence experiment. Commit and push it before the corrected timing
run. The measured quality source remains 113d4bf2bef946bb1db4e09c9423b2cccf201723;
the original causal-evidence protocol, configuration, inputs, fits, decisions,
quality rows and thresholds remain unchanged.

The original cost run completed nine fixtures, then failed its exact-mask preflight
on full completion. Safetensors preserves coefficient values but normalizes matrix
layout. The original FP64 solve returned column-major coefficient matrices; the
cost loader reconstructed contiguous matrices. At full-completion input 4, layer 7
(both zero-based),
a 1.1920928955078125e-07 score difference swaps groups 943 and 975. This is a timing
replay failure, not a new model quality result. Preserve its 468 timing rows and
console failure; they do not constitute the complete cost grid.

Reconstruct the original fit with the unchanged fitting implementation and the
retained second-pass calibration examples. Require every coefficient to equal the
frozen fitted tensor bit-for-bit. Transfer those reconstructed tensors to CUDA with
their original layout. Record strides and fit identity. This is not retraining on
development data or choosing new coefficients. An unequal reconstruction must fail.

Keep the exact-mask preflight, every workload, batch size, clock boundary, warmup,
repetition count and cost gate unchanged. Rerun the complete cost grid once in a
fresh directory; do not combine selected timings from the failed and corrected runs.
Save separate committed timing source/configuration/protocol snapshots and identify
the original measured source explicitly. The independent analyzer verifies both
snapshots, unchanged configuration, coefficient equality, layout and complete grids.

The previously reviewed analyzer correction for checkpoint vocabulary width remains
analysis-only. Neither correction authorizes new model scoring or relaxed gates.

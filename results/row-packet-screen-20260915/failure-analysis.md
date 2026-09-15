# Failed v0.33 first attempt

Source a1119214c711ad7439adfebfb16d8c0b00f2ca2c. Worker stopped after
18.0888369 seconds at the numerical check following episode-5 (packet, document1).
Six of ten episodes exist; the complete acquisition/runtime gates were not run.
All four completed hybrid episode ID sequences match their original references.
Document0 stream/sparse/packet each have maximum per-position logit relative L2
1.6675842781353612e-6. Document1 packet reaches 1.3045548089693852e-5 on its eighth
position, exceeding the unchanged 1e-5 tolerance. No restored repeat or remaining
document1 controls ran. No summary, completed resource audit or full validation
is claimed. Raw files, source/protocol snapshots and supervisor remain unchanged.

Diagnostic calculation uses only the saved full logits, not new inference/fitting.
The hybrid operator uses neuron-row contiguous storage and x@W plus separate bias,
unlike the original contiguous F.linear(weight,bias). A separately frozen corrected
run will preserve original weight layout and fused linear operator for all three
conditions. This is not a tolerance change or a new diagnostic subset.

**BLOCKED**

- [retention_screen.py:221](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/retention_screen.py:221): The acquisition gate rejects exactly 10% savings, contrary to the frozen protocol. A CPU reproducer using valid retention transitions produced packet bytes **16,810,000** and retained bytes **15,129,000**; subtraction yields `0.09999999999999998`, making `Hacquisition=False`. Compare integer totals directly: `10 * retained_bytes <= 9 * packet_bytes`.

All 16 CPU tests passed. No other blocking findings.
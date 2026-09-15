**BLOCKED for source freeze.**

1. **Worker failures exit successfully.** [specialist_screen.py:229](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/specialist_screen.py:229) prints supervisor results without propagating failure. Mocked `error` and `timeout` results both produced exit code 0. Return a nonzero status for failed supervision.

2. **Construction transfer accounting omits buffers.** [specialist_screen.py:130](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/specialist_screen.py:130) counts only parameters, but `.to('cuda')` also transfers registered rotary buffers. [specialist_analysis.py:119](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/specialist_analysis.py:119) enforces that undercount. Charge the buffers and update the audit.

3. **NumPy threading is neither frozen nor recorded.** [specialist_screen.py:81](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/specialist_screen.py:81) sets PyTorch threads only. In the baseline `.venv`, NumPy’s BLAS pool remained at **24 threads** after this setting. Freeze and record the BLAS thread count for fitting/scoring and replay; distinguish it from the four-thread PyTorch setting.

The ridge algebra, fit-only isolation, full-vocabulary scoring, repeat selection, and teacher-forced claim boundaries checked out. **17 focused tests passed**, plus tiny CPU checks of readout reconstruction, diagnostic isolation, and supervisor dispatch.

No real checkpoints loaded, experiments scored, files modified, or publication actions taken.
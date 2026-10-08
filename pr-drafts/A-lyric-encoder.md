**Title:** fix(cuda): compute the lyric encoder's last projection in float32 on float16 GPUs

### Summary

- On GPUs without bfloat16 (Volta, Turing, Pascal: V100, T4, RTX 20-series, GTX 10-series, P40) the model runs in
  float16, and **every song with lyrics fails with "NaN or Inf latents", while instrumentals work** (#1055, #1243,
  both closed as stale; see also the closed PR #1274).
- Root cause: one layer. With a hook on every module in a float32 run, the lyric encoder's last MLP projection
  (`lyric_encoder.layers[7].mlp.down_proj`) reaches **~505,000** on ordinary lyrics; float16's maximum is 65,504.
  Everything before it stays under ~11,500. Measured on all 8 shipped `examples/text2music` songs, for the 2B and the
  XL lyric encoder alike. In a float16 run, that projection is the first module whose output goes non-finite.
- Fix: compute only that projection in float32, from its own float16 weights upcast on the fly; the residual add
  promotes to float32, and the final norm's output is cast back to float16. No weights are changed or copied, so CPU
  offload's later `model.to(dtype=...)` can't undo it. It costs ~12 MB of transient memory.

This complements #1312 / #1185 (make `ACESTEP_DTYPE=float32` work): those are the escape hatch, this makes the
default float16 path work. On a V100, float16 is 2x faster than float32 and uses half the memory.

### Scope

- New `acestep/core/generation/handler/fp16_lyric_encoder.py` (83 lines): a small `nn.Linear` subclass that computes
  in float32, `compute_lyric_encoder_tail_in_float32()`, and `apply_float16_lyric_encoder_fix()`, which logs.
- `init_service_loader.py`: +3 lines -- the import and `if self.dtype == torch.float16:
  apply_float16_lyric_encoder_fix(self.model)` after `self.model.eval()`. The file was already 217 lines (over the
  200-line cap) before this change; the logic is kept out of it.
- Tests: `fp16_lyric_encoder_test.py`, `fp16_lyric_encoder_loader_test.py`.
- Out of scope: the DiT decoder overflows in the CFG (sft/base) and XL models (separate PRs #B and #C), and the
  `ACESTEP_DTYPE` override (#1312).

### Risk and Compatibility

- Target path: float16 models only. float16 is chosen automatically on pre-Ampere CUDA, or opted into on ROCm with
  `ACESTEP_ROCM_DTYPE=float16`. **Non-target paths unchanged:** bfloat16 and float32 never call it (tested at the
  loader level); MPS / XPU / CPU / MLX never use float16 here.
- The patched module is still an `nn.Linear`, so state_dict keys, PEFT wrapping, `torch.compile`, deepcopy and
  pickling keep working (tested). Autocast is disabled inside it, so a float16 autocast context (the trainer uses
  one) can't push it back to float16.
- It patches the last layer that `AceStepLyricEncoder.forward` actually runs (`layers[: config.num_hidden_layers]`),
  leaves a replaced or quantized Linear alone, and logs a warning if a float16 model's lyric encoder isn't recognised.
- Accuracy: over the 8 songs, the encoder output's relative error against a fully-float32 encoder is 0.2-7% (2B) and
  0.2-2.4% (XL). Same-seed songs are as similar to the float32 render as two renders on different cards are to each
  other.

### Regression Checks

- 13 new unit tests, using the real `AceStepLyricEncoder` with a tiny config:
  - a control proving it overflows float16 without the fix;
  - with the fix, the output is finite, float16 and close to float32;
  - it survives `.to(dtype=float16)` and float16 autocast;
  - deepcopy and pickle work;
  - a float32 model is unchanged;
  - idempotent;
  - it patches the last layer that runs;
  - a replaced Linear and non-matching models are left alone;
  - it warns when not recognised;
  - the loader applies it, to the loaded model, for float16 only.

  Each one fails under a targeted change to the code it covers. We broke the code nine ways (e.g. deleting the
  autocast guard, widening the dtype gate) and a test failed every time.
- Full `unittest discover -p "*_test.py"` suite (CPU): `main` runs 1,230 tests with 44 pre-existing failures/errors; with this change the same 44 and no new ones, plus the 13 new tests passing (checked with all three PRs applied: 1,270 tests, same 44).
- Real songs, float16, all 8 shipped example songs (5 languages, 2:22-3:56):
  - `acestep-v15-turbo` on a Tesla V100 32 GB: 0 NaN with the fix; stock, every one fails.
  - GTX 1070 8 GB (Pascal), 30-s song: stock gives all-NaN latents; with the fix it takes 15.4 s, also with
    `offload_dit_to_cpu=True`.
- RTX 4070 (bfloat16): unchanged, because the function is never called.

### Reviewer Notes

- Pre-existing, not addressed:
  - `init_service_test.py::test_load_main_model_ignores_cuda_sync_cleanup_error` fails identically on `main`
    (`_Host` lacks `_sync_alignment_config`).
  - The official install's torch 2.10.0+cu128 has no Pascal (sm_61) kernels; Pascal users need the cu126 build.
  - `acestep/training_v2/model_loader.py` loads the model through its own path, so float16 training doesn't get this
    fix.
- Follow-ups: #B (DiT residual stream, CFG and XL models) and #C (XL attention scores). With all three, all six
  models make all 8 songs in float16 on a V100. The three PRs are independent; each adds one import and one call in
  the same spot of the loader, so whichever merges second needs a trivial rebase.
- Prepared with AI assistance (Claude). Per CONTRIBUTING, the commit had two review passes by a separate agent, and
  all findings were applied: a module-level subclass instead of a closure, the autocast guard, loader-level and
  edge-case tests, and its own module.
- Measurements, scripts and raw results: https://github.com/christopherrobertbrooks-tech/volta-ace-step

**Title:** fix(cuda): keep the DiT residual stream inside float16 range (CFG and XL models)

### Summary

- On GPUs without bfloat16 the model runs in float16. Even with the lyric-encoder fix (#A), the CFG models (`sft`,
  `base`) and all three XL models still produce all-NaN latents: the DiT decoder's residual stream overflows.
- Measured in float32 with a hook on every module:
  - 2B sft: the gated MLP product reaches ~137,000 at `decoder.layers.20`.
  - XL, at `decoder.layers.18`: the stream reaches ~531,000 (xl-turbo), ~789,000 (xl-base) and
    **~1.36 million** (xl-sft).

  float16's maximum is 65,504. The branch outputs themselves stay under ~30,000; what overflows is the gated product
  and the running sum.
- Fix: every branch reads the stream through an RMSNorm (which ignores overall scale), and the stream ends in
  `norm_out`. So forward hooks divide what enters the stream by 64: `proj_in`'s output and each layer's
  self-attention, cross-attention and MLP outputs. The stream-reading norms' epsilon is divided by 64^2. The decoder's
  output is bit-identical in float32, and the worst case (xl-sft) now peaks at 21,328, about 3x headroom.

### Scope

- New `acestep/core/generation/handler/fp16_dit_stream.py` (110 lines): `rescale_dit_residual_stream()` and
  `apply_float16_dit_stream_fix()`, which logs.
- `init_service_loader.py`: +3 lines (the import and a float16-only call after `self.model.eval()`; the file was
  already 217 lines).
- Tests: `fp16_dit_stream_test.py` (fake decoder), `fp16_dit_stream_model_test.py` (real tiny `AceStepDiTModel`) and
  `fp16_dit_stream_loader_test.py`.
- Out of scope: the lyric encoder (#A) and XL attention scores (#C).

### Risk and Compatibility

- Target path: float16 models only (automatic on pre-Ampere CUDA; ROCm with `ACESTEP_ROCM_DTYPE=float16`).
  **Non-target paths unchanged:** bfloat16 and float32 never call it (tested at the loader level).
- **No weights are modified, only hooks.** This matters for adapters: ACE-Step's default LoRA targets include
  `o_proj`. With the hooks on the branch modules, a LoRA/LoKr delta inside a branch is scaled together with it, so an
  adapter has the same effect as on a bfloat16 card (tested with real PEFT). Scaling the weights instead would make
  such adapters 64x too strong, and would leak into anything that saves weights.
- **Limits (stated in the module docstring):**
  - Each branch output must fit float16 before it is divided. Measured peak ~30,000, which stock float16 already
    computes the same way, so this is no new risk.
  - The unscaled stream must stay under 64 × 65,504 ≈ 4.2 million (measured 1.36 million).
- Exactness relies on every read of the stream going through `self_attn_norm` / `cross_attn_norm` / `mlp_norm` /
  `norm_out` (traced in all six model files). If any of these is missing or is not an RMSNorm with a
  `variance_epsilon`, nothing is changed and the loader logs a warning. The factor must be a power of two.
- Known edge: `AceStepDiTModel.forward(return_hidden_states=...)` returns the raw stream before `norm_out`, which is
  scaled by 1/64 in float16 mode. There are no in-repo callers.
- Run-time cost: one elementwise division per branch per layer per step. See the timings below.

### Regression Checks

- 15 new unit tests:
  - **Fake decoder** (same residual pattern as `AceStepDiTLayer`):
    - output unchanged, no weights changed, q/k norms untouched;
    - a float16 control that overflows without the fix and is finite with it;
    - a stream small enough for epsilon to matter is unchanged, and its control shows the epsilon scaling is
      required;
    - idempotent (a different factor warns); a shared norm is scaled once;
    - unrecognised decoders and non-power-of-two factors are rejected; it warns when not recognised.
  - **Real tiny XL `AceStepDiTModel`:**
    - float32 output bit-identical;
    - a stream peaking ~237K overflows float16 without the fix, and is finite and accurate with it;
    - a real PEFT LoRA on q/k/v/o_proj has the same effect as without the rescale, and `get_base_model()` keeps the
      hooks.
  - **Loader:** float16 only, called with the loaded model.

  Each one fails under a targeted change to the code it covers: we broke the code 13 ways and a test failed every
  time.
- Full `unittest discover -p "*_test.py"` suite (CPU): `main` runs 1,230 tests with 44 pre-existing failures/errors; with this change the same 44 and no new ones, plus the 15 new tests passing (checked with all three PRs applied: 1,270 tests, same 44).
- Real songs, Tesla V100, float16, with #A and #C: all six models × all 8 shipped example songs = 48/48, 0 NaN (stock: every song with lyrics fails).
- Timings (3:48 song, V100 float16, warm): sft 42.8 s, base 42.8 s, xl-sft 105.0 s, xl-base 104.9 s, turbo 5.7 s -- within 1% of an earlier
  weight-scaling version of this fix (42.5 / 42.5 / 104.4 / 104.4 / 5.7 s), so the hooks cost nothing measurable.
  For reference, the same song on an RTX 4070 in bfloat16: turbo 6.7 s, sft 36.9 s.

### Reviewer Notes

- The three PRs are independent; each adds an import and a call next to `self.model.eval()` (trivial rebase for
  whichever merges second).
- Not addressed: `acestep/training_v2/model_loader.py` loads the model through its own path.
- Prepared with AI assistance (Claude). A separate agent reviewed the commit per CONTRIBUTING and its findings were
  applied: real-model and PEFT tests, the limits documented, its own module, and the nits.
- Measurements and raw results: https://github.com/christopherrobertbrooks-tech/volta-ace-step

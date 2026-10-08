**Title:** fix(cuda): apply the attention score scale to the queries on float16 models

### Summary

- On GPUs without bfloat16, ACE-Step runs float16 with eager attention ("for float16 numerical stability"). The XL
  models (`xl-turbo`, `xl-sft`, `xl-base`) still fail: all 8 shipped example songs give all-NaN latents, and the first
  non-finite output is `decoder.layers.0.self_attn.o_proj`.
- Cause: in the XL checkpoints, `decoder.layers.0`'s `q_norm` and `k_norm` weights reach ~31.6 each, so one q.k over
  128 dims can reach ~128,000. Eager attention computes `matmul(q, k^T)` in float16 and only then multiplies by
  `1/sqrt(head_dim)`. For comparison, the 2B models' largest q_norm x k_norm product is 9.8; the XL's is 1,000.7.
- Fix: a forward hook multiplies `q_norm`'s output by the scale, before RoPE (a rotation, which commutes with it),
  and the module's `scaling` is set to 1.0. The same scores, up to float16 rounding, come out of the matmul already
  scaled.

### Scope

- New `acestep/core/generation/handler/fp16_attention_scale.py` (72 lines): `apply_attention_scale_to_queries()` and
  `apply_float16_attention_scale_fix()`, which logs.
- `init_service_loader.py`: +3 lines (the import and a float16-only call after `self.model.eval()`; the file was
  already 217 lines).
- Tests: `fp16_attention_scale_test.py` (single modules), `fp16_attention_scale_model_test.py` (a whole tiny XL model)
  and `fp16_attention_scale_loader_test.py`.
- Out of scope: the lyric encoder (#A) and the DiT residual stream (#B).

### Risk and Compatibility

- Target path: float16 models only. **Non-target paths unchanged:** bfloat16 and float32 never call it (tested).
- It applies to every `AceStepAttention` in a float16 model (decoder and encoders, all sizes), with any attention
  backend, since eager, SDPA and flash all receive `scaling=self.scaling`. Modules are matched by class name, because
  each model file defines its own copy of the class. Look-alikes such as nano-vllm's `Qwen3Attention`, which copies
  `scaling` at construction, are not touched.
- No weights are modified (a hook plus one attribute), so state_dict, saving and LoRA (on `q_proj`, which feeds
  `q_norm`) behave as before. The original scale is kept on the attention module, so a second call is a no-op, and a
  replaced `q_norm` gets its scale back.

### Regression Checks

- 12 new unit tests:
  - **Real XL `AceStepAttention`, with transformers' eager attention:**
    - self-attention with RoPE is unchanged;
    - a control with the XL layer-0 norm weight (31.6) overflows float16 without the fix, and is finite and correct
      with it;
    - cross-attention output and the attention weights it returns are unchanged;
    - no weights change; idempotent;
    - a replaced `q_norm` gets the scale back;
    - look-alike classes and LoRA layers are left alone;
    - logging.
  - **Whole tiny XL `AceStepConditionGenerationModel`:**
    - exactly its `AceStepAttention` modules are hooked;
    - `generate_audio` is unchanged with eager and SDPA, with and without the cross-attention cache.
  - **Loader:** float16 only, called with the loaded model.

  Each one fails under a targeted change to the code it covers: we broke the code 10 ways and a test failed every
  time.
- Full `unittest discover -p "*_test.py"` suite (CPU): `main` runs 1,230 tests with 44 pre-existing failures/errors; with this change the same 44 and no new ones, plus the 12 new tests passing (checked with all three PRs applied: 1,270 tests, same 44).
- Real songs, Tesla V100, float16, with #A and #B: xl-turbo, xl-sft and xl-base each make all 8 shipped example songs
  (24/24, 0 NaN; stock: 0/24). Timings (3:48 song, warm): xl-turbo 10.9 s, xl-sft 105.0 s, xl-base 104.9 s
  (peak 13.1-15.3 GB); float32 xl-turbo for comparison: 26.4 s, 23.9 GB.

### Reviewer Notes

- Independent of #A and #B; all three add a call next to `self.model.eval()` (trivial rebase for whichever lands
  second).
- Prepared with AI assistance (Claude). A separate agent reviewed the commit per CONTRIBUTING and its findings were
  applied: whole-model, RoPE and cross-attention tests, class-name matching, and the wording about scope.
- Measurements and raw results: https://github.com/christopherrobertbrooks-tech/volta-ace-step

**Title:** fix(nano-vllm): keep the 5 Hz LM inside float16 range on GPUs without bfloat16

### Summary

- On GPUs without bfloat16 (V100, T4, RTX 20-series, GTX 10-series), the bundled nano-vllm switches the 5 Hz LM to
  float16 and the LM then produces NaN logits: every sampled token is id 0, so the planner returns `!!!!...`, with
  no song plan and no audio codes (simple mode and `thinking=True` both broken; the `pt` backend in bfloat16 works
  but is ~4.5x slower for the codes).
- Root cause (float32 hooks on the real weights): one "massive activation". From decoder layer 2 on, the residual
  stream holds ~2,000,000 at `<|im_start|>`, channel 1793; the SwiGLU product feeding layer 2's `down_proj` is
  ~85,000, and layer 27's MLP output ~68,000. Float16's maximum is 65,504. Plain transformers in float16 fails the
  same way, so this is the model's numbers in float16, not a nano-vllm bug as such.
- Fix (float16 only): `RMSNorm.add_rms_forward` keeps the residual stream in float32 (it already added in float32),
  and every MLP's `up` projection is divided by 128 and its output multiplied back in float32. 128 is a power of
  two, so this is numerically equivalent; the worst float16 tensor left is 16,264 (~4x headroom).

Independent of #1352 (lyric encoder, DiT side); this one only touches `acestep/third_parts/nano-vllm` and `uv.lock`.

### Scope

- New `nanovllm/layers/fp16_range.py` (61 lines): `apply_fp16_mlp_scale()` and `maybe_apply_fp16_range(model,
  dtype)`, called once in `ModelRunner.__init__` right after `load_model`.
- `nanovllm/layers/layernorm.py` (+3/-2): float32 residual for float16 weights; output dtype from the norm weight.
- `nanovllm/models/qwen3.py` (+3): `Qwen3MLP.out_scale` (1.0 unless set).
- `nano-vllm` version 0.2.0 -> 0.2.1 in its `pyproject.toml` and in `uv.lock`: it is a non-editable path
  dependency, and uv only rebuilds it when its `pyproject.toml` changes -- without the bump, users keep the old copy.
- Tests: `nanovllm/layers/fp16_range_test.py` (12).

### Risk and Compatibility

- **bfloat16 is unchanged:** a reviewer checked final hidden states and logits are `torch.equal` before/after on the
  real 5 Hz LM weights (3 prompts). Float32 is unchanged too.
- Float16 only, decided in one place (`maybe_apply_fp16_range`). Tensor parallel: each rank's `[gate; up]` shard is
  scaled in its own up half; the `down_proj` all-reduce adds the scaled partial sums before the multiply-back.
- Memory: the float32 residual and MLP output add roughly 128-256 MiB at the warm-up peak (estimate); a few extra
  element-wise passes, float16 only.
- `add_rms_forward` (torch.compile'd) sees more input-dtype combinations in float16, so a few more recompiles.

### Regression Checks

- Tesla V100-PCIe 32 GB, float16:
  - simple mode (`create_sample`): before, every request ran to its token limit and returned nothing; after, real
    captions, BPM, keys and lyrics in 8-11 s;
  - greedy (top-k 1) vs transformers float32, 64 tokens: identical on 4 of 5 prompts (the 5th parts at a near-tie
    on the BPM);
  - full pipeline, `thinking=True` + turbo: real audio codes, LM step 24 s for a 3:10 song (110-126 s on `pt`).
- A reviewer's teacher-forced check on the real weights: float16 top-token agreement with float32 100% / 80% / 100%
  (bfloat16, for comparison: 98% / 60% / 95%); float16 logits ~10x closer to float32 than bfloat16's.
- Unit tests (12; run from `acestep/third_parts/nano-vllm` with `python -m unittest
  nanovllm.layers.fp16_range_test` -- nano-vllm's subpackages have no `__init__.py`, so discovery doesn't find
  them, as with its existing tests): real `Qwen3MLP` / `RMSNorm`, including a test pinned to the real layer-2
  numbers (fails with a scale of 32 or less), float16 control that overflows without the fix, bfloat16 unchanged,
  power-of-two check, and that the model runner applies it after loading. Each was checked by breaking the code it
  covers (8 of 8 caught). nano-vllm's existing tests: same 3 pre-existing `model_runner_shm_test` errors as `main`.

### Reviewer Notes

- Pre-existing, not addressed: in float32, `add_rms_forward` returns the normalised tensor as the residual
  (`x.float()` aliases `x`), so float32 nano-vllm runs are also wrong; `x.clone()` for float32 too would fix it --
  happy to send separately.
- Prepared with AI assistance (Claude); reviewed by a separate agent per CONTRIBUTING, findings applied (uv.lock,
  tests that catch a removed hook or too small a scale, wording). Measurements and scripts:
  https://github.com/christopherrobertbrooks-tech/volta-ace-step

🤖 Generated with [Claude Code](https://claude.com/claude-code)

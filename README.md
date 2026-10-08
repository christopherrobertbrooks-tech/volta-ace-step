# ACE-Step 1.5 on a Tesla V100 (Volta, sm_70)

**Status: in progress (started 2026-10-07).** Root cause found and fixed locally; more checks before proposing it upstream.

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) makes full songs with vocals from lyrics and a style
description. On cards without BF16 hardware (Volta, Turing, Pascal) people report:

- songs **with lyrics** fail with "NaN or Inf latents" in FP16, while instrumentals work;
- the audio decoder outputs **silence** in FP16;
- the error message suggests `ACESTEP_DTYPE=float32`, a setting the code doesn't read.

Reports: [#1055](https://github.com/ace-step/ACE-Step-1.5/issues/1055) (P40, GTX 1080, Titan Xp),
[#1243](https://github.com/ace-step/ACE-Step-1.5/issues/1243) (T4),
[#1274](https://github.com/ace-step/ACE-Step-1.5/issues/1274) (Pascal / Turing),
[#927](https://github.com/ace-step/ACE-Step-1.5/issues/927) (T4) -- all closed as stale, none fixed.

## First results (2026-10-07)

One fixed request: 30 s, English lyrics, `acestep-v15-turbo`, 8 steps, seed 1234, song model only (no LM), installed
the official way (`uv sync`: torch 2.10.0+cu128 -- which still includes sm_70).

| card | dtype | result | time for 30 s of song |
| :--- | :--- | :--- | ---: |
| RTX 4070 | bfloat16 (automatic) | song, 0 NaN, peak 0.89, RMS 0.168 | 1.8 s |
| Tesla V100 | float16 (automatic) | **fails: all 48,000 latents NaN** ("Generation produced NaN or Inf latents") | -- |
| Tesla V100 | float32 (`ACESTEP_DTYPE=float32`, with fix 1) | song, 0 NaN, peak 0.89, RMS 0.140 | 2.6 s |
| **Tesla V100** | **float16 + lyric encoder in float32 (fix 2)** | **song, 0 NaN, peak 0.89, RMS 0.132** | **1.4 s** |

- The bug is still there in today's code (`ca1e85f`), even with ACE-Step's own pre-Ampere workaround (eager
  attention "for float16 numerical stability").
- **`ACESTEP_DTYPE` does nothing upstream:** it appears once in the code -- inside the error message that tells you to
  set it. AMD cards already have the equivalent (`ACESTEP_ROCM_DTYPE`); NVIDIA cards never got one.
- **Fix 1 (done, local branch `volta/cuda-dtype-override`, 2afe386):** `_resolve_cuda_dtype()` honours
  `ACESTEP_DTYPE=float32|float16|bfloat16`, modelled on the ROCm override, with 5 unit tests (fail before, pass
  after). The one other failure in `init_service_test.py` (`test_load_main_model_ignores_cuda_sync_cleanup_error`)
  fails the same way on unpatched `main`.
- **Root cause (probe on all 1,466 layers, `work/probe.py`):** in an FP32 run exactly one layer exceeds FP16's
  65,504 -- `model.encoder.lyric_encoder.layers.7.mlp.down_proj`, max |x| = **285,393** (4.4x the limit). Next
  largest: `model.decoder.layers.23` at 29,738. In the FP16 run the first non-finite output is that same
  `down_proj`. The lyric encoder only matters when there are lyrics -- hence "lyrics fail, instrumentals work".
- **Fix 2 (local branch, after fix 1):** `_keep_lyric_encoder_in_float32(model)` in `init_service_loader.py` --
  when the model loads in FP16, the lyric encoder alone is kept in FP32 (input upcast, output cast back); it runs
  once per song, so it costs nothing measurable. 4 unit tests, including a control that proves the fake encoder
  overflows without the fix. **Result: the V100 makes the song in FP16 in 1.4 s** (FP32: 2.6 s; RTX 4070 BF16: 1.8 s).
- Similarity (same seed): waveforms of generated music don't line up across precisions (4070 vs V100 FP32 correlate
  at -0.04 too), so the check is by ear plus spectrum similarity: FP16-fix vs FP32 0.85, 4070 vs FP32 0.90.

## Test 1: eight full-length songs (2026-10-07)

ACE-Step's own `examples/text2music/example_NN.json` (caption, lyrics, BPM, key, length as shipped), V100, FP16 with
fix 2, seed 1234, a hook on every layer. Max |value| per part; FP16's limit is 65,504.

| # | language | length | lyrics chars | result | lyric encoder max | decoder max (layer) | time* |
| ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| 01 | zh | 160 s | 460 | song, 0 NaN | 501,389 | 26,016 (layers.23) | 15.9 s |
| 02 | es | 159 s | 1450 | song, 0 NaN | 449,292 | 23,952 (layers.14) | 16.1 s |
| 03 | fr | 142 s | 2729 | song, 0 NaN | 480,436 | 24,000 (layers.23) | 15.4 s |
| 05 | ja | 200 s | 1549 | song, 0 NaN | 444,358 | 26,560 (layers.20) | 20.7 s |
| 10 | en | 228 s | 1184 | song, 0 NaN | 504,648 | 25,408 (layers.23) | 23.5 s |
| 113 | zh | 220 s | 243 | song, 0 NaN | 423,573 | 23,440 (layers.23) | 22.1 s |
| 118 | zh | 178 s | 210 | song, 0 NaN | 415,467 | 28,064 (layers.23) | 17.9 s |
| 108 | zh | 236 s | 133 | song, 0 NaN | 267,289 | 23,264 (layers.14) | 23.8 s |

\*with all 1,466 hooks attached, which slows it down; not a speed figure.

- **All eight make songs.** Without fix 2 every one would fail: the lyric encoder reaches 267,000-505,000 on all of
  them -- even #108, an instrumental whose "lyrics" are only section tags.
- **The decoder never comes near the limit:** 23,264-28,064 across languages, styles and lengths up to 3:56, the same
  range as the 30-s song (29,738). Longer songs do not push it higher.
- The VAE hooks recorded nothing (max 0) -- the decoder-to-audio step isn't covered by this probe yet; the audio
  itself is fine (RMS 0.12-0.18, no NaN).

## The fixes (patches/)

Against `ace-step/ACE-Step-1.5` at `ca1e85f`; apply with `git am patches/*.patch`.

1. `0001` -- honour `ACESTEP_DTYPE=float32|float16|bfloat16` on CUDA (`_resolve_cuda_dtype()`, 5 tests).
2. `0002` -- keep the lyric encoder in float32 when the model runs in float16 (`_keep_lyric_encoder_in_float32()`,
   4 tests). This is the one that makes pre-Ampere cards work at full FP16 speed.

3. `0003` -- a comment on why the wrapper only upcasts `inputs_embeds` (`AceStepLyricEncoder.forward` asserts
   `input_ids is None`; the integer attention mask passes through).

4. `0004` -- scale the DiT decoder's residual stream by 1/8 in float16 (`_rescale_dit_residual_stream()`, 4 tests,
   including one that checks the output is unchanged). Needed for the CFG models (sft/base): see below.

Not yet proposed upstream: tests 2-5 below come first.

### Second overflow: the CFG models (found in test 2)

The `acestep-v15-sft` model (50 steps, CFG 7.0) still failed in FP16 with fix 2: 3 of 3 songs gave all-NaN latents,
first non-finite output in `model.decoder.layers.20`. An FP32 probe (`scripts/probe2.py`) shows the decoder's layer-20
output reaching **136,973** (2.1x FP16's limit) while every branch inside it stays under 16,000 -- the spike is the
MLP output times its timestep gate (`ff_output * c_gate_msa`), and the stream comes back down in layers 21-23
(24K, 19K). Turbo models (no CFG, 8 steps) peak at ~28K, which is why test 1 never hit it.

Every branch reads the stream through an RMSNorm, which ignores scale, and the stream ends in `norm_out`; so fix 3
divides `proj_in` and every `o_proj` / `down_proj` by 8 at load time: exact in FP16, zero run-time cost, same output.
With it, sft makes all three failed songs (layer-20 peak 21-22K).

### Open question for the maintainers: loader-side wrapper or model-side dtype handling?

Fix 2 wraps `lyric_encoder.forward` in the loader. The model-side alternative is Transformers' own hook,
`_keep_in_fp32_modules = ["lyric_encoder"]` on `AceStepPreTrainedModel`, plus an upcast in
`AceStepLyricEncoder.forward` and a cast back in `AceStepConditionEncoder`. Trade-offs:

- **Model-side** is the more idiomatic Transformers pattern, but the lyric encoder is duplicated in **six** model
  files (`base`, `sft`, `turbo`, `xl_base`, `xl_sft`, `xl_turbo`), and the loader's `self.model.to(device).to(self.dtype)`
  right after `from_pretrained` would cast the module back to float16 anyway -- so it needs a loader change too.
- **Loader-side** (this patch) is one place, covers every variant, and runs after that `.to(dtype)`.

We'll offer both in the PR and follow the maintainers' preference.

## Still to check

2. The other model variants (XL 4B turbo / sft / base, 2B sft / base) -- each has its own lyric encoder.
3. The full pipeline with the 5 Hz LM on the `pt` backend (nano-vllm / vLLM don't support sm_70).
4. The audio decoder (VAE) in FP16 -- the "silent output" reports; this probe didn't cover it.
5. A Pascal card (GTX 1070, sm_61), the family of the P40 / GTX 1080 reports.

## Plan

1. Install exactly the official way and record what a V100 owner runs into.
2. Reproduce the failures on the V100; run the same songs on the RTX 4070 (BF16 hardware) as the reference.
3. Find the layer(s) that overflow in FP16, compare against the 4070, fix only those.
4. Prove it with songs (before: NaN / silence, after: music), speed and quality numbers.
5. Propose the fix upstream following ACE-Step's CONTRIBUTING.md (AI-assisted rules).

Also on Volta: the default LM backend uses vLLM (nano-vllm), which does not support sm_70; the `pt` backend
is the fallback.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the machine.

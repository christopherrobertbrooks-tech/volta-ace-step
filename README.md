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

## Plan

1. Install exactly the official way and record what a V100 owner runs into.
2. Reproduce the failures on the V100; run the same songs on the RTX 4070 (BF16 hardware) as the reference.
3. Find the layer(s) that overflow in FP16, compare against the 4070, fix only those.
4. Prove it with songs (before: NaN / silence, after: music), speed and quality numbers.
5. Propose the fix upstream following ACE-Step's CONTRIBUTING.md (AI-assisted rules).

Also on Volta: the default LM backend uses vLLM (nano-vllm), which does not support sm_70; the `pt` backend
is the fallback.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the machine.

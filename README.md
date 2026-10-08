# ACE-Step 1.5 on a Tesla V100 (Volta, sm_70)

**Status: in progress (started 2026-10-07).** First results below; the real FP16 fix is still to do.

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
| Tesla V100 | float32 (`ACESTEP_DTYPE=float32`, with our patch) | song, 0 NaN, peak 0.89, RMS 0.140 | 2.6 s |

- The bug is still there in today's code (`ca1e85f`), even with ACE-Step's own pre-Ampere workaround (eager
  attention "for float16 numerical stability").
- **`ACESTEP_DTYPE` does nothing upstream:** it appears once in the code -- inside the error message that tells you to
  set it. AMD cards already have the equivalent (`ACESTEP_ROCM_DTYPE`); NVIDIA cards never got one.
- **Fix 1 (done, local branch `volta/cuda-dtype-override`, 2afe386):** `_resolve_cuda_dtype()` honours
  `ACESTEP_DTYPE=float32|float16|bfloat16`, modelled on the ROCm override, with 5 unit tests (fail before, pass
  after). The one other failure in `init_service_test.py` (`test_load_main_model_ignores_cuda_sync_cleanup_error`)
  fails the same way on unpatched `main`.
- FP32 works on the V100 but gives up the FP16 tensor cores. Next: find which layer overflows in FP16 and keep only
  that part in FP32.

## Plan

1. Install exactly the official way and record what a V100 owner runs into.
2. Reproduce the failures on the V100; run the same songs on the RTX 4070 (BF16 hardware) as the reference.
3. Find the layer(s) that overflow in FP16, compare against the 4070, fix only those.
4. Prove it with songs (before: NaN / silence, after: music), speed and quality numbers.
5. Propose the fix upstream following ACE-Step's CONTRIBUTING.md (AI-assisted rules).

Also on Volta: the default LM backend uses vLLM (nano-vllm), which does not support sm_70; the `pt` backend
is the fallback.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the machine.

# ACE-Step 1.5 on a Tesla V100 (Volta, sm_70)

**Status: in progress (started 2026-10-07).** Nothing below is a finding yet -- only the questions.

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) makes full songs with vocals from lyrics and a style
description. On cards without BF16 hardware (Volta, Turing, Pascal) people report:

- songs **with lyrics** fail with "NaN or Inf latents" in FP16, while instrumentals work;
- the audio decoder outputs **silence** in FP16;
- the error message suggests `ACESTEP_DTYPE=float32`, a setting the code doesn't read.

Reports: [#1055](https://github.com/ace-step/ACE-Step-1.5/issues/1055) (P40, GTX 1080, Titan Xp),
[#1243](https://github.com/ace-step/ACE-Step-1.5/issues/1243) (T4),
[#1274](https://github.com/ace-step/ACE-Step-1.5/issues/1274) (Pascal / Turing),
[#927](https://github.com/ace-step/ACE-Step-1.5/issues/927) (T4) -- all closed as stale, none fixed.

## Plan

1. Install exactly the official way and record what a V100 owner runs into.
2. Reproduce the failures on the V100; run the same songs on the RTX 4070 (BF16 hardware) as the reference.
3. Find the layer(s) that overflow in FP16, compare against the 4070, fix only those.
4. Prove it with songs (before: NaN / silence, after: music), speed and quality numbers.
5. Propose the fix upstream following ACE-Step's CONTRIBUTING.md (AI-assisted rules).

Also on Volta: the default LM backend uses vLLM (nano-vllm), which does not support sm_70; the `pt` backend
is the fallback.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the machine.

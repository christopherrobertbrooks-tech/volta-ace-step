# ACE-Step 1.5 on a Tesla V100 (and other cards without BF16)

**Status (2026-10-08): every model works in FP16 on the V100 -- 48 of 48 songs -- and on a GTX 1070.** Not yet
proposed upstream.

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) makes full songs with vocals from lyrics and a style
description. On cards without BF16 hardware (Volta, Turing, Pascal) it falls back to FP16, and people report songs
**with lyrics** failing with "NaN or Inf latents" while instrumentals work, silent output, and an error message that
suggests `ACESTEP_DTYPE=float32` -- a setting the code doesn't read. Reports:
[#1055](https://github.com/ace-step/ACE-Step-1.5/issues/1055) (P40, GTX 1080, Titan Xp),
[#1243](https://github.com/ace-step/ACE-Step-1.5/issues/1243) (T4),
[#927](https://github.com/ace-step/ACE-Step-1.5/issues/927) (T4) -- all closed as stale, none fixed (see also the closed,
unmerged PR [#1274](https://github.com/ace-step/ACE-Step-1.5/pull/1274), a float32 fallback).

## Result

Same 8 shipped example songs (`examples/text2music/example_NN.json`: 5 languages, 2:22-3:56 long, as shipped) on
every model, seed 1234, song model only, ACE-Step at `ca1e85f`.

| model | steps / CFG | stock, V100 FP16 | all fixes, V100 FP16 | 3:48 song, V100 FP16 | same, V100 FP32 | same, RTX 4070 BF16 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| turbo (2B) | 8 / none | fails | **8/8 songs** | **5.7 s** | 11.4 s | 6.7 s |
| sft (2B) | 50 / 7.0 | fails | **8/8** | 42.5 s | -- | 36.9 s |
| base (2B) | 50 / 7.0 | fails | **8/8** | 42.5 s | -- | -- |
| xl-turbo (4B) | 8 / none | fails | **8/8** | 10.8 s | 26.4 s | -- |
| xl-sft (4B) | 50 / 7.0 | fails | **8/8** | 104.4 s | -- | -- |
| xl-base (4B) | 50 / 7.0 | fails | **8/8** | 104.4 s | -- | -- |

"Fails" = every song with lyrics comes out NaN: each model's own lyric encoder peaks at 499,000-505,000 in the final
run (`data/final-*.json`, `lyric_max`), and a stock FP16 lyric encoder gives NaN on all 8 songs (`scripts/probe_lyric2.py`,
2B and XL). Stock turbo also failed end to end (test 1), and the CFG and XL models fail in the decoder as well. Timings: one clean run without hooks, the second (warm) of two, example
song 10 (3:48, English), `scripts/timing.py`. 48/48: zero NaN samples, RMS 0.07-0.28.

- **FP16 is what makes the V100 worth it:** 2.0-2.4x faster than FP32 and about half the memory (turbo 7.3 vs 13.0 GB,
  xl-turbo 13.1 vs 23.9 GB). With it the V100 makes a 3:48 song in 5.7 s -- a bit faster than an RTX 4070 in BF16
  (6.7 s); the 4070 is ahead on the 50-step CFG models (36.9 vs 42.5 s).
- The XL models fit easily: 15.3 GB peak for xl-sft/base.

**GTX 1070 (Pascal, 8 GB):** stock FP16 also gives all-NaN latents; with the fixes a 30-s song takes 15.4 s
(CPU offload on, ~6 GB free). See [Pascal](#pascal-gtx-1070-sm_61).

## What overflows, and the fix for each

FP16's largest number is 65,504. Measured in FP32 with a hook on every layer:

| where | FP32 peak | models | fix |
| :--- | ---: | :--- | :--- |
| lyric encoder, `layers[-1].mlp.down_proj` | ~505,000 | all six | compute that one projection in FP32 (`0006`) |
| DiT residual stream (`decoder.layers.N` output) | 175K (sft), 531K (xl-turbo), 789K (xl-base), 1.36M (xl-sft) | all but 2B turbo | scale the stream by 1/64, exactly (`0004` + `0007`) |
| XL self-attention scores, `decoder.layers.0` | up to ~128,000 before scaling | the three XL | fold the 1/sqrt(d) scale into `q_norm` (`0005`) |

1. **Lyric encoder.** Its last MLP output reaches ~505,000 while everything before it stays under ~11,500, and the
   stream goes straight into the final norm after it. Only that projection is computed in FP32, from its own FP16
   weights upcast on the fly (a 3072x1024 matrix); the residual add promotes to FP32 and the final norm's output is
   cast back. Relative error vs a fully-FP32 encoder: 0.2-7% (2B), 0.2-2.4% (XL). This is why "lyrics fail,
   instrumentals work": the lyric encoder only matters when there are lyrics. (Even #108, whose "lyrics" are only
   section tags, reaches 267,000.)
2. **DiT residual stream.** Each layer adds gated branch outputs to the stream; with CFG the gated MLP output alone
   reaches 137,000 (2B sft, layer 20), and the XL stream reaches 1.36 million (xl-sft, layer 18). Every branch reads
   the stream through an RMSNorm, which ignores overall scale, and the stream ends in `norm_out`; so `proj_in` and
   every `o_proj` / `down_proj` are divided by 64 at load time, and those norms' epsilon by 64^2. That is exact --
   same output, zero run-time cost -- and leaves ~3x headroom on the worst case (xl-sft: 21,328 after scaling).
3. **XL attention.** The XL checkpoints' layer-0 `q_norm` and `k_norm` weights reach ~31.6 each (the 2B models'
   worst product is 9.8, the XL's 1,000.7), so one q.k over 128 dims can reach ~128,000. Eager attention -- which
   ACE-Step forces on pre-Ampere cards "for float16 numerical stability" -- computes `matmul(q, k^T)` in FP16 and
   only then multiplies by `1/sqrt(128)`. Multiplying `q_norm.weight` by that scale and setting `scaling = 1.0`
   gives the same scores, already scaled, straight out of the matmul (RoPE is a rotation, so it commutes).

## The patches (`patches/`)

Against `ace-step/ACE-Step-1.5` at `ca1e85f`; apply with `git am patches/*.patch`. Each has unit tests that fail
before and pass after, including a control proving the overflow is real (`attention_scale_fold_test.py`,
`dit_stream_scale_test.py`, `lyric_encoder_precision_test.py`, `CudaDtypeTests`); the one other failure in
`init_service_test.py` (`test_load_main_model_ignores_cuda_sync_cleanup_error`) fails the same way on unpatched `main`.

| patch | what | status |
| :--- | :--- | :--- |
| `0001` | honour `ACESTEP_DTYPE=float32\|float16\|bfloat16` on CUDA (`_resolve_cuda_dtype()`, modelled on `ACESTEP_ROCM_DTYPE`) | keep |
| `0002`, `0003` | the whole lyric encoder in FP32 (+ a comment) | **superseded by `0006`**: +0.9 GB VRAM (OOM on the 8 GB GTX 1070), and CPU offload's `model.to(dtype=float16)` undid it ("mat1 and mat2 must have the same dtype") |
| `0004` | DiT residual stream / 8 | factor and epsilon updated by `0007` |
| `0005` | XL attention: scale folded into `q_norm` | keep |
| `0006` | only the lyric encoder's last projection in FP32 | keep |
| `0007` | stream / 64, norms' epsilon / 64^2 (/8 left xl-turbo within 1% of the limit and xl-sft overflowing) | keep |

For the upstream PR these collapse to four changes: `0001`, `0006`, `0004`+`0007`, `0005`.

### Open question for the maintainers: loader-side or model-side?

All three numeric fixes run in the loader (`init_service_loader.py`) after `from_pretrained` and the
`.to(device).to(self.dtype)`, so they cover all six model variants from one place. The model-side alternative --
doing the same inside `AceStepLyricEncoder`, `AceStepAttention` and the DiT -- is more conventional, but those
classes are duplicated in six model files (`base`, `sft`, `turbo`, `xl_base`, `xl_sft`, `xl_turbo`) and the
checkpoints carry their own copy of the modeling code. We'll offer both and follow the maintainers' preference.

## Other findings

- **The 5 Hz LM (planner): the `pt` backend works on the V100; the default `vllm` backend does not.** Bundled
  nano-vllm detects Volta, switches itself to FP16 and runs eager -- but its output is garbage: the "Debug output
  text" is `!!!!!!!!...` (the classic sign of non-finite logits), for the song plan and for the audio codes alike.
  The `pt` backend writes real plans and real audio codes (3/3 songs; LM 110-126 s for a 3:48 song). *Correction
  (2026-10-08): an earlier version of this page said nano-vllm works on sm_70 because the songs came out; they did
  only because the caption and lyrics were supplied, so the planner's garbage wasn't needed.* Likely another FP16
  overflow, in the LM this time -- next thing to look at.
- **The audio decoder (VAE) is fine in FP16:** no NaN, no silence, RMS 0.150 vs 0.147 forced-FP32 on the same song.
  The "silent output" reports are most likely the NaN latents above, decoded.
- **`ACESTEP_DTYPE` does nothing upstream:** it appears once in the code -- inside the error message telling you to
  set it.
- **Turing** (RTX 20-series, T4) has no BF16 either and takes the same FP16 path; not tested here.

### Pascal (GTX 1070, sm_61)

- The official install (`uv sync`, torch 2.10.0+cu128) has no Pascal kernels: "CUDA error: no kernel image is
  available for execution on the device". The same version from the cu126 index has them (sm_50 ... sm_90); see
  [ENVIRONMENT.md](ENVIRONMENT.md) for the one-line reinstall (and why the obvious one silently does nothing).
- With cu126 torch, stock code: all-NaN latents, as on the V100.
- With the fixes (`0001`-`0006`), 30-s song: **15.4 s** (CPU offload, default), 20.0 s (DiT offloaded too). Both
  settings failed with `0002` (OOM, then the dtype error above). Same-seed similarity to the V100 and 4070 renders
  (log-mel correlation 0.58-0.74) is in the same range as two 1070 renders with different offload settings (0.58).

## Measurement notes

Two of tonight's surprises were the instrument, not the model:

- The per-layer hooks found each output's max with `a[torch.isfinite(a)]`, which copies the tensor -- including eager
  attention's weights (batch x heads x L x L). While those were NaN the copy was tiny; once fix `0005` made them
  finite, a 4-minute CFG song asked for 15.5 GB and ran the V100 out of memory. Hooks now skip 4-D tensors.
- `uv pip install torch==2.10.0 --index-url .../cu126` over an installed 2.10.0+cu128 reports "Checked 3 packages"
  and changes nothing.

## Layout

- `patches/` -- the fixes, as above.
- `scripts/` -- `repro.py` (one 30-s song), `probe.py` / `probe2.py` (hook every layer), `probe_batch.py` (the 8
  songs), `probe_lyric2.py` (lyric encoder alone, FP32 vs FP16 variants), `test3.py` (LM backends), `test4.py` (VAE),
  `timing.py` (clean speed), `test5*.sh` (GTX 1070).
- `data/` -- per-song results.
- [ENVIRONMENT.md](ENVIRONMENT.md) -- the two machines.

## Upstream PRs (drafted, not posted)

The fixes are prepared as **three independent PRs** against `main` (`ca1e85f`), following ACE-Step's CONTRIBUTING.md
and AGENTS.md (one problem per PR, each fix in its own small module, the loader gets 3 lines, tests that fail without
the change -- each test was also checked by deliberately breaking the code it covers):

| PR | change | patch | description draft |
| :--- | :--- | :--- | :--- |
| A | lyric encoder: last projection in FP32 | `patches/upstream-prs/A-lyric-encoder.patch` | `pr-drafts/A-lyric-encoder.md` |
| B | DiT residual stream / 64 via hooks | `patches/upstream-prs/B-dit-stream.patch` | `pr-drafts/B-dit-stream.md` |
| C | attention score scale on the queries | `patches/upstream-prs/C-xl-attention.patch` | `pr-drafts/C-xl-attention.md` |

Differences from `patches/0001-0007` (the working versions): no weights are modified any more -- B and C use forward
hooks, so LoRA adapters on `o_proj` keep their normal strength (the weight-scaling version made them 64x too strong
in float16) -- and the `ACESTEP_DTYPE` override (`0001`) is left to the existing PRs #1312 / #1185. Each PR's commit
was reviewed by a separate agent (two passes for A) and the findings applied. Full `unittest discover` suite: `main`
1,230 tests with 44 pre-existing failures; with all three PRs 1,270 tests, the same 44.

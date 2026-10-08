#!/usr/bin/env python3
"""test4.py: the audio decoder (VAE) in FP16 on the V100 -- the "silent output" reports. Wraps vae.decode to record the
dtype, range and NaN/silence of what goes in and out, for the default (FP16) VAE and a forced-FP32 VAE, same song.
(volta-ace-step, 2026-10-07)"""
import os, sys, json
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, soundfile as sf
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
dit = AceStepHandler()
print("DIT", dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="auto", offload_to_cpu=False)[1], dit.dtype, flush=True)
print("VAE dtype as loaded:", next(dit.vae.parameters()).dtype, flush=True)
seen = []
orig = dit.vae.decode
def decode(z, *a, **k):
    out = orig(z, *a, **k)
    s = out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, (tuple, list)) else out)
    zf, sf_ = z.detach().float(), s.detach().float()
    seen.append({"in_dtype": str(z.dtype), "in_absmax": round(zf.abs().max().item(), 3), "out_dtype": str(s.dtype),
                 "out_nan": int(torch.isnan(sf_).sum()), "out_absmax": round(float(torch.nan_to_num(sf_).abs().max()), 4),
                 "out_rms": round(float(torch.nan_to_num(sf_).pow(2).mean().sqrt()), 5)})
    return out
dit.vae.decode = decode
ex = json.load(open("examples/text2music/example_10.json"))
p = GenerationParams(task_type="text2music", thinking=False, caption=ex["caption"], lyrics=ex["lyrics"], bpm=ex.get("bpm"),
    keyscale=ex.get("keyscale", ""), timesignature=ex.get("timesignature", ""), vocal_language="en", duration=ex["duration"],
    inference_steps=8, guidance_scale=1.0, seed=1234)
for label in ("vae_fp16_default", "vae_fp32_forced"):
    if label == "vae_fp32_forced":
        dit.vae.float(); dit._get_vae_dtype = lambda device=None: torch.float32
    seen.clear()
    r = generate_music(dit, LLMHandler(), params=p, config=GenerationConfig(batch_size=1, audio_format="wav"), save_dir=f"/mnt/steam/ace-step/work/out-t4-{label}")
    row = {"run": label, "ok": r.success, "vae_param_dtype": str(next(dit.vae.parameters()).dtype), "decode_calls": len(seen), "decode": seen[:3]}
    for a in (r.audios or []):
        if a.get("path") and os.path.exists(a["path"]):
            x, _ = sf.read(a["path"], dtype="float32"); row.update(file_nan=int(np.isnan(x).sum()), file_rms=round(float(np.sqrt(np.mean(x ** 2))), 4), file_peak=round(float(np.abs(x).max()), 4))
    print("ROW", json.dumps(row), flush=True)

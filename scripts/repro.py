#!/usr/bin/env python3
"""repro.py <tag>: one fixed 30-s song with English lyrics, song model only (no LM), on whatever GPU is visible.
Reports: finished?, dtype used, NaN/Inf in the audio, peak and RMS (silence check), time. (volta-ace-step, 2026-10-07)"""
import os, sys, time, json
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, soundfile as sf
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
tag = sys.argv[1]; out = f"/mnt/steam/ace-step/work/out-{tag}"; os.makedirs(out, exist_ok=True)
gpu = torch.cuda.get_device_name(0); cc = torch.cuda.get_device_capability(0)
t0 = time.time()
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="auto", offload_to_cpu=False)
print("INIT", ok, msg[:200].replace("\n", " "), "| dtype:", getattr(dit, "dtype", None), flush=True)
if not ok: sys.exit(1)
params = GenerationParams(task_type="text2music", thinking=False,
    caption="Upbeat indie pop with bright electric guitar, warm bass, steady drums and a clear female vocal.",
    lyrics="[Verse]\nWalking down the morning street\nSunlight dancing at my feet\n[Chorus]\nSing it loud, sing it clear\nEvery little thing is here",
    bpm=110, keyscale="C major", timesignature="4", vocal_language="en", duration=30, inference_steps=8, guidance_scale=1.0, seed=1234)
t1 = time.time()
res = generate_music(dit, LLMHandler(), params=params, config=GenerationConfig(batch_size=1, audio_format="wav"), save_dir=out)
el = time.time() - t1
rep = {"tag": tag, "gpu": gpu, "cc": cc, "dtype": str(getattr(dit, "dtype", None)), "success": res.success,
       "status": (res.status_message or "")[:300], "gen_s": round(el, 1), "init_s": round(t1 - t0, 1)}
for a in (res.audios or []):
    p = a.get("path")
    if p and os.path.exists(p):
        x, sr = sf.read(p, dtype="float32")
        rep.update(file=p, sr=sr, seconds=round(len(x) / sr, 1), nan=int(np.isnan(x).sum()), inf=int(np.isinf(x).sum()),
                   peak=round(float(np.nanmax(np.abs(x))), 4), rms=round(float(np.sqrt(np.nanmean(x ** 2))), 5))
print("RESULT", json.dumps(rep), flush=True)

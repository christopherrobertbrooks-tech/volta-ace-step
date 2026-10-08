#!/usr/bin/env python3
"""timing.py <config> <steps> <cfg|-> <tag>: clean speed figure -- no hooks; example song 10 (3:48, English), seed 1234,
song model only; generates it twice and reports the second (warm) run. Dtype from ACESTEP_DTYPE / the default.
(volta-ace-step, 2026-10-08)"""
import os, sys, json, time
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, soundfile as sf
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
cfgname, steps, cfg, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
dit = AceStepHandler(); t0 = time.time()
ok = dit.initialize_service(project_root=ROOT, config_path=cfgname, device="auto", offload_to_cpu=False)[1]
init_s = round(time.time() - t0, 1)
ex = json.load(open("examples/text2music/example_10.json"))
p = GenerationParams(task_type="text2music", thinking=False, caption=ex["caption"], lyrics=ex["lyrics"], bpm=ex.get("bpm"),
    keyscale=ex.get("keyscale", ""), timesignature=ex.get("timesignature", ""), vocal_language="en", duration=ex["duration"],
    inference_steps=steps, seed=1234, **({} if cfg == "-" else {"guidance_scale": float(cfg)}))
times = []
for run in range(2):
    torch.cuda.synchronize(); t1 = time.time()
    r = generate_music(dit, LLMHandler(), params=p, config=GenerationConfig(batch_size=1, audio_format="wav"), save_dir=f"/mnt/steam/ace-step/work/out-timing-{tag}")
    torch.cuda.synchronize(); times.append(round(time.time() - t1, 1))
row = {"tag": tag, "config": cfgname, "dtype": str(dit.dtype), "gpu": torch.cuda.get_device_name(0), "ok": r.success, "init_s": init_s,
       "first_s": times[0], "warm_s": times[1], "song_s": ex["duration"], "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
for a in (r.audios or []):
    if a.get("path") and os.path.exists(a["path"]):
        x, _ = sf.read(a["path"], dtype="float32"); row.update(nan=int(np.isnan(x).sum()), rms=round(float(np.sqrt(np.mean(x ** 2))), 4))
if not r.success: row["err"] = (r.status_message or "")[:160]
print("TIMING", json.dumps(row), flush=True)

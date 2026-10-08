#!/usr/bin/env python3
"""test3.py: the full pipeline -- 5 Hz LM (planner, 1.7B) + song model -- on the V100 in FP16 with the fixes.
Part A: backend "vllm" (the CUDA default) -- does it fail, and how? Part B: backend "pt" -- 3 songs with thinking on.
(volta-ace-step, 2026-10-07)"""
import os, sys, json, time, traceback, signal
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, soundfile as sf
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
part = sys.argv[1]
dit = AceStepHandler()
print("DIT", dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="auto", offload_to_cpu=False)[1], dit.dtype, flush=True)
lm = LLMHandler(); t0 = time.time()
try:
    msg, ok = lm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B",
                            backend=part, device="auto", offload_to_cpu=False, dtype=None)
    print("LM_INIT", part, ok, round(time.time() - t0, 1), "s |", (msg or "")[:300].replace("\n", " "), flush=True)
except Exception as e:
    print("LM_INIT_EXCEPTION", part, type(e).__name__, str(e)[:300].replace("\n", " "), flush=True); traceback.print_exc(); sys.exit(0)
if not ok: sys.exit(0)
for i in (["10", "113", "05"] if part == "pt" else ["10"]):
    ex = json.load(open(f"examples/text2music/example_{i}.json"))
    p = GenerationParams(task_type="text2music", thinking=True, caption=ex.get("caption", ""), lyrics=ex.get("lyrics", ""),
        bpm=ex.get("bpm"), keyscale=ex.get("keyscale", ""), timesignature=ex.get("timesignature", ""),
        vocal_language=ex.get("language", "en"), duration=ex.get("duration"), inference_steps=8, guidance_scale=1.0, seed=1234)
    t1 = time.time()
    try:
        r = generate_music(dit, lm, params=p, config=GenerationConfig(batch_size=1, audio_format="wav"), save_dir=f"/mnt/steam/ace-step/work/out-t3-{part}-{i}")
        row = {"ex": i, "backend": part, "ok": r.success, "total_s": round(time.time() - t1, 1), "status": (r.status_message or "")[:160]}
        for a in (r.audios or []):
            if a.get("path") and os.path.exists(a["path"]):
                x, sr = sf.read(a["path"], dtype="float32"); row.update(seconds=round(len(x) / sr, 1), nan=int(np.isnan(x).sum()), rms=round(float(np.sqrt(np.mean(x ** 2))), 4))
    except Exception as e:
        row = {"ex": i, "backend": part, "ok": False, "exception": f"{type(e).__name__}: {str(e)[:200]}"}
    print("ROW", json.dumps(row, ensure_ascii=False), flush=True)

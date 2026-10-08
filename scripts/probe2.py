#!/usr/bin/env python3
"""probe.py <tag>: the repro song with a hook on every layer of the song model, text encoder and VAE.
Records each layer's largest |output| and whether it produced NaN/Inf; prints the layers past fp16's max (65504)
and, for an fp16 run, the FIRST layer (in call order) whose output went non-finite. (volta-ace-step, 2026-10-07)"""
import os, sys, json, math
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
tag = sys.argv[1]; FP16_MAX = 65504.0
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=ROOT, config_path=os.environ.get("ACE_CONFIG", "acestep-v15-turbo"), device="auto", offload_to_cpu=False)
print("INIT", ok, "dtype", dit.dtype, flush=True)
stats, order, first_bad = {}, [], []
def tensors(o):
    if torch.is_tensor(o): yield o
    elif isinstance(o, (list, tuple)):
        for x in o: yield from tensors(x)
    elif isinstance(o, dict):
        for x in o.values(): yield from tensors(x)
    elif hasattr(o, "__dict__") and type(o).__name__.endswith("Output"):
        for x in vars(o).values(): yield from tensors(x)
def hook(name):
    def f(mod, inp, out):
        mx, bad = 0.0, False
        for t in tensors(out):
            if not t.is_floating_point() or t.numel() == 0: continue
            a = t.detach()
            fin = torch.isfinite(a)
            if not bool(fin.all()): bad = True
            v = a[fin].abs().max().item() if bool(fin.any()) else 0.0
            mx = max(mx, v)
        s = stats.setdefault(name, {"max": 0.0, "bad": False, "calls": 0, "type": type(mod).__name__})
        s["max"] = max(s["max"], mx); s["calls"] += 1
        if bad and not s["bad"]:
            s["bad"] = True
            if not first_bad: first_bad.append((len(order), name, type(mod).__name__))
        order.append(name)
    return f
n = 0
for part in ("model", "text_encoder", "vae"):
    m = getattr(dit, part, None)
    if isinstance(m, torch.nn.Module):
        for name, mod in m.named_modules():
            mod.register_forward_hook(hook(f"{part}.{name}" if name else part)); n += 1
print("HOOKS", n, flush=True)
params = GenerationParams(task_type="text2music", thinking=False,
    caption="Upbeat indie pop with bright electric guitar, warm bass, steady drums and a clear female vocal.",
    lyrics="[Verse]\nWalking down the morning street\nSunlight dancing at my feet\n[Chorus]\nSing it loud, sing it clear\nEvery little thing is here",
    bpm=110, keyscale="C major", timesignature="4", vocal_language="en", duration=30, inference_steps=int(os.environ.get("ACE_STEPS", 8)), seed=1234, **({"guidance_scale": float(os.environ["ACE_CFG"])} if os.environ.get("ACE_CFG") else {}))
res = generate_music(dit, LLMHandler(), params=params, config=GenerationConfig(batch_size=1, audio_format="wav"),
                     save_dir=f"/mnt/steam/ace-step/work/out-{tag}")
print("GEN", res.success, (res.status_message or "")[:120].replace("\n", " "), flush=True)
over = sorted(((k, v) for k, v in stats.items() if v["max"] > FP16_MAX / 4), key=lambda kv: -kv[1]["max"])
print("FIRST_NONFINITE", json.dumps(first_bad[0] if first_bad else None))
print(f"LAYERS_OVER_16K ({len(over)}; fp16 max 65504):")
for k, v in over[:40]: print(f"  {v['max']:>14.1f}  {'OVER' if v['max'] > FP16_MAX else '    '}  {v['type']:<28} {k}")
top = sorted(stats.items(), key=lambda kv: -kv[1]["max"])[:15]
print("TOP15:"); [print(f"  {v['max']:>14.1f}  {v['type']:<28} {k}") for k, v in top]
json.dump(stats, open(f"/mnt/steam/ace-step/work/probe-{tag}.json", "w"))

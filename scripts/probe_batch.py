#!/usr/bin/env python3
"""probe_batch.py <tag> <example ids...>: shipped example songs (full length, their lyrics/caption/bpm/key), song model
only, with a hook on every layer. Per song: success, NaN check, generation time, and the largest |value| in the lyric
encoder, the decoder and the VAE (fp16 max = 65504). (volta-ace-step, 2026-10-07)"""
import os, sys, json, time
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, soundfile as sf
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
tag, ids = sys.argv[1], sys.argv[2:]
dit = AceStepHandler()
ok = dit.initialize_service(project_root=ROOT, config_path=os.environ.get("ACE_CONFIG", "acestep-v15-turbo"), device="auto", offload_to_cpu=False)[1]
print("INIT", ok, dit.dtype, os.environ.get("ACE_CONFIG", "acestep-v15-turbo"), flush=True)
cur = {}
def tensors(o):
    if torch.is_tensor(o): yield o
    elif isinstance(o, (list, tuple)):
        for x in o: yield from tensors(x)
    elif isinstance(o, dict):
        for x in o.values(): yield from tensors(x)
    elif hasattr(o, "__dict__") and type(o).__name__.endswith("Output"):
        for x in vars(o).values(): yield from tensors(x)
def hook(name):
    group = "lyric" if ".lyric_encoder" in name else "decoder" if ".decoder" in name else "vae" if name.startswith("vae") else "other"
    def f(mod, inp, out):
        for t in tensors(out):
            # Skip 4-D tensors: eager attention also returns its weights (batch x heads x L x L, softmax <= 1), and
            # a[isfinite(a)] on those copied ~15 GB for a 4-minute CFG song and ran the V100 out of memory.
            if t.is_floating_point() and t.numel() and t.dim() != 4:
                a = t.detach(); fin = torch.isfinite(a)
                v = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
                if v > cur.get(group, (0, ""))[0]: cur[group] = (v, name)
                if not bool(fin.all()) and "first_bad" not in cur: cur["first_bad"] = name
    return f
for part in ("model", "text_encoder", "vae"):
    m = getattr(dit, part, None)
    if isinstance(m, torch.nn.Module):
        for name, mod in m.named_modules(): mod.register_forward_hook(hook(f"{part}.{name}" if name else part))
rows = []
for i in ids:
    ex = json.load(open(f"examples/text2music/example_{i}.json"))
    cur.clear()
    p = GenerationParams(task_type="text2music", thinking=False, caption=ex.get("caption", ""), lyrics=ex.get("lyrics", ""),
        bpm=ex.get("bpm"), keyscale=ex.get("keyscale", ""), timesignature=ex.get("timesignature", ""),
        vocal_language=ex.get("language", "en"), duration=ex.get("duration"), inference_steps=int(os.environ.get("ACE_STEPS", 8)), seed=1234, **({"guidance_scale": float(os.environ["ACE_CFG"])} if os.environ.get("ACE_CFG") else {}))
    t0 = time.time()
    r = generate_music(dit, LLMHandler(), params=p, config=GenerationConfig(batch_size=1, audio_format="wav"), save_dir=f"/mnt/steam/ace-step/work/out-{tag}-{i}")
    el = time.time() - t0
    row = {"ex": i, "lang": ex.get("language"), "dur": ex.get("duration"), "lyr_chars": len(ex.get("lyrics", "")), "ok": r.success, "gen_s": round(el, 1),
           "lyric_max": round(cur.get("lyric", (0,))[0]), "decoder_max": round(cur.get("decoder", (0,))[0]), "decoder_where": cur.get("decoder", (0, ""))[1].replace("model.decoder.", ""),
           "vae_max": round(cur.get("vae", (0,))[0]), "first_bad": cur.get("first_bad")}
    for a in (r.audios or []):
        if a.get("path") and os.path.exists(a["path"]):
            x, sr = sf.read(a["path"], dtype="float32"); row.update(nan=int(np.isnan(x).sum()), rms=round(float(np.sqrt(np.mean(x ** 2))), 4))
    if not r.success: row["err"] = (r.status_message or "")[:120]
    rows.append(row); print("ROW", json.dumps(row, ensure_ascii=False), flush=True)
json.dump(rows, open(f"/mnt/steam/ace-step/work/batch-{tag}.json", "w"), ensure_ascii=False, indent=1)

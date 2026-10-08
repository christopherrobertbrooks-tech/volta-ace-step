#!/usr/bin/env python3
"""probe_lyric2.py: the lyric encoder alone (it only needs the text encoder's embedding table), on the 8 example songs.
float32 reference vs stock float16 vs float16 with the residual stream divided by F (embed_tokens, v_proj and up_proj
divided by F: every branch reads the stream through RMSNorm and the stream ends in a norm, so the output is unchanged).
Records the float32 maxima of every intermediate, including down_proj's input. (volta-ace-step, 2026-10-08)"""
import os, sys, json, copy
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from transformers import AutoModel, AutoTokenizer
dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "acestep-v15-turbo"
model = AutoModel.from_pretrained(f"checkpoints/{CKPT}", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
enc32 = model.encoder.lyric_encoder.float().to(dev).eval(); del model
te = AutoModel.from_pretrained("checkpoints/Qwen3-Embedding-0.6B", dtype=torch.bfloat16)
emb = te.embed_tokens.float().to(dev); tok = AutoTokenizer.from_pretrained("checkpoints/Qwen3-Embedding-0.6B"); del te
stats = {}
def rec(k, t): stats[k] = max(stats.get(k, 0.0), t.detach().float().abs().max().item())
hooks = [enc32.embed_tokens.register_forward_hook(lambda m, i, o: rec("embed_tokens.out", o))]
for li, L in enumerate(enc32.layers):
    hooks += [L.register_forward_hook(lambda m, i, o, li=li: rec(f"L{li}.stream", o[0] if isinstance(o, tuple) else o)),
              L.self_attn.o_proj.register_forward_hook(lambda m, i, o, li=li: rec(f"L{li}.o_proj.out", o)),
              L.self_attn.o_proj.register_forward_pre_hook(lambda m, i, li=li: rec(f"L{li}.o_proj.in", i[0])),
              L.mlp.down_proj.register_forward_pre_hook(lambda m, i, li=li: rec(f"L{li}.down_proj.in", i[0])),
              L.mlp.down_proj.register_forward_hook(lambda m, i, o, li=li: rec(f"L{li}.down_proj.out", o))]
def rescaled(F):
    e = copy.deepcopy(enc32)
    for h in list(e._forward_hooks): pass
    with torch.no_grad():
        mods = [e.embed_tokens] + [L.self_attn.v_proj for L in e.layers] + [L.mlp.up_proj for L in e.layers]
        for m in mods:
            m.weight.div_(F)
            if m.bias is not None: m.bias.div_(F)
    return e
def strip(m):
    for x in m.modules(): x._forward_hooks.clear(); x._forward_pre_hooks.clear()
    return m
variants = {"fp16 stock": strip(copy.deepcopy(enc32)).half()}
for F in (8,): variants[f"fp16 /{F}"] = strip(rescaled(F)).half()
import torch.nn.functional as TF
def tail32(e):
    down, norm = e.layers[-1].mlp.down_proj, e.norm
    down.forward = lambda x: TF.linear(x.float(), down.weight.float(), None if down.bias is None else down.bias.float())
    norm.register_forward_hook(lambda m, i, out: out.to(m.weight.dtype))
    return e
variants["fp16 tail32"] = tail32(strip(copy.deepcopy(enc32)).half())
variants["fp32 copy"] = strip(copy.deepcopy(enc32))
res = {k: [] for k in variants}
for i in ["01", "02", "03", "05", "10", "113", "118", "108"]:
    ex = json.load(open(f"examples/text2music/example_{i}.json"))
    text = f"# Languages\n{ex.get('language', 'en')}\n\n# Lyric\n{ex.get('lyrics', '')}<|endoftext|>"
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(dev)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        x = emb(ids)
        ref = enc32(inputs_embeds=x, attention_mask=mask).last_hidden_state
        for k, e in variants.items():
            out = e(inputs_embeds=x if k == "fp32 copy" else x.half(), attention_mask=mask).last_hidden_state.float()
            fin = bool(torch.isfinite(out).all())
            err = ((out - ref).norm() / ref.norm()).item() if fin else float("nan")
            res[k].append((i, ids.shape[1], fin, round(err, 5)))
print("FLOAT32 MAXIMA (top 16):")
for k in sorted(stats, key=lambda k: -stats[k])[:16]: print(f"  {stats[k]:>12.1f}  {k}")
print("VARIANTS (song, tokens, finite, relative error vs float32):")
for k, v in res.items(): print(f"  {k:<11}", " ".join(f"{s}:{'ok' if f else 'NaN'}/{e}" for s, n, f, e in v))
json.dump({"maxima": stats, "variants": res}, open(f"/mnt/steam/ace-step/work/probe-lyric2-{CKPT}.json", "w"), indent=1)

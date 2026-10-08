#!/usr/bin/env bash
# Test 5 (overnight 2026-10-07/08): ACE-Step 1.5 on the GTX 1070 (Pascal, sm_61, 8 GB, also drives the display).
# Waits until the gateway's model downloads are done (shared internet), then: official install -> Pascal check ->
# stock FP16 song -> our patches -> FP16 song, timing. Low-VRAM: CPU offload on.
set -u; D=/mnt/data/ace-step; L=$D/test5.log; cd $D
say(){ echo "$(date +%T) $*" | tee -a $L; }
until ssh -o BatchMode=yes ember-gateway 'grep -q "DOWNLOADS DONE" /mnt/steam/ace-step/work/download-xl.log'; do sleep 60; done
say "gateway downloads done; starting"
[ -x uvtool/bin/uv ] || { python3 -m venv uvtool && uvtool/bin/pip -q install uv; }
[ -d ACE-Step-1.5 ] || git clone -q https://github.com/ace-step/ACE-Step-1.5 && cd ACE-Step-1.5 && git checkout -q ca1e85f
export UV_CACHE_DIR=$D/uv-cache UV_PYTHON_INSTALL_DIR=$D/uv-python HF_HOME=$D/hf
say "uv sync (official)"; $D/uvtool/bin/uv sync >> $L 2>&1; say "uv sync exit $?"
say "torch: $(.venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.get_arch_list())' 2>&1 | tail -1)"
say "GPU check (official torch): $(.venv/bin/python -c 'import torch;x=torch.ones(2,device="cuda");print("ok", (x*2).sum().item())' 2>&1 | tail -1)"
if ! .venv/bin/python -c 'import torch;x=torch.ones(2,device="cuda");(x*2).sum().item()' >/dev/null 2>&1; then
  say "installing torch 2.10.0 cu126 (still has sm_61)"
  $D/uvtool/bin/uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
  say "torch now: $(.venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.get_arch_list())' 2>&1 | tail -1)"
  say "GPU check: $(.venv/bin/python -c 'import torch;x=torch.ones(2,device="cuda");print("ok", (x*2).sum().item())' 2>&1 | tail -1)"
fi
sed -e "s#/mnt/steam/ace-step/ACE-Step-1.5#$D/ACE-Step-1.5#; s#/mnt/steam/ace-step/work/out-#$D/out-#; s#offload_to_cpu=False#offload_to_cpu=True#" \
  ~/Code/volta-ace-step/scripts/repro.py > $D/repro.py
say "song 1: stock code, FP16 (automatic)"; timeout 3600 .venv/bin/python $D/repro.py p1070-stock >> $L 2>&1; say "exit $?"
git checkout -q -b volta && git -c user.name=x -c user.email=x@x am -q ~/Code/volta-ace-step/patches/*.patch && say "patches applied: $(git log --oneline -1)"
say "song 2: with fixes, FP16"; timeout 3600 .venv/bin/python $D/repro.py p1070-fix >> $L 2>&1; say "exit $?"
say "song 3: with fixes, ACESTEP_DTYPE=float32 (8 GB card: may not fit)"; ACESTEP_DTYPE=float32 timeout 3600 .venv/bin/python $D/repro.py p1070-fp32 >> $L 2>&1; say "exit $?"
say "TEST5 DONE"

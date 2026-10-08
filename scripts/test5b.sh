#!/usr/bin/env bash
# Test 5, second try: the first try's "uv pip install torch==2.10.0 --index-url cu126" was a no-op (same version
# number already installed as +cu128, which lacks sm_61). This waits for the first try's song 1 (which is also doing
# the model download) to finish, then force-reinstalls the cu126 build and runs the three songs.
set -u; D=/mnt/data/ace-step; L=$D/test5.log; cd $D/ACE-Step-1.5
say(){ echo "$(date +%T) $*" | tee -a $L; }
export UV_CACHE_DIR=$D/uv-cache UV_PYTHON_INSTALL_DIR=$D/uv-python HF_HOME=$D/hf
while kill -0 2093387 2>/dev/null; do sleep 20; done
say "first try's song 1 finished (official cu128 torch, no sm_61); retrying with the cu126 build"
say "reinstalling torch 2.10.0 cu126"
$D/uvtool/bin/uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
say "torch now: $(.venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.get_arch_list())' 2>&1 | tail -1)"
say "GPU check: $(.venv/bin/python -c 'import torch;x=torch.ones(2,device="cuda");print("ok", (x*2).sum().item())' 2>&1 | tail -1)"
say "song 1: stock code, FP16 (automatic)"; timeout 3600 .venv/bin/python $D/repro.py p1070-stock >> $L 2>&1; say "exit $?"
git checkout -q -b volta && git -c user.name=x -c user.email=x@x am -q ~/Code/volta-ace-step/patches/*.patch && say "patches applied: $(git log --oneline -1)"
say "song 2: with fixes, FP16"; timeout 3600 .venv/bin/python $D/repro.py p1070-fix >> $L 2>&1; say "exit $?"
say "song 3: with fixes, ACESTEP_DTYPE=float32 (8 GB card: may not fit)"; ACESTEP_DTYPE=float32 timeout 3600 .venv/bin/python $D/repro.py p1070-fp32 >> $L 2>&1; say "exit $?"
say "TEST5 DONE"

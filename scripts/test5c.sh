#!/usr/bin/env bash
# Test 5, low-VRAM retry: song 2 OOMed (8 GB card, 1.3 GB held by a pinned ollama model). DiT offload on too.
set -u; D=/mnt/data/ace-step; L=$D/test5.log; cd $D/ACE-Step-1.5
say(){ echo "$(date +%T) $*" | tee -a $L; }
export UV_CACHE_DIR=$D/uv-cache HF_HOME=$D/hf PYTORCH_ALLOC_CONF=expandable_segments:True
say "song 2b: with fixes, FP16, DiT offload too"; timeout 3600 .venv/bin/python $D/repro_lowvram.py p1070-fix-lowvram >> $L 2>&1; say "exit $?"
git stash -q 2>/dev/null; git checkout -q ca1e85f && say "back on stock code: $(git log --oneline -1)"
say "song 1b: stock code, FP16, DiT offload too (same settings, for timing)"; timeout 3600 .venv/bin/python $D/repro_lowvram.py p1070-stock-lowvram >> $L 2>&1; say "exit $?"
git checkout -q volta
say "TEST5C DONE"

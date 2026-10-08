#!/usr/bin/env bash
# Test 5, with fix 2b (patch 0006: only the lyric encoder's last projection in float32) -- the song that OOMed /
# hit the offload dtype bug with the old fix 2.
set -u; D=/mnt/data/ace-step; L=$D/test5.log; cd $D/ACE-Step-1.5
say(){ echo "$(date +%T) $*" | tee -a $L; }
export HF_HOME=$D/hf PYTORCH_ALLOC_CONF=expandable_segments:True
git -c user.name=x -c user.email=x@x am -q ~/Code/volta-ace-step/patches/0006-*.patch && say "patch 0006 applied: $(git log --oneline -1)"
say "song 4: all fixes (0001-0006), FP16, CPU offload (the setting that OOMed)"; timeout 3600 .venv/bin/python $D/repro.py p1070-fix2b >> $L 2>&1; say "exit $?"
say "song 5: all fixes, FP16, DiT offload too (the setting that hit the dtype bug)"; timeout 3600 .venv/bin/python $D/repro_lowvram.py p1070-fix2b-lowvram >> $L 2>&1; say "exit $?"
say "TEST5D DONE"

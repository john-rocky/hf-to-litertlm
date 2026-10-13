#!/bin/zsh
# Round 5 of the Qwen3-ASR-1.7B speed ladder: build F6 (run under r5_job.sh). r5_build.sh F6
# F6 = F3 (15 s window, encoder fp16) + prefill buckets 128,256: r3_build.sh's F1 steps with buckets 128,256 and the
# round's own roots (out/r5), then the pack of F3 with this LM:
#   fp32 export 30 s / KV 1024 / prefill 128,256 (export_fp32.py) -> unpack -> arm F6_C (quant_r5.py: LM + embedder
#   dynamic_wi8_afp32 = the ship recipe, encoder fp16) -> unpack F6_C_src -> pack_r5.py with F3's 15 s fp16 encoder
#   (out/r3/enc/enc15/enc__fp16.tflite, sha256 asserted = F3's encoder section) and KV 1024.
# A step whose output exists is skipped (a restart redoes only what is missing).
set -eu
cd ${0:A:h:h:h}  # qwen3_asr17_work
PYX=$HOME/venvs/ltmain0918/bin/python
LTLM=$HOME/venvs/lt0171run/bin/litert-lm
export PYTHONPATH=out/pyoverlay_builder017
step() { echo "[$(date '+%H:%M:%S')] STEP $1 epoch $(/usr/bin/python3 -c 'import time; print(f"{time.time():.3f}")')"; }
disk() {
  local free=$(df -g . | awk 'NR==2 {print $4}')
  echo "disk free ${free} GiB"
  if (( free < 60 )); then echo "STOP: disk free below 60 GiB"; exit 3; fi
}
F3_ENC_SHA=3f6394ff85db33f3c2b6560c1fefb626d3b8e2610b21667ed4f92aa462fa1385  # speed/r3/forms.json F3_win15 Section5
case $1 in
  F6)
    mkdir -p out/r5/unpack out/r5/export
    disk
    if [[ -f out/r5/f32_pb128_256.done ]]; then echo "skip export f32_pb128_256"; else
      step "export f32_pb128_256"
      $PYX export_fp32.py --out out/r5/f32_pb128_256 --input_sec 30 --cache_length 1024 --prefill_lengths 128,256
      touch out/r5/f32_pb128_256.done
    fi
    if [[ -f out/r5/unpack/f32_pb128_256.done ]]; then echo "skip unpack f32_pb128_256"; else
      step "unpack f32_pb128_256"
      $LTLM unpack out/r5/f32_pb128_256/model.litertlm --output-dir out/r5/unpack/f32_pb128_256
      touch out/r5/unpack/f32_pb128_256.done
    fi
    disk
    if [[ -f out/r5/export/F6_C.done ]]; then echo "skip arm F6_C"; else
      step "arm F6_C (lm dynamic_wi8_afp32)"
      $PYX speed/scripts/quant_r5.py --src out/r5/unpack/f32_pb128_256 --tag F6_30s_pb --arm F6_C \
          --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack
      touch out/r5/export/F6_C.done
    fi
    if [[ -f out/r5/unpack/F6_C_src.done ]]; then echo "skip unpack F6_C_src"; else
      step "unpack F6_C_src"
      $LTLM unpack out/r5/export/F6_C/model.litertlm --output-dir out/r5/unpack/F6_C_src
      touch out/r5/unpack/F6_C_src.done
    fi
    step "pack F6_win15_pb128_256"
    $PYX speed/scripts/pack_r5.py --src out/r5/unpack/F6_C_src --enc out/r3/enc/enc15/enc__fp16.tflite \
        --expect_enc_sha256 $F3_ENC_SHA --name F6_win15_pb128_256 --kv 1024 --prompt litert \
        --form_out speed/forms/F6_win15_pb128_256/model.litertlm
    ;;
  *) echo "usage: r5_build.sh F6"; exit 2 ;;
esac
step "done $1"

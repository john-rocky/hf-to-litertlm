#!/bin/bash
# Mac timing rows (card numbers) for the ship candidates: per-L, v2 ids on CPU / Metal f32 / Metal default, embeds on Metal f32 / default.
# Runs after the L512 chain, inside one quiet_hold window; --timing = fixture-median row, 300 / 200 / 128-token rows, median of 20 after 5 warm-ups.
set -u
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
P=${P:-python3}
cd "$W"
true
run() { "$P" scripts/litert_parity.py "$@" --limit 40 --timing --tag _timing2 2>&1 | grep -v "W1010\|Warning\|NOTE:\|absl\|I0000\|INFO" | grep '"total_ms"\|"id"\|"median"' | head -20; }
for L in 256 384 512; do
  echo "== L$L v2 ids cpu8 $(date '+%T')";        run --tflite exports/phocinae_L${L}_ids_v2_fp16fc_i8emb_shift.tflite --L $L --accel cpu --threads 8
  echo "== L$L v2 ids gpu f32";                   run --tflite exports/phocinae_L${L}_ids_v2_fp16fc_i8emb_shift.tflite --L $L --accel gpu --f32
  echo "== L$L v2 ids gpu default";               run --tflite exports/phocinae_L${L}_ids_v2_fp16fc_i8emb_shift.tflite --L $L --accel gpu
  echo "== L$L embeds gpu f32";                   run --tflite exports/phocinae_L${L}_embeds_wfp16_shift.tflite --L $L --accel gpu --f32 --table exports/tables/tok_embeddings_fp16.bin
  echo "== L$L embeds gpu default";               run --tflite exports/phocinae_L${L}_embeds_wfp16_shift.tflite --L $L --accel gpu --table exports/tables/tok_embeddings_fp16.bin
  echo "== L$L embeds cpu8";                      run --tflite exports/phocinae_L${L}_embeds_wfp16_shift.tflite --L $L --accel cpu --threads 8 --table exports/tables/tok_embeddings_fp16.bin
done
echo "== timing done $(date '+%T')"

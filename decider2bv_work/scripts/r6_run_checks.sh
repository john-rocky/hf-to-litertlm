#!/bin/zsh
# Round 6 final: public-fixture check of the reference in the MINIMAL venv (out/venv-ref: numpy, pillow, tokenizers,
# ai-edge-litert only; lock requirements-lock-reference.txt) on v7c then fp16 (CPU, 8 threads), then the bit-identity
# comparison with round 4's CPU graph readouts.
#   scripts/r6_run_checks.sh   (from decider2bv_work/)
cd "$(dirname $0)/.."
for v in v7c fp16; do
  echo "# $(date '+%F %T %Z') check $v"
  out/venv-ref/bin/python -B publish/reference/check_fixtures.py --bundle out/bundle_r4/$v/decider-2b-vision_$v.litertlm \
    --cache-dir out/ref_cache --out results/r6/check_${v}_cpu.json > logs/r6/check_${v}_cpu.log 2> logs/r6/check_${v}_cpu.stderr
  echo "# $(date '+%F %T %Z') check $v exit $?"
  python3 -B scripts/r6_bit_identity.py --variant $v --check results/r6/check_${v}_cpu.json
done
echo "# $(date '+%F %T %Z') done"

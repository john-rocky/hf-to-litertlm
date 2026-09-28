#!/bin/zsh
# Round 6 addendum (supervisor ship plan A, 03:0x JST): the reference on the int8 (dyn8) bundle, CPU exact-fit + bit
# identity with round 4's readout_r4_dyn8.json, then the GPU rule; minimal venv.
cd "$(dirname $0)/.."
v=dyn8
echo "# $(date '+%F %T %Z') check $v cpu"
out/venv-ref/bin/python -B publish/reference/check_fixtures.py --bundle out/bundle_r4/$v/decider-2b-vision_$v.litertlm \
  --cache-dir out/ref_cache --out results/r6/check_${v}_cpu.json > logs/r6/check_${v}_cpu.log 2> logs/r6/check_${v}_cpu.stderr
echo "# $(date '+%F %T %Z') exit $?"
python3 -B scripts/r6_bit_identity.py --variant $v --check results/r6/check_${v}_cpu.json
echo "# $(date '+%F %T %Z') check $v gpu"
out/venv-ref/bin/python -B publish/reference/check_fixtures.py --bundle out/bundle_r4/$v/decider-2b-vision_$v.litertlm \
  --backend gpu --cache-dir out/ref_cache --out results/r6/check_${v}_gpu.json > logs/r6/check_${v}_gpu.log 2> logs/r6/check_${v}_gpu.stderr
echo "# $(date '+%F %T %Z') exit $?"

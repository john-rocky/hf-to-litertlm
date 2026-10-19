#!/bin/bash
# The three reference models through the pinned toolchain, twice each (docs/one-command-lock.md
# "Raising the toolchain"): convert.py (build 1: lock + 8-question gate + check) -> verify_lock.py
# (build 2 from the lock, compared byte for byte) -> check_bundle.py (litert-lm CPU, one prompt).
# Each lock is kept as locks/refs/<name>.<toolchain id>.lock.json (never overwritten), and
# locks/refs/CHANGELOG.md gets one row per model and toolchain. Prints a table at the end.
#
#   uv sync && bash scripts/verify_refs.sh                 # all three
#   bash scripts/verify_refs.sh google/gemma-4-E2B-it      # one of them
set -uo pipefail
cd "$(dirname "$0")/.."

REFS=(
  "Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674"
  "google/gemma-4-E2B-it 3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
ONLY="${1:-}"
LOGS="out/verify_refs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS" locks/refs
CHANGELOG=locks/refs/CHANGELOG.md
[ -f "$CHANGELOG" ] || printf '%s\n' \
  "# Reference locks, one row per model and toolchain" "" \
  "Written by scripts/verify_refs.sh. Build 1 = convert.py, build 2 = verify_lock.py's rebuild; peak RSS from /usr/bin/time -l." \
  "compat = the toolchain-skew shims convert.py applied (recipe.compat of the lock); each one goes away with the release that fixes it." "" \
  "| date | toolchain | model @ revision | bundle sha256 | vs the previous toolchain | build 2 | check (litert-lm CPU) | 17 + 25 (information) | compat | seconds build 1 / build 2 | peak RSS build 1 / build 2 |" \
  "|---|---|---|---|---|---|---|---|---|---|---|" > "$CHANGELOG"

# timed <log> <command...> -> "<exit> <seconds> <peak RSS GB>", the command's output in <log>
timed() {
  local log=$1; shift
  local t0; t0=$(date +%s)
  /usr/bin/time -l "$@" > "$log" 2>&1
  local rc=$?
  local rss; rss=$(awk '/maximum resident set size/ {printf "%.1f", $1 / 1e9}' "$log" | tail -1)
  echo "$rc $(( $(date +%s) - t0 )) ${rss:-?}"
}

json() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@"; }

TABLE="| model | build 1 | gate | build 2 | check | seconds 1 / 2 | peak RSS GB 1 / 2 | bundle sha256 |"$'\n'"|---|---|---|---|---|---|---|---|"
STATUS=0
for ref in "${REFS[@]}"; do
  read -r repo rev <<< "$ref"
  [ -n "$ONLY" ] && [ "$ONLY" != "$repo" ] && continue
  name=${repo##*/}
  echo "=== $repo @ $rev"
  read -r c_rc c_s c_rss <<< "$(timed "$LOGS/$name.convert.log" uv run python scripts/convert.py "$repo" --revision "$rev")"
  grep -E '^(FAIL|lock:|converted_|source:)' "$LOGS/$name.convert.log"
  lock="out/$name/$name.lock.json"
  if [ ! -f "$lock" ]; then
    TABLE+=$'\n'"| $name | exit $c_rc, no lock | - | - | - | $c_s / - | $c_rss / - | - |"
    STATUS=1; continue
  fi
  gate=$(json "out/$name/convert_report.json" "f\"{d['gate']['score']}/8\" if d.get('gate') else 'none'")
  read -r v_rc v_s v_rss <<< "$(timed "$LOGS/$name.verify.log" uv run python scripts/verify_lock.py "$lock")"
  verdict=$(grep -oE '^(MATCH|DIFF|FAIL)' "$LOGS/$name.verify.log" | head -1)
  grep -E '^(MATCH|DIFF|FAIL|  section|  system_metadata)' "$LOGS/$name.verify.log"
  bundle=$(json "$lock" "'out/$name/' + d['bundle']['file']")
  timed "$LOGS/$name.check.log" uv run python scripts/check_bundle.py "$bundle" > /dev/null
  check=$(python3 -c "import json,sys; t=open(sys.argv[1]).read(); d=json.loads(t[t.index('{'):t.rindex('}')+1]); a=d.get('arithmetic') or {}; print(('PASS' if d['pass'] else 'FAIL') + ' ' + repr(d['answer_head']) + ' (litert-lm ' + str(d['engine']) + '; 17 + 25 -> ' + repr(a.get('answer_head')) + ', expected 42)')" "$LOGS/$name.check.log" 2>/dev/null || echo "FAIL (no result)")
  echo "check: $check"
  sha=$(json "$lock" "d['bundle']['sha256']")
  arith=$(json "$lock" "repr(((d.get('check') or {}).get('arithmetic') or {}).get('answer_head'))")
  compat=$(json "$lock" "', '.join(d['recipe'].get('compat') or []) or 'none'")
  tid=$(json "$lock" "d['toolchain']['id']")
  kept="locks/refs/$name.$tid.lock.json"
  if [ -f "$kept" ] && [ "$(json "$kept" "d['bundle']['sha256']")" != "$sha" ]; then
    echo "NOTE: $kept records a different sha256 for the same toolchain; kept as is"
  elif [ ! -f "$kept" ]; then
    cp "$lock" "$kept"
  fi
  prev=$(ls locks/refs/"$name".*.lock.json 2>/dev/null | grep -v "/$name.$tid.lock.json" | tail -1)
  if [ -z "$prev" ]; then vs="first lock"
  elif [ "$(json "$prev" "d['bundle']['sha256']")" = "$sha" ]; then vs="same bytes"
  else vs="changed bytes"; fi
  if ! grep -q "| $tid | $repo @" "$CHANGELOG"; then
    echo "| $(date +%Y-%m-%d) | $tid | $repo @ ${rev:0:8} | \`${sha:0:16}\` | $vs | ${verdict:-?} | ${check%% (*} | $arith (expected 42) | $compat | $c_s / $v_s | $c_rss / $v_rss GB |" >> "$CHANGELOG"
  fi
  TABLE+=$'\n'"| $name | exit $c_rc | $gate | ${verdict:-?} | ${check%% *} | $c_s / $v_s | $c_rss / $v_rss | \`${sha:0:16}\` |"
  { [ "$c_rc" = 0 ] && [ "$verdict" = MATCH ] && [ "${check%% *}" = PASS ]; } || STATUS=1
done
echo
echo "$TABLE"
echo
echo "logs: $LOGS"
exit $STATUS

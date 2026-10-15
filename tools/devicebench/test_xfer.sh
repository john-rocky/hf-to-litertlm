#!/bin/bash
# test_xfer.sh: chain.sh's xfer() (the exec-in push with its byte-count poll) against a fake phone and a fake clock, so
# the poll schedule is tested without a phone and without waiting: fake_adb.py writes a whole file before exec-in
# returns, so the dry run never reaches the polls. Cases: complete at once, complete 0.7 s after exec-in returned,
# stuck (no growth for 30 s), slow but growing, never complete (XFER_MAX_S), the phone lost in the transfer.
#   bash test_xfer.sh [chain.sh]        (default: chain.sh next to this file; exit 0 = every case as expected)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHAIN=${1:-$HERE/chain.sh}
eval "$(awk '/^xfer\(\) \{/,/^}$/' "$CHAIN")"
declare -F xfer > /dev/null || { echo "no xfer() in $CHAIN"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
SRC=$T/src.bin; head -c 2000000 /dev/zero > "$SRC"; WANT=2000000
PKG=fake.pkg; XFER_MAX_S=900
CLOCK=0; SCEN=done; DONE_AT=0; RATE=0; LOST=0; bad=0; n=0

ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
now() { echo "${CLOCK%.*}"; }
sleep() { CLOCK=$(awk -v c="$CLOCK" -v d="$1" 'BEGIN { printf "%.3f", c + d }'); echo "$1" >> "$T/sleeps"; }
present() { [ "$LOST" = 0 ]; }
hard_stop() { echo "HARD STOP: $1"; exit 9; }   # as lib.sh's; xfer runs in $(...) here, so only that subshell exits
size_now() {  # the phone's byte count of the file at the fake clock
  case "$SCEN" in
    done) echo "$WANT" ;;
    lag) if ge "$CLOCK" "$DONE_AT"; then echo "$WANT"; else echo $(( WANT / 2 )); fi ;;
    stuck) echo $(( WANT / 2 )) ;;
    slow|never) awk -v c="$CLOCK" -v r="$RATE" -v w="$WANT" -v s="$SCEN" \
                  'BEGIN { x = int(c * r); m = (s == "never") ? w - 1 : w; print (x > m) ? m : x }' ;;
  esac
}
A() {  # the fake adb: exec-in swallows stdin; stat answers size_now; rm is recorded
  case "$*" in
    exec-in*) cat > /dev/null; return 0 ;;
    *" stat -c %s "*) echo "stat $CLOCK" >> "$T/polls"; size_now ;;
    *" rm -f "*) echo "rm $*" >> "$T/rm" ;;
  esac
}
case_() {  # name scen done_at rate want_rc want_max_s [want_rm]: run xfer once; check its exit code, the fake seconds, rm
  local name=$1 rc out s polls rmd
  SCEN=$2; DONE_AT=$3; RATE=$4; CLOCK=0; : > "$T/sleeps"; : > "$T/polls"; : > "$T/rm"
  out=$(xfer "$SRC" f.bin); rc=$?
  # xfer ran in a subshell: its clock is the sum of its sleeps
  s=$(awk '{ t += $1 } END { printf "%.1f", t }' "$T/sleeps"); polls=$(wc -l < "$T/polls" | tr -d ' ')
  rmd=$([ -s "$T/rm" ] && echo yes || echo no)
  n=$((n + 1))
  if [ "$rc" = "$5" ] && ge "$6" "$s" && [ "$rmd" = "${7:-no}" ]; then
    echo "ok   $name: rc $rc, ${s}s of sleeps, $polls polls, removed=$rmd | $out"
  else
    echo "FAIL $name: rc $rc (want $5), ${s}s (want <= $6), removed=$rmd (want ${7:-no}) | $out"; bad=1
  fi
}
case_ "complete when exec-in returns" done 0 0 0 0
case_ "complete 0.7 s after exec-in returned" lag 0.7 0 0 1
case_ "stuck at half: stops after 30 s without growth, removes the file" stuck 0 0 1 36 yes
case_ "slow but growing (100 kB/s): never stops early" slow 0 100000 0 30
XFER_MAX_S=60; case_ "never complete: stops at XFER_MAX_S, removes the file" never 0 20000 1 66 yes; XFER_MAX_S=900
LOST=1; out=$(xfer "$SRC" f.bin); rc=$?; LOST=0; n=$((n + 1))
if [ "$rc" = 9 ] && echo "$out" | grep -q "HARD STOP: device_lost_in_transfer_f.bin"; then echo "ok   the phone lost: hard stop"; else echo "FAIL the phone lost: rc $rc | $out"; bad=1; fi
[ $bad = 0 ] && echo "PASS test_xfer: $n cases ($(basename "$CHAIN"))" || echo "FAIL test_xfer ($(basename "$CHAIN"))"
exit $bad

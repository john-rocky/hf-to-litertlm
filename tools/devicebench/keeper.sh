#!/bin/bash
# keeper.sh: the hold keeper of one chain run: the named waiting line (queue_cli.py) and an owned hold (device_hold.py
# through hold_cli.py). chain.sh starts it from a copy in the round dir (one keeper per hold), so an edit of this file
# never reaches a running keeper.
#   usage: keeper.sh <signal dir> <hold file> <script name>        env: CA (the hold tools' dir), SERIAL
# 1. `queue_cli.py wait <hold file> <script name> <this shell's pid>`: in the line behind the entries whose keepers are
#    waiting, acquires the owned hold with THIS shell's pid when the hold is free; never forces, never clears another
#    session's hold (a dead holder's stale hold is cleared by device_hold.acquire itself);
# 2. writes <signal dir>/HOLD_ACQUIRED ("<serial>", "pid <pid> since <time>");
# 3. keeps the hold until <signal dir>/RELEASE_HOLD appears or HOLD_LIMIT seconds (default 1,800 = the 30-minute window)
#    after acquiring, then `hold_cli.py release <hold file> <pid>` (removes the hold only when it is this pid's) and writes
#    <signal dir>/HOLD_RELEASED. Gives up waiting after WAIT_LIMIT seconds (default 4 h; queue_cli exit 3 -> GAVE_UP).
set -u
SIG=${1:?signal dir}; LOCK=${2:?hold file}; NAME=${3:?script name}
CA=${CA:?CA = the directory of queue_cli.py and hold_cli.py}
SERIAL=${SERIAL:?SERIAL = the phone}
HOLD_LIMIT=${HOLD_LIMIT:-1800}
WAIT_LIMIT=${WAIT_LIMIT:-14400}
mkdir -p "$SIG"; rm -f "$SIG/HOLD_ACQUIRED" "$SIG/RELEASE_HOLD" "$SIG/HOLD_RELEASED" "$SIG/GAVE_UP"
echo "keeper pid $$ started $(date '+%F %T') hold=$LOCK name=$NAME hold_limit=${HOLD_LIMIT}s wait_limit=${WAIT_LIMIT}s"
echo "line at start:"; python3 "$CA/queue_cli.py" list "$LOCK" | sed 's/^/  /'
if ! python3 "$CA/queue_cli.py" wait "$LOCK" "$NAME" $$ --timeout "$WAIT_LIMIT" --poll 2; then
  echo "gave up waiting $(date '+%F %T')"; date '+%F %T' > "$SIG/GAVE_UP"; exit 3
fi
ACQ=$(date +%s)
END=$(( ACQ + HOLD_LIMIT ))
{ echo "$SERIAL"; echo "pid $$ since $(date -r "$ACQ" '+%Y-%m-%d %H:%M:%S')"; } > "$SIG/HOLD_ACQUIRED"
echo "HOLD ACQUIRED by pid $$ at $(date -r "$ACQ" '+%F %T'); hold file: $(cat "$LOCK")"
echo "auto release at $(date -r $END '+%T')"
while [ ! -f "$SIG/RELEASE_HOLD" ]; do
  if [ "$(date +%s)" -ge "$END" ]; then echo "hold limit reached $(date '+%F %T')"; break; fi
  sleep 2
done
python3 "$CA/hold_cli.py" release "$LOCK" $$
rm -f "$SIG/HOLD_ACQUIRED"
date '+%F %T' > "$SIG/HOLD_RELEASED"
echo "HOLD RELEASED at $(date '+%F %T'); hold file now: $(cat "$LOCK" 2>/dev/null || echo '<absent>')"

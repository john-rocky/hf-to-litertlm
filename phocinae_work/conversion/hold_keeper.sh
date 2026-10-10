#!/bin/bash
# Galaxy S26 hold keeper for the phocinae-litert lane. Enqueues itself in the phone's waiting line (queue_cli.py), waits
# cooperatively until the hold is free and no entry ahead of it has a waiting keeper, takes the hold with THIS shell's
# pid (hold_cli / device_hold: owned by a live pid), writes device/HOLD_ACQUIRED, keeps it until device/RELEASE_HOLD
# appears or the safety limit passes, then releases and dequeues. Never forces a hold.
#   nohup scripts/hold_keeper.sh > device/keeper.log 2>&1 &     # then wait for device/HOLD_ACQUIRED
#   touch device/RELEASE_HOLD                                    # when the batch is done
SIG=${W:-$(cd "$(dirname "$0")/.." && pwd)}/device
CA=${CA:-$(cd "$(dirname "$0")/../../community_accel_work" && pwd)}
LOCK=$CA/s2_npu_sweep/.device_hold
NAME=phocinae-litert
MINUTES=${MINUTES:-40}
WAIT_LIMIT=${WAIT_LIMIT:-14400}
HOLD_LIMIT=${HOLD_LIMIT:-5400}
mkdir -p "$SIG"
# NO_CLEAN=1: a second keeper queued behind this lane's own expiring hold leaves the first keeper's markers alone
[ -n "${NO_CLEAN:-}" ] || rm -f "$SIG/HOLD_ACQUIRED" "$SIG/RELEASE_HOLD" "$SIG/HOLD_RELEASED" "$SIG/GAVE_UP"
echo "keeper pid $$ started $(date '+%F %T') name=$NAME minutes=$MINUTES wait_limit=$WAIT_LIMIT hold_limit=$HOLD_LIMIT"
python3 "$CA/queue_cli.py" enqueue "$LOCK" "$NAME" phocinae-litert "$MINUTES"
python3 "$CA/queue_cli.py" list "$LOCK"
if ! python3 "$CA/queue_cli.py" wait "$LOCK" "$NAME" $$ --timeout "$WAIT_LIMIT" --poll "${POLL:-15}"; then
  echo "gave up waiting $(date '+%F %T')"; date '+%F %T' > "$SIG/GAVE_UP"; python3 "$CA/queue_cli.py" dequeue "$LOCK" "$NAME"; exit 3
fi
ACQ=$(date +%s)
echo "RFGL80R6A6H" > "$SIG/HOLD_ACQUIRED"
echo "HOLD ACQUIRED by pid $$ at $(date '+%F %T')"
python3 "$CA/hold_cli.py" read "$LOCK"
while [ ! -f "$SIG/RELEASE_HOLD" ]; do
  if [ $(( $(date +%s) - ACQ )) -gt $HOLD_LIMIT ]; then echo "hold safety limit reached $(date '+%F %T')"; break; fi
  sleep 10
done
python3 "$CA/hold_cli.py" release "$LOCK" $$
rm -f "$SIG/HOLD_ACQUIRED"
date '+%F %T' > "$SIG/HOLD_RELEASED"
echo "HOLD RELEASED at $(date '+%F %T')"

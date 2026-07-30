#!/bin/bash
# One-command device verification of the CompiledModel (CLiteRT) rework on the
# connected iPhone (USB, unlocked). The bundle id reuses the existing Bonsai
# container, so the 3.97 GiB model set already on the phone is picked up as-is
# (first-ever run on a device: see device_run.sh header for the one-time
# model copy command).
#
# Usage: ./device_verify.sh [seed] [steps]
# PASS = console (or Documents) shows step timings ~12-13 s/step and a new
# bonsai_*_seed<seed>.png appears; the seed-7 image should match the previous
# classic-API build's seed-7 output (same noise, same XNNPACK numerics).
set -euo pipefail
cd "$(dirname "$0")"
UDID="${BONSAI_UDID:?set BONSAI_UDID to your device UDID (xcrun devicectl list devices)}"
SEED="${1:-7}"
STEPS="${2:-4}"

[ -d CLiteRT.xcframework ] || { echo "run ./prep_clitert.sh first"; exit 1; }
[ -f Resources/vocab.json ] || { echo "run ./prep_resources.sh first"; exit 1; }
[ -d BonsaiApp.xcodeproj ] || xcodegen generate

echo "== building for device =="
xcodebuild -project BonsaiApp.xcodeproj -scheme BonsaiApp -configuration Release \
  -destination "generic/platform=iOS" -derivedDataPath build build \
  | grep -E "^\*\*|error:" || true
APP=build/Build/Products/Release-iphoneos/BonsaiApp.app
[ -d "$APP" ] || { echo "device build failed"; exit 1; }

echo "== installing =="
xcrun devicectl device install app --device "$UDID" "$APP"

echo "== launching (autorun seed=$SEED steps=$STEPS) =="
ENV=$(printf '{"BONSAI_AUTORUN":"1","BONSAI_SEED":"%s","BONSAI_STEPS":"%s"}' "$SEED" "$STEPS")
xcrun devicectl device process launch --console --terminate-existing \
    --environment-variables "$ENV" --device "$UDID" com.bonsai.devicetest \
    || echo "CLI launch failed — app is installed; launch from the home screen."

echo "== fetch the result when done =="
echo "xcrun devicectl device copy from --device $UDID \\"
echo "  --domain-type appDataContainer --domain-identifier com.bonsai.devicetest \\"
echo "  --source Documents --destination /tmp/bonsai_device_out"

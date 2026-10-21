#!/bin/bash
# Build redux_asr_runner_b for Android arm64 against LiteRT 2.3.0 and collect the LiteRT runtime libraries it loads.
#   NDK=<Android NDK r29 directory> bash build_android.sh
# Needs curl, unzip and shasum (macOS) or sha256sum (Linux). Writes next to this script:
#   build/litert_cc_sdk/   the LiteRT C API headers (GitHub release v2.3.0 asset litert_cc_sdk.zip)
#   libs/libLiteRt.so                                  Maven com.google.ai.edge.litert:litert:2.3.0
#   libs/libLiteRtClGlAccelerator.so                   Maven com.google.ai.edge.litert:litert-gpu:2.3.0 (--accel gpu)
#   libs/libLiteRtDispatch_Qualcomm.so, libs/libLiteRtCompilerPlugin_Qualcomm.so
#                                                      Maven com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0
#   redux_asr_runner_b     the Android arm64 binary (API level 28, libc++ linked statically)
# Every download is checked against the sha256 below. With NDK 29.0.13113456 on macOS the binary's sha256 is
# a8b00c6800c9ac9f9bc79116f9b1786f243a479850887efb1077c175b5361de6.
set -euo pipefail
cd "$(dirname "$0")"
NDK=${NDK:?set NDK to an Android NDK r29 directory, e.g. ~/Library/Android/sdk/ndk/29.0.13113456}
MAVEN=https://dl.google.com/android/maven2/com/google/ai/edge/litert
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fetch() {  # url file sha256: download once, then check
  if [ ! -f "$2" ]; then curl -fsSL -o "$2.part" "$1"; mv "$2.part" "$2"; fi
  local got; got=$(sha "$2")
  if [ "$got" != "$3" ]; then echo "$2: sha256 $got, expected $3" >&2; exit 1; fi
}
mkdir -p build libs
fetch https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/litert_cc_sdk.zip build/litert_cc_sdk.zip \
  44ebb188ae1a510f233639fa499928fdaaba46436a153a2dbfbb9d87d59728d2
fetch "$MAVEN/litert/2.3.0/litert-2.3.0.aar" build/litert-2.3.0.aar \
  9508c6e9142646f0f2bc1fe68b3925fb470ee7ddd29114540339a04a8ab189dc
fetch "$MAVEN/litert-gpu/2.3.0/litert-gpu-2.3.0.aar" build/litert-gpu-2.3.0.aar \
  173a4c3813601f7e9542f4e6a49e237775e2f9ce1e0948f35407098bffb1824f
fetch "$MAVEN/litert-npu-runtime-qualcomm/2.3.0/litert-npu-runtime-qualcomm-2.3.0.aar" build/litert-npu-runtime-qualcomm-2.3.0.aar \
  55aa22e89b7decb9e2f56548188b9e8c42cc8f7a828e2ec4a17715079b313679
rm -rf build/litert_cc_sdk
unzip -q build/litert_cc_sdk.zip 'litert_cc_sdk/litert/c/*' 'litert_cc_sdk/litert/build_common/*' -d build
# build_config.h is generated from build_config.h.in by CMake; GPU and NPU stay enabled (each toggle = 0)
sed -E 's/^#cmakedefine01 ([A-Z_]+)/#define \1 0/' build/litert_cc_sdk/litert/build_common/build_config.h.in \
  > build/litert_cc_sdk/litert/build_common/build_config.h
unzip -q -o -j build/litert-2.3.0.aar jni/arm64-v8a/libLiteRt.so -d libs
unzip -q -o -j build/litert-gpu-2.3.0.aar jni/arm64-v8a/libLiteRtClGlAccelerator.so -d libs
unzip -q -o -j build/litert-npu-runtime-qualcomm-2.3.0.aar 'jni/arm64-v8a/*.so' -d libs
CXX=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android28-clang++ | head -n 1)
"$CXX" -std=c++17 -O2 -Wall -Werror -static-libstdc++ -I build/litert_cc_sdk redux_asr_runner_b.cc -L libs -lLiteRt \
  -o redux_asr_runner_b
echo "redux_asr_runner_b $(sha redux_asr_runner_b)"
for f in libs/*.so; do echo "$f $(sha "$f")"; done

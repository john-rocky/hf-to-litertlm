# The measurement app behind the Galaxy S26 numbers

`d1omni_gate/` is the debug-only app that produced the phone's GPU (OpenCL) and CPU (XNNPACK) rows for the decision graphs, the vision tower, the projector and the audio encoder on a Samsung Galaxy S26 (SM-S942Q, 12 GB, Android 16) with LiteRT 2.2.0 from Maven. It is not a sample app: one Activity, a status line, a report file per run. The NPU rows and a second set of GPU rows came from a C-API runner on LiteRT 2.2.0, which is not included; neither are the NPU runtime libraries.

## Build

```bash
cd d1omni_gate
echo "sdk.dir=<your Android SDK>" > local.properties
keytool -genkeypair -keystore .local/debug.keystore -storepass android -alias androiddebugkey -keypass android \
  -dname "CN=Android Debug,O=Android,C=US" -keyalg RSA -validity 10000     # the build signs with .local/debug.keystore
JAVA_HOME=<JDK 17> ./gradlew :app:assembleDebug :app:testDebugUnitTest
```

Kotlin 2.2.21, Android Gradle Plugin 8.9.1, Gradle 8.11.1, compileSdk 35, arm64-v8a. The unit tests (22) cover the byte layout the scorer reads, the six inputs of a decision row, the shape rules and the generic mode's byte contract.

## What it runs

The app loads one graph with the Kotlin `CompiledModel` API and runs the rows a file lists:

- `mode=gate`: one call per row; for a decision graph the K scores at the option markers are saved, for any other graph (`mode=generic`) the whole output.
- `mode=timing` (`generic_timing`): warm-up calls, then timed rounds; one call = write the inputs + run + read the output (the GPU's `run()` returns at once and the work is waited for in the read).
- `accel=gpu|cpu`, `precision=fp32|fp16acc32|default` (GPU: FP32 or FP16_WITH_FP32_ACCUM), `threads=4` (CPU).
- `cool_ms`: before each timing set, wait until the GPU clock cap and temperature are back to their values before the compile.
- A file named `STOP` in the app's `files/` ends a run after the current row or round.

The rows, the plan of each phone run and the scoring are in `../conversion/`: `s26_rows.py` writes the rows, `s26_chain.sh` installs the app, pushes the graph, runs the legs and samples the phone's clocks, temperatures and memory every 2 s, and `s26_score.py` compares the saved scores with the reference on a Mac (the probabilities are read out by the host's own code).

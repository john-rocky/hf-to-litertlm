# Speech recognition route: a standalone ASR checkpoint → one `.litertlm` with Speech2TextMetadata

`scripts/convert.py` has a second route next to the LLM one. Given a standalone speech recognition
checkpoint (today: transformers' `ParakeetForTDT`, model_type `parakeet_tdt`), it writes one `.litertlm`
file in the form of litert-torch
[PR #1426](https://github.com/google-ai-edge/litert-torch/pull/1426) ("Package standalone ASR models into
.litertlm with Speech2TextMetadata", open as of 2026-11-03): a Speech2TextMetadata section, the repo's
own `tokenizer.json` and one `.tflite` with the `encode` / `decode` / `decode_1` signatures that LiteRT-LM's
Parakeet runtime runs. It gates the file by transcribing LibriSpeech clips through MediaPipe's
`AudioTranscriber` on the CPU. No NeMo: the graphs are plain PyTorch modules over the transformers checkpoint.

```bash
uv run python scripts/convert.py nvidia/parakeet-tdt-0.6b-v3 --smoke   # model_type parakeet_tdt takes the route by itself
uv run python scripts/convert.py <org>/<parakeet-tdt finetune> --task asr
# -> out/<name>/<name>_int8.litertlm, <name>.lock.json, convert_report.json, asr_gate.json
```

`--check` runs the entry gate only; `--smoke` gates on one clip (word error rate ≤ 0.2), the default
gate on 73 clips (≤ 0.10); `--no-gate` builds and locks only. The clips come through the Hugging Face cache
(`hf-internal-testing/librispeech_asr_dummy` at `5be91486`, sha256-checked). Exit codes and the report's
status words are convert.py's: 0 converted and gated (`smoke_pass`, `converted_pass`, `built`), 1 the gate
failed (`smoke_failed`, `converted_gate_failed`), 2 refused, 3 failed. `scripts/verify_lock.py` rebuilds
from the lock (it passes `--task asr`) and compares the bytes.

## Which runtime reads the file (checked 2026-11-03)

| runtime | reads it? | how we know |
|---|---|---|
| MediaPipe `AudioTranscriber`, pip `mediapipe` 1.2.0 (PyPI 2026-10-27; macOS arm64, Linux, Windows wheels), CPU | yes | run on a Mac (M4 Max): litert-community's own `parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm` and this route's file, the same clips (table below). The same task exists for Android (`com.google.mediapipe:tasks-audio`), iOS / macOS and Web per that model card; not run here |
| LiteRT-LM v0.19.0 `OmniEngine` / `omni_runner` (C++, built from source) | yes, by the source | `omni/omni_registry.cc` merges the file's Speech2TextMetadata into the session; `omni/speech2text/parakeet_tdt.cc` registers `parakeet-tdt-0.6b-v3`, `parakeet-tdt` and two int8 names. It picks the model by `--model_name` or by a registered name inside the file name, so this route names the file `<repo name>_int8.litertlm` (`parakeet-tdt-0.6b-v3_int8.litertlm`). Not built or run here |
| `litert-lm` 0.19.0 (pip: Python API and the `litert-lm` CLI) | no | `python/litert_lm/` at v0.19.0 and at main (2026-11-03) has no speech-to-text entry point |
| litert-lm-builder 0.18.0 (this repo's pinned writer) | cannot write it | 0.18.0 has no `add_speech2text_metadata` and no Speech2TextMetadata proto; 0.19.0 (PyPI 2026-10-28) has both |

So the file needs no LiteRT-LM 0.20.0: MediaPipe 1.2.0 runs it today, and the LiteRT-LM 0.19.0 C++
engine reads its sections. Only the writer is newer than this repo's pin, which is why the bundle and gate
steps run in their own locked environment (`asr/pyproject.toml`, `asr/uv.lock`: litert-lm-builder 0.19.0,
mediapipe 1.2.0) while the export runs in the root `uv.lock` (litert-torch 0.9.4, torch 2.13.0,
transformers 5.14.1).

## What the route accepts

| model_type | route | why |
|---|---|---|
| `parakeet_tdt` (transformers `ParakeetForTDT`) | converted | nvidia/parakeet-tdt-0.6b-v3 and finetunes saved with transformers |
| `parakeet_ctc` | refused | an encoder-only CTC graph (`TF_LITE_AUDIO_ENCODER`); this route exports TDT only so far |
| `parakeet_rnnt` | refused | RNN-T without durations; the v0.19.0 Parakeet runtime decodes TDT logits |
| `whisper`, `moonshine` | refused | their encoder-decoder graphs are not exported by this route; litert-torch's own `Whisper` and `Moonshine` wrappers (transformers models, no NeMo) get their Speech2TextMetadata in PR #1426 |
| `nemotron_asr_streaming`, `nemotron3_5_asr` | refused | a cache-aware streaming RNN-T: Speech2TextModelType has no entry for it; `nemotron_asr_stream_work/` converts it to plain `.tflite` graphs for a host loop |
| any other, with `--task asr` | refused | not a standalone speech recognition type |
| a `.nemo`-only repo (nvidia/parakeet-tdt-0.6b-v2) | refused | no transformers `config.json` / safetensors |

The export also refuses a `parakeet_tdt` checkpoint the v0.19.0 runtime would read wrong: an encoder
width other than 1024 (`tdt_decoder.cc` reads the encoder frames as numel / 1024), durations other than
0..K−1 (it steps by the duration index), a blank that is not the last token, or a front end other than the
runtime's Parakeet default (128 mel bins, 16 kHz, 25 ms window, 10 ms hop, n_fft 512, pre-emphasis 0.97).
A repo without `processor_config.json` (moondream/parakeet-ultra ships `config.json`, the weights and
`tokenizer.json` only) is taken with that default front end, the encoder's `num_mel_bins` checked, and the
report says so.

## The file

Three sections, in the order PR #1426's `package_speech2text_model` writes them:

| # | section | content |
|---|---|---|
| 0 | `Speech2TextMetadataProto` (value 11; litert-lm-builder 0.18.0's schema names the same value `AsrMetadataProto`, the name in the lock's section list) | `asr_model_type { parakeet {} }`, `input_milliseconds: 5000`, `log_mel_spectrogram_config { n_frames: 500 }`, `decode_statefully_after: 4` |
| 1 | `HF_Tokenizer_Zlib` | the repo's `tokenizer.json` at the pinned commit |
| 2 | `TFLiteModel`, `model_type: tf_lite_encoder_decoder` | the graphs below, int8 weights |

The `.tflite` (shapes for the 5 s window; the same signatures, shapes and dtypes as
litert-community's `parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite`, read back from both files):

| signature | inputs | outputs |
|---|---|---|
| `encode` | `args_0` [1, 128, 500] f32 (the runtime's log-mel, mel-major) | `output_0` [1, 1024, 63] f32 |
| `decode` | `args_0` [1, 1024, 63] f32, `args_1` [1, 4] i32, `args_2` / `args_3` [2, 1, 640] f32 (LSTM h, c) | `output_0` [1, 63, 4, 8198] f32 (log-softmax over 8,192 tokens + blank + 5 durations), `output_1` / `output_2` [2, 1, 640] |
| `decode_1` | the same with `args_1` [1, 1] | `output_0` [1, 63, 1, 8198], the states |

The runtime runs `decode` from zero states on the blank and up to three emitted tokens, then switches to
`decode_1` with the states `decode` returned (`TdtDecoder` in `omni/speech2text/tdt_decoder.cc`).

## Field by field against PR #1426

PR #1426 head `a57c1842` (2026-11-02): `package_speech2text_model` in `core/litert_lm_builder.py`,
`AsrModel.build_speech2text_metadata` in `core/speech/asr_model.py`, `ParakeetTDT.get_speech2text_metadata`
and `_FlattenedJointNet` in `model_ext/parakeet/parakeet_tdt.py`, the defaults in
`exportable_module_config.py`. Values for nvidia/parakeet-tdt-0.6b-v3 with `input_sec` 5 and
`stateful_after` 4 (the litert-community file's window).

| PR #1426 | its value here | this route | same? |
|---|---|---|---|
| `builder.add_system_metadata(Authors = 'ODML')` | `ODML` | `Authors` = `""` (this repo's hand-built bundles write an empty string) | no; the runtime does not read it |
| `uuid`, `creation_timestamp` (builder: `uuid4()`, `now()`) | random | uuid5 of `<repo>@<commit>#asr-int8#<toolchain id>`, the commit's date, so a rebuild gives the same bytes | no; header only |
| `builder.add_speech2text_metadata(<speech2text_metadata.pb>)` | section 0 | the same call, litert-lm-builder 0.19.0 | yes |
| `asr_model_type={'parakeet': {}}` (`ParakeetTDT`) | `parakeet` | `parakeet` | yes |
| `input_milliseconds = round(input_sec * 1000)` | 5000 | 5000 | yes |
| `log_mel_spectrogram_config.n_frames = round(input_sec * mel_frames_per_second)`, `mel_frames_per_second=100` | 500 | 500 | yes |
| `decode_statefully_after = stateful_after` when > 0 | 4 | 4 | yes |
| `**kwargs` (`sample_rate_hz`, `decode_stop_token_id`, ...) | none passed by `ParakeetTDT` | none | yes |
| `decode_start_token_ids` | not written (the runtime's Parakeet default is 8192) | not written while the blank is 8192; written as `[blank]` otherwise | yes for every checkpoint the route has converted |
| tokenizer: `.json` → `add_hf_tokenizer`, else `add_sentencepiece_tokenizer` | `tokenizer.json` | the repo's `tokenizer.json` → `add_hf_tokenizer` | yes; litert-community's section equals nvidia/parakeet-tdt-0.6b-v3's `tokenizer.json` at `541d1f99` byte for byte |
| model type `TF_LITE_ENCODER_DECODER` when the model has > 1 exportable, else `TF_LITE_AUDIO_ENCODER` | encoder_decoder | encoder_decoder | yes |
| `.tflite` = `export_asr_models`: `encode` + `decode` (+ `decode_1` when `stateful_after` > 1) from NeMo's `ParakeetTDT` | the signatures above | the same signatures, shapes, dtypes and log-softmax, from transformers' `ParakeetForTDT` (no NeMo) | contract yes, graph source no |
| `_FlattenedJointNet`: the joint's linear on [B, T·U, H] | flattened | flattened the same way | yes |
| decode tokens `num_tokens = ceil(input_sec·8/32)·32`, cut to `stateful_after` | 4 | 4 | yes |
| LayerNorm as the `odml.group_norm` composite (`AsrLayerNorm`, "for GPU inference") | 120 composites in `encode` | plain LayerNorm ops (MEAN, SQUARED_DIFFERENCE, RSQRT, ...) | no; same arithmetic on the CPU, GPU form not built |
| `quantization_recipe` | litert-community's i8 file: int8 weights on every FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D, the 24 BATCH_MATMUL constants and the embedding | `dynamic_wi8_afp32` through litert-torch's `quantize_model`: int8 on the same op classes, the same counts (193 / 50 / 27 / 24 in `encode`) | yes by the read-back; the recipe name of the litert-community file is not stated anywhere we read |
| output `model.litertlm` | | `<repo name>_int8.litertlm` | no; MediaPipe chose the model from the content (a copy named `model.litertlm` transcribed the same), `omni_runner` from the name or `--model_name` |

## Which existing code it calls

- `scripts/convert.py`: the Hub fetch, the gated / remote-code / pre-quantized refusals, the download and
  its per-file size and hash check against the Hub tree, the toolchain record, the section identity of the
  output, the lock and the one-line `FAIL <stage>` failures. `scripts/asr_route.py` is the route.
- `parakeet_redux_work/scripts/r2_graphs.py`: the LSTM written as explicit gates (torch's LSTMCell
  arithmetic), here unrolled over the `decode` token slots.
- The parakeet-redux / parakeet-ultra entries of REPRODUCE.md: the transformers `ParakeetForTDT` reference
  without NeMo; the float16 LayerNorm overflow they found applies to a GPU form of this graph, which this
  route does not build.
- `nemotron_asr_stream_work/`: not called (see the table of accepted types).

## Measured (Mac M4 Max, 2026-11-03)

nvidia/parakeet-tdt-0.6b-v3 at `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`:

- export 44–56 s (`encode` / `decode` / `decode_1` conversion and int8); float32 `.tflite` 2,421,212,992 B,
  int8 613,589,776 B (litert-community's i8 `.tflite`: 614,261,072 B). The whole `--smoke` run took 63 s
  with the files cached, peak resident set 17.2 GB (`/usr/bin/time -l`).
- the bundle: 613,917,456 B, sha256 `06547994cf355b33e3cc6a7e34c0f4152bbc668b1908072f2e4897777da7b51e`, the
  same bytes from every run (the smoke, the 73-clip gate, and `scripts/verify_lock.py` on the gate's lock:
  MATCH, rebuilt in 56 s).
- PyTorch checks on random mel input: float32 `encode` within 9.9e-7 of the PyTorch module, `decode`
  log-probabilities within 5.3e-4 of the checkpoint's own `nn.LSTM` + joint network.
- the smoke clip, LibriSpeech dev-clean 1272-128104-0000 (5.86 s) from
  `hf-internal-testing/librispeech_asr_dummy` at `5be91486`:

  > reference: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
  >
  > this route's file: mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.

  0 word errors of 17; litert-community's file gives the same text.
- all 73 clips of that set, each through a fresh `AudioTranscriber` (CPU), one call with
  `end_of_stream=True`:

  | file | word error rate | same text as litert-community's file |
  |---|---:|---:|
  | litert-community `parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm` (`1f78e5eb`) | 0.0609 | — |
  | this route's `parakeet-tdt-0.6b-v3_int8.litertlm` | 0.0626 | 64 of 73 |

  The other 9 differ in punctuation and in 1–2 words. Both files lose words where the runtime's 5 s windows
  meet ("after all" / "after And", a clip that ends mid-phrase); that is the runtime's windowing, the same for
  both. The route's own 73-clip gate run gave the same 0.0626.

moondream/parakeet-ultra at `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` (a finetune of the same
architecture with no `processor_config.json`, so the runtime's default front end): export 55 s, the bundle
613,917,456 B (sha256 `91e7e0589564a46e9fcd1ee058c6a9f38380ea6c1ace527f2648f7dc7d07246f`), the smoke clip

  > Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his God. gospel.

  1 word error of 17 (an extra "God." before "gospel."; the clip spans two of the runtime's 5 s windows),
  WER 0.0588: passes the 0.2 bar.

## Not done yet

- A GPU form (the LayerNorm composites, the float16 LayerNorm rescale of the parakeet-ultra entry) and any
  speed comparison against litert-community's file.
- The PyPI package (`uvx hf-to-litertlm`) carries the route from 0.1.2 on; 0.1.1 does not. A wheel
  built locally from `45e3905` (`pypi/build.py`'s payload) ran `hf-to-litertlm nvidia/parakeet-tdt-0.6b-v3
  --revision 541d1f99… --smoke` to the same bundle bytes (`06547994…`, 0 word errors) and `hf-to-litertlm
  verify` on its lock to MATCH. Its first run installs two environments: the converter's (`uv.lock`) and
  `asr/` (25 packages: mediapipe 1.2.0 brings jax, jaxlib, opencv-contrib-python and matplotlib).
- The phone runs (MediaPipe on Android / iOS, `omni_runner` on Android).

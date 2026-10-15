# EmbeddingGemma 2 Transcribe: the LiteRT files

These scripts build the four `.tflite` files of [mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT): EmbeddingGemma 2's audio encoder for 30 s and 10 s windows, and the reader that turns the encoder's tokens into words, for 750 and 250 tokens. The Android app that runs them is [eg2-transcribe](https://github.com/john-rocky/eg2-transcribe) (`android/`).

| Script | What it does |
|---|---|
| `export_encoder.py kmap` | the rescale exponent k of every norm site, from the float32 ranges over 30 probe windows (writes a k map; `kmap.json` here is the one the published files used) |
| `export_encoder.py export` | loads the model inside litert-torch's Gemma 4 patch, rewrites every RMSNorm into an fp16-safe form that is exact in float32 ([docs/fp16-safe-norms.md](../../docs/fp16-safe-norms.md)), checks the rewrite bit for bit, converts one file per window and stores the fully connected and convolution weights as float16 |
| `export_encoder.py parity` | each encoder file on the CPU against the float32 PyTorch module, on the 20 test utterances |
| `export_reader.py export` | the reader, its 200,000-word table and the argmax as one graph per window length, the table stored as float16 |
| `export_reader.py check` | each reader file on the CPU: its words against the PyTorch reader and against the reader's release notes |
| `common.py` | fixtures, windows, cosine, sha256 |

## Inputs

- The model: `google/embeddinggemma-2` at revision `914f7f89142e33e77833254d9c9b90c3cef7303b` (`model.safetensors` sha256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`).
- The reader: [mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader) at revision `1af9c5f52991e90afe0ff5f792a135d6500bf3b3` (`reader.pt`, `reader_config.json`, `vocab.txt`, `word_table.npy`, `RELEASE.md`; `SHA256SUMS` lists them).
- The reader's model code and the test fixtures: [eg2-transcribe](https://github.com/john-rocky/eg2-transcribe) at commit `40ce30725e9a0fbbbe8f8507f3fd98e335c73f80`, `reader/train/` and `reader/fixtures/`. The fixtures' arrays (20 LibriSpeech test-clean utterances and a 54-minute LibriVox recording, with the log-mel and the reference soft tokens of each) are not in git: rebuild them as `reader/fixtures/README.md` "Rebuild" says. `MANIFEST.json` gives every file's sha256, and the scripts stop when an array differs from it.

## Environment

The published files were built, and rebuilt on 2026-10-15, in one environment: Python 3.12.13, torch 2.12.0, transformers 5.19.0, litert-torch 0.10.0 installed editable from [google-ai-edge/litert-torch](https://github.com/google-ai-edge/litert-torch) main at `0a81bf6f3243a6f7c846d05d62afeff4008f2542`, litert-converter 0.5.0, ai-edge-quantizer-nightly 0.10.0.dev20261013, ai-edge-litert-nightly 2.4.0.dev20261011, numpy 2.5.4, on an Apple M4 Max (macOS 27.0). litert-torch 0.9.4, the current release, has no Gemma 4 audio classes in `model_ext/gemma4/patch.py`, so main is needed. These install lines give those versions; the rebuild below ran in an environment that already had them, not in one made with these lines:

```sh
python3.12 -m venv .venv
git clone https://github.com/google-ai-edge/litert-torch && git -C litert-torch checkout 0a81bf6f3243a6f7c846d05d62afeff4008f2542
.venv/bin/pip install -e ./litert-torch "torch==2.12.0" "transformers==5.19.0" "ai-edge-quantizer-nightly==0.10.0.dev20261013" "ai-edge-litert-nightly==2.4.0.dev20261011" huggingface_hub
```

## Commands

```sh
git clone https://github.com/john-rocky/eg2-transcribe && git -C eg2-transcribe checkout 40ce30725e9a0fbbbe8f8507f3fd98e335c73f80
# rebuild eg2-transcribe/reader/fixtures as its README says ("Rebuild"), then:
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --local-dir hf/embeddinggemma-2
hf download mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader --revision 1af9c5f52991e90afe0ff5f792a135d6500bf3b3 --local-dir hf/reader
FIX=eg2-transcribe/reader/fixtures
python scripts/eg2_transcribe/export_encoder.py kmap   --model hf/embeddinggemma-2 --fixtures $FIX --out out/eg2/kmap_rerun.json   # optional: recomputes kmap.json
python scripts/eg2_transcribe/export_encoder.py export --model hf/embeddinggemma-2 --fixtures $FIX --out out/eg2
python scripts/eg2_transcribe/export_encoder.py parity --fixtures $FIX --out out/eg2
python scripts/eg2_transcribe/export_reader.py export --reader-dir hf/reader --reader-code eg2-transcribe/reader/train --fixtures $FIX --out out/eg2
python scripts/eg2_transcribe/export_reader.py check  --reader-dir hf/reader --reader-code eg2-transcribe/reader/train --fixtures $FIX --out out/eg2
```

Each step loads the model in its own process (the Gemma 4 patch is a context manager, and only a load inside it builds the patched audio classes). Run the files as scripts: the converted graphs' tensor names carry the Python path of each module class (`export_nobool.EG2AudioEncoderNoBool/…`, `__main__.F16SafeRMSNorm_…`, `__main__.build_graph_class.<locals>.ReaderGraph/…`), and the published files have those names. Renamed classes give other bytes, not other numbers.

## Checks and the rebuild of 2026-10-15

- `export`: the rewritten module against the unrewritten one in float32 on every window (the 20 utterances and 10 windows of the long recording): bit-equal, or nothing is converted.
- `parity`: valid tokens of the 20 utterances, each cut into the file's windows, against the float32 module: mean cosine ≥ 0.999 and minimum ≥ 0.99 per file.
- `check`: the reader files' word or blank per token equal to the PyTorch reader's, and every whole utterance's words equal to the reader's release notes (30 s: 20 utterances; 10 s: the ones of 250 tokens or fewer).

The rebuild of 2026-10-15 ran the commands above from a copy of this directory, on the inputs and in the environment above. All four files came out byte for byte equal to the published ones (the model repository's `FILES.sha256` and the sha256 of its LFS files):

| File | Bytes | SHA-256 | Check on the CPU (LiteRT Interpreter, XNNPACK, 4 threads) |
|---|---:|---|---|
| `eg2_audio_f16safe_fp16w_w30.tflite` | 589,273,696 | `f0f6c37ee62b51372ab28013617aad53793a84aaddd96e1e844d0cea0eb2f11d` | 5,036 tokens: cosine mean 0.9999999999983, minimum 0.9999999991 |
| `eg2_audio_f16safe_fp16w_w10.tflite` | 589,015,376 | `1405986b353c3152e4a677d9446a5694a0f8fd0b13b794d26c892ef85295188c` | 5,036 tokens: cosine mean 0.9999999999968, minimum 0.9999999937 |
| `reader_w30.tflite` | 361,494,720 | `76bdcce94e7311ee36529727e4bfd94675fe225c9d361c5628e98f5c432742e8` | the PyTorch reader's word or blank on 5,036 of 5,036 tokens; 20 of 20 transcripts equal to the release notes |
| `reader_w10.tflite` | 361,238,720 | `9419000ac89aa3f34478906dda81523f48bf0e8c802929df3345ba0850b72d3f` | 4,426 of 4,426 tokens (at most 250 per utterance); 11 of 11 transcripts |

`kmap` gave the k of `kmap.json` on all 111 sites. The rewritten module was bit-equal to the unrewritten one on 30 windows of 30 s and 39 windows of 10 s, and its soft tokens differ from the fixtures' reference soft tokens by 7.3e-06 at most. The float32 files before the weight cast (`eg2_audio_f16safe_w30.tflite`, `eg2_audio_f16safe_w10.tflite`) pass the same parity check.

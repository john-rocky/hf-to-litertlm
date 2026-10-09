# d1-omni-600M (Liquid AI) → LiteRT: decision graphs at six lengths, a vision tower and audio encoders for a model that answers with probabilities and never generates text

Published: [litert-community/d1-omni-600M-LiteRT](https://huggingface.co/litert-community/d1-omni-600M-LiteRT), release of 2026-10-09. The repository holds twelve raw LiteRT graphs (`.tflite`): six decision graphs for rows of up to 128, 256, 512, 1,024, 2,048 and 4,096 positions, the vision tower, the projector, and four audio encoders for clips of up to 5, 10, 20 and 30 s. It also holds `contract.json`, `tokenizer.json`, the Python host (`host/`), the public check set (`fixtures/`), `LICENSE`, `NOTICE` and the model card `README.md`: 43 files, 6,503,130,609 bytes. This folder holds byte copies of the host, the check set and `NOTICE`, the conversion and check scripts, an example, the measurement app behind the phone numbers, and the reproduction guide.

## What the model is

d1-omni-600M is a decision model by Liquid AI (`LiquidAI/d1-omni-600M`, revision `414f8d6438174f5b2133a9c21a478fc42625e308`, LFM Open License v1.0). It reads a state and typed questions about it: noul (yes or no), choice and score. For each question it returns a probability per option; for text requests, the checkpoint's temperatures calibrate them. It never generates text. Images and audio enter as prefix rows ahead of the text: the vision tower and the projector turn each image crop into rows, and the audio encoder turns one 16 kHz mono clip into rows.

The decision graph holds the bidirectional trunk and the decision head. Its signature `decide_<L>` takes `ids` int32 [1, L] (bos, state, question, options and the decide token, after the media rows), `prefix` float32 [1, L, 1024] (the media rows), `media`, `pad` and `keep_right` float32 [1, L], and `qtype_onehot` float32 [1, 3]. It returns `scores` float32 [1, L]. The host reads the scores at the option markers, divides them by the temperature for a text request, takes the softmax, and orders a noul answer as [yes, no]. Each question is one call. A graph computes all L positions whatever the row's length, so the host picks the next bucket at or above the row's length: P media rows plus n token ids.

The vision tower takes one crop as up to 1,024 patches of 16 × 16 pixels (`pixels` [1, 1024, 768]), the 16 × 16 position table resized by the host to the crop's patch grid (`pos` [1, 1024, 768]) and a `mask` [1, 1024]. It returns `features` [1, 1024, 768]. The host unshuffles them into `soft` [1, 256, 3072] for the projector, which returns `prefix` [1, 256, 1024]. The host tiles a large image into up to 10 tiles of 512 px plus a thumbnail, one tower call per crop. An audio encoder `audio_<T>` takes the host's log-mel (`mel` [1, 128, T]) and four validity masks and returns `prefix` [1, T3, 1024]. T501, T1001, T2001 and T3001 hold clips of up to 5, 10, 20 and 30 s, which give up to 63, 125, 250 and 375 prefix rows. `contract.json` gives every input, bucket rule and host step.

## What the repository ships

| File | What it takes | Bytes |
|---|---|---:|
| `d1-omni-600M_decide_L{128,256,512,1024,2048,4096}_fp16.tflite` | one question's row of up to L positions | 896,250,176 to 898,281,808 |
| `d1-omni-600M_vision_tower_fp16.tflite` | one image crop of up to 1,024 patches | 171,563,424 |
| `d1-omni-600M_projector_fp16.tflite` | one crop's tower features after the host's pixel unshuffle | 16,791,504 |
| `d1-omni-600M_audio_T{501,1001,2001,3001}_fp16.tflite` | one clip of up to 5, 10, 20 or 30 s as log-mel frames | 221,138,784 to 242,933,600 |
| `contract.json` | | 66,567 |
| `tokenizer.json` | | 4,733,371 |

`contract.json` lists every file with its SHA-256, every signature, bucket rule, token id and temperature, the host steps, the GPU precision of each graph, the limits and the measured checks. `tokenizer.json` is the provider's file, unchanged. `host/` is the Python host: numpy, tokenizers, Pillow, soundfile and ai-edge-litert 2.2.0, with no PyTorch. `fixtures/` is the public check set with the provider's float32 answers, its photos and speech clips, and three licence files. `LICENSE` is the LFM Open License v1.0, unchanged, and `NOTICE` lists the changes.

Every graph stores its fully connected weights in fp16. The decision graphs rewrite 36 of their 50 norms, the vision tower 13 of its 25 LayerNorms and each audio encoder 31 of its 87 LayerNorms in an fp16-safe form: the norm's input is multiplied by 2^-k and its epsilon by 4^-k. In float32 the rewrite changes no bit. Without it, the decision graphs gave uniform probabilities on every row under float16 storage: on Metal at the default precision, and on the Galaxy S26 GPU at FP16_WITH_FP32_ACCUM.

Limits. A row of more than 4,096 positions raises an error in the host; `truncate_state=True` cuts the state with the provider's own rule instead (the provider's code reads up to 16,384). A request carries images or one audio clip, not both, and a clip longer than 30 s is cut, as the provider's code does. On the 12 GB Galaxy S26, compiling the L4096 graph for the GPU took the free memory from 7.30 GB to 1.64 GB, so L4096 runs on the CPU there. Do not set `GpuOptions(constant_tensor_sharing=True)`: with the float32 embedding table, the Metal delegate aborts the process at compile.

## What this folder holds

| File | Content |
|---|---|
| `REPRODUCE.md` | The reproduction guide: the sources with SHA-256, the environments, the 12 numbered steps, the pass criterion, the results of the shipped files on the Mac and on the Galaxy S26, and where each number comes from |
| `conversion/` | The scripts that made and checked the graphs, the host files and the check set, and timed them on the Mac and on the phone; `conversion/README.md` gives the work-directory layout, the three Python environments and the commands in order |
| `host/` | A byte copy of the repository's `host/`: `d1_omni.py` (the `D1Omni` class), `d1_prompt.py` (the provider's `prompt.py` under a header comment), `d1_host.py`, `d1_vision_host.py`, `d1_audio_host.py`, `tests.py`, `verify.py`, `requirements.txt` and `vision_position_table.npy` |
| `fixtures/` | A byte copy of the repository's public check set: `public_text.json` (242 records, 273 questions), `public_image.json` (5, 12) and `public_audio.json` (6, 18) with the provider's float32 answers; `media/` (5 photos, 6 speech clips); `LICENSE-SemIf-MIT.txt`, `LICENSE-MMLU-MIT.txt` and `LICENSE-kev-Apache-2.0.txt` |
| `examples/run_example.py`, `examples/run_example.expected.json` | A text request with two questions, a photo and a voice note, the three kinds of request in the provider's quick-start code, sent through the host; and their expected CPU responses |
| `android/d1omni_gate/` | The measurement app behind the Galaxy S26 GPU and CPU runs (Kotlin CompiledModel API, LiteRT 2.2.0 from Maven). It is not a sample app; `android/README.md` gives its build and run options |
| `NOTICE` | A byte copy of the repository's attribution and list of changes |

## How to reproduce

[`conversion/README.md`](conversion/README.md) gives the work-directory layout, the three Python environments and the commands in order. [`REPRODUCE.md`](REPRODUCE.md) adds the sources with SHA-256, the pass criterion, the results of the shipped files and the result file behind each number. The reference is the provider's own float32 code on the CPU; every check compares post-temperature probabilities with it.

The phone runs used the app in `android/d1omni_gate/` on a Galaxy S26. The NPU runs and the timed GPU pairs used a C-API runner on LiteRT 2.2.0, which is not included. The NPU runtime libraries are not included.

## How to run the published files

From this folder, with Python 3.12:

```bash
hf download litert-community/d1-omni-600M-LiteRT --local-dir d1-omni-600M-LiteRT
pip install -r host/requirements.txt
python examples/run_example.py --repo d1-omni-600M-LiteRT --check
python host/verify.py --repo d1-omni-600M-LiteRT --check-files
python host/verify.py --repo d1-omni-600M-LiteRT --gpu
```

The download is 6.5 GB. `host/` here is the same as the repository's, so either copy of `requirements.txt` works.

`examples/run_example.py` sends the three requests through the repository's host on the CPU and prints the responses. `--check` compares them with `examples/run_example.expected.json`: the same answers and usage counts, and every probability within 1e-4. `--gpu` runs the example on the GPU, where `--check` allows 0.02: the audio encoder runs at the GPU's default precision, which moved the voice note's probability by 1.3e-3 on an Apple M4 Max (the text and photo answers stayed within 1e-5 of the CPU's).

`host/verify.py` runs the 303 public questions through `D1Omni` and compares them with the provider's float32 answers. It exits with 0 when every mode passes the bar of `REPRODUCE.md`. `--check-files` also checks the SHA-256 of every file that `contract.json` lists. `--gpu` runs each graph at its precision from `contract.json`: fp32 (`GpuOptions(enforce_f32=True)`) for the decision graphs, the vision tower and the projector, the default precision for the audio encoders. On the Mac (Apple M4 Max, ai-edge-litert 2.2.0) both runs pass: text max |Δp| 0.00191 on the CPU and 0.00192 on Metal, image 0.00324 and 0.00331, audio 0.00154 and 0.00318. The CPU run took 44 s, the Metal run 25 s.

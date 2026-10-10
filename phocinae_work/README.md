# Phocinae-Largha-150M-v1 (Phocinae) → LiteRT: a typed-decision model in the laya contract, with its Galaxy S26 GPU and NPU numbers

Published: [litert-community/Phocinae-Largha-150M-LiteRT](https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT) (pending the upload).

## What the model is

`Phocinae/Phocinae-Largha-150M-v1` (v1.1, Apache-2.0) is a decision model by the Phocinae project: it reads a text (the state) and typed questions about it (noul yes/no, choice, score) and returns calibrated probabilities from one forward pass per question, generating no text. It is the laya architecture on `jhu-clsp/mmBERT-small` (hidden 384, 22 layers, 6 heads, GeGLU, RoPE, a ±64-token sliding window on two of every three layers, a 256k tokenizer) with a two-layer head, a scorer read at the option markers and an act head; 144.3 M parameters. The publisher's runtime is `phocinae-server` (pure PyTorch), which is the reference of every check here.

## What this folder holds

| File | Content |
|---|---|
| `REPRODUCE.md` | Sources with their pins, the environment, the commands in order |
| `conversion/` | The fixture builder (the public test split, the author's flip protocol), the reference runner (the publisher's Engine), the graph (`graph.py`: encoder + head written out, additive masks, the exact fp16 LayerNorm rewrite), the exports, the storage variants, the Mac gates and timing, the phone scripts |
| `host/phocinae_litert.py` | Python host: request → rows → graphs → answers (tokenizers + numpy + ai-edge-litert) |
| `examples/` | An invented request, the runner, the expected response |
| `android/gate/` | The debug gate activity (LiteRT CompiledModel Kotlin API, the same I/O as hfmodels-android's laya family) that produced the phone numbers; the Qualcomm NPU libraries are not included (they ship inside an application) |

## What was found

- **The sequence and option rendering of the author's eval harness reproduce the card's numbers to the count** before any conversion: agreement with gold 0.9055 on the 2,000 public test decisions (the author's own re-run) and the flip400 counts 13 / 11 / 9 / 6. The fixture rows are therefore the rows behind the card.
- **One graph serves every backend once the LayerNorm inputs of encoder layers 12–21 and the final norm are computed on x·2⁻⁶ with eps·2⁻¹²** (their squares reach 2.0e7, over float16's 65,504). The rewrite is bit-identical in fp32 and makes Metal's default precision, the Galaxy S26's FP16 and FP16_WITH_FP32_ACCUM and the Qualcomm HTP pass the parity bar.
- **Two table forms**: an int8 EMBEDDING_LOOKUP inside the graph (one file, max |Δp| 0.0066–0.0107 against the reference; the int8 table is the whole residual) and the token table as float16 rows looked up on the host (`inputs_embeds`; the publisher's exact values, 3.8e-4–8.0e-4). The GPU delegate refuses a float16 table inside the graph; dynamic int8 FULLY_CONNECTED fails the bar (0.0835).
- **Galaxy S26 (LiteRT 2.2.0), one question's row, rested phone, median of 20 after 5 warm-ups**: GPU FP16 15.3 / 22.4 / 30.7 ms, NPU (HTP, JIT) 16.8 / 19.7 / 26.9 ms, GPU FP32 26.9 / 42.1 / 62.7 ms, CPU 4 threads 39.0 / 63.0 / 96.6 ms at windows 256 / 384 / 512; every form passes the bar on every row of the public test split. A 2,000-row leg heats the phone (the GPU clock fell from 1,300 to 461 MHz), so the timing came from short legs after a rest.
- **The host-lookup form runs whole on the HTP** (one partition; compile 3.9 / 11 / 43 s cold, 0.2 s cached); the int8-table form leaves its lookup on the CPU (two partitions).

Numbers and their files: the Hub card and `conversion/` results in the Hub repository.

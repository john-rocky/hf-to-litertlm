# Gemma 4 E2B finetune → `.litertlm`

These tools turn a merged Gemma 4 E2B finetune (the 16-bit folder that Unsloth's `model.save_pretrained_merged(...)` writes) into a `.litertlm` file for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), then load the file on the CPU and ask it one question. They were written for LiteRT-LM cells in Unsloth's Gemma 4 E2B text notebook (`Gemma4_(E2B)-Text.ipynb`), which fetch them from this folder at a fixed commit and check each file's sha256 before running it.

| File | What it does |
|---|---|
| `requirements.txt` | Pins every package of the export environment (Linux x86-64; the pinned packages have wheels for Python 3.12 and 3.13). |
| `gemma4_to_litertlm.py` | `versions` compares the installed packages with `requirements.txt`; `export` writes the `.litertlm` file; `fix` runs `bf16to32.py` and checks its output; `ask` loads the file on the CPU and answers one prompt greedily. |
| `bf16to32.py` | Rewrites the BFLOAT16 tensors the export leaves in the graph as FLOAT32, because LiteRT-LM's GPU delegate does not take BFLOAT16 tensors, and checks the result against its input. |

## How the notebook runs them

The export runs on the CPU in a venv of its own, so the notebook's training environment stays as it is. In order, the cells:

1. download the three files from `https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/<commit>/notebooks/unsloth_gemma4_e2b/` and check their sha256;
2. make a venv and install torch from the PyTorch CPU index, then `requirements.txt`, wheels only (`pip install --only-binary=:all:`), and check them with `gemma4_to_litertlm.py versions requirements.txt`;
3. `gemma4_to_litertlm.py export gemma-4-finetune <work>/model_bf16.litertlm --work <work>`;
4. `gemma4_to_litertlm.py fix <work>/model_bf16.litertlm gemma-4-finetune.litertlm`;
5. `gemma4_to_litertlm.py ask gemma-4-finetune.litertlm "What is 17 + 25?"`.

## What `export` does

- It stops unless the folder is Gemma 4 E2B (`model_type` `gemma4`, hidden size 1536, 35 layers) with `.safetensors` weights and a chat template.
- It checks the chat template. LiteRT-LM 0.18 renders the template with minijinja 2.14, gives it every message content as a list of text parts, and gives its maps no Python methods, so a template written for transformers can stop on `x.get('key')` or print the list where the text belongs. The export renders the template as LiteRT-LM renders it and as transformers renders it, on five short conversations (a user message; a system and a user message; user, assistant, user; `enable_thinking` true and false — the last two only when the template reads `enable_thinking`), and edits it only as far as needed for the two renders to match: `x.get('key')` becomes `x['key']`, and if that is not enough, every `message` / `messages[0]` content read goes through a `format_content` macro taken from LiteRT-LM's Qwen 3 chat template (commit a8178d7d; it takes a string or a list of parts). It stops with a message if they still differ. The template that Unsloth's `get_chat_template(tokenizer, chat_template = "gemma-4")` sets needs both edits: with the first one only, minijinja renders the system message's content as `[{"type": "text", "text": ...}]`.
- It runs litert-torch's Hugging Face export: text only, int4 weights (blocks of 128; the embedding tables int4 per channel), prefill signatures for 1024 and 128 tokens, a 2048-token KV cache, the embedders as sections of their own, and the checkpoint read in bfloat16 straight from the safetensors (`use_v2`, `delete_in_memory_params`, `model_dtype="bfloat16"`), which keeps the export's memory down.
- It casts the per-layer embedder's output to float32, as litert-torch already does for the main embedder: in a bfloat16 export it would stay bfloat16, and LiteRT-LM refuses that file ("The output tensor from the Embedding model must be of type float32.").

## Versions

torch 2.13.0 (CPU), litert-torch 0.10.0, ai-edge-quantizer 0.10.0, litert-converter 0.5.0, litert-lm-builder 0.18.0, LiteRT-LM 0.18.0, transformers 5.14.1, minijinja 2.14.0 and tf-nightly 2.23.0.dev20261014 (litert-torch's V2 converter calls `tensorflow.lite.python.convert.convert_mlir_bytecode`, and litert-torch does not list TensorFlow among its requirements); `requirements.txt` has the rest.

## Tested

- 2026-10-21, a Linux x86-64 Docker container (emulated on an Apple silicon Mac) with Python 3.12.11, on a merged Gemma 4 E2B finetune (a LoRA merged into `google/gemma-4-E2B-it`, with that model's chat template): `export` took 302 s with a peak RSS of 30.2 GB under a 30 GiB memory limit, and 284 s with 21.5 GB under a 20 GiB limit; `fix` took 13 s; `ask` answered "17 + 25 = **42**". That run used these tools before the chat template check and its `minijinja` pin were added, so the pinned set as published has not been installed in a run yet; on that template the check writes the template the run used, byte for byte.
- A file with the same content on a Pixel 8a with LiteRT-LM v0.18.0 (the export writes a new uuid and creation time into the file's header every time; everything else was the same, byte for byte), prefill 1024 tokens, decode 256, context 2048: decode 12.80 tok/s on the GPU and 15.09 tok/s on the CPU (median of two runs each).

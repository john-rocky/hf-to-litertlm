# Gemma 4 QAT checkpoints and litert-torch `export_hf` (checked 2026-11-10)

Google publishes the Gemma 4 QAT weights on Hugging Face as quantized checkpoints (`-qat-mobile-*`, `-qat-w4a16-ct`) and as bfloat16 checkpoints (`-qat-q4_0-unquantized`); the `-qat-q4_0-gguf` repos hold GGUF files and are not covered here. litert-torch's `export_hf`, the public exporter to `.litertlm`, does not convert the quantized ones: it raises `NotImplementedError('Quantized checkpoint is not supported yet.')` when `config.json` has a `quantization_config` ([export_lib.py L130–135](https://github.com/google-ai-edge/litert-torch/blob/13644e33d20cb18b48c7e5fbb2698fa5795740bf/litert_torch/generative/export_hf/core/export_lib.py#L130-L135), added by [litert-torch#1050](https://github.com/google-ai-edge/litert-torch/pull/1050)). That check runs only for the `text_generation` task ([L794–797](https://github.com/google-ai-edge/litert-torch/blob/13644e33d20cb18b48c7e5fbb2698fa5795740bf/litert_torch/generative/export_hf/core/export_lib.py#L794-L797)).

| `google/…` repos | `quantization_config` in `config.json` | `export_hf` |
|---|---|---|
| `gemma-4-E2B-it-qat-mobile-transformers`, `gemma-4-E4B-it-qat-mobile-transformers` | `quant_method: gemma` | E2B run below |
| `gemma-4-E2B-it-qat-mobile-ct`, `gemma-4-E4B-it-qat-mobile-ct`, `gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct` | `quant_method: compressed-tensors` | not run |
| `gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-qat-q4_0-unquantized` and their `-assistant` repos | none (bfloat16) | not stopped by the check (not run here); the `-assistant` repos were not looked at beyond `config.json` |

## Run on 2026-11-10 (Mac, Python 3.12)

litert-torch-nightly 0.11.0.dev20261109 (its `export_lib.py`, `export.py`, `model_ext/gemma4/extension.py` and `model_ext/gemma4/patch.py` are byte-identical to litert-torch main at [13644e33](https://github.com/google-ai-edge/litert-torch/commit/13644e33d20cb18b48c7e5fbb2698fa5795740bf); other files in the package differ), torch 2.14.1, transformers 5.22.2. First download [google/gemma-4-E2B-it-qat-mobile-transformers](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers) at revision `dd693ff40353f057ca5f07e945ad867f4afbf2ec` into a folder named `gemma-4-E2B-it-qat-mobile-transformers`; the commands run next to that folder.

```bash
uv venv --python 3.12 .venv-nightly
VIRTUAL_ENV=$PWD/.venv-nightly uv pip install litert-torch-nightly==0.11.0.dev20261109
export PATH=$PWD/.venv-nightly/bin:$PATH HF_HUB_OFFLINE=1

litert-torch export_hf --model=gemma-4-E2B-it-qat-mobile-transformers --output_dir=out_dryrun --dry_run=True
litert-torch export_hf --model=gemma-4-E2B-it-qat-mobile-transformers --output_dir=out_default
litert-torch export_hf --model=gemma-4-E2B-it-qat-mobile-transformers --output_dir=out_textgen --task=text_generation
```

| Command | Result |
|---|---|
| `--dry_run=True` | The plan: `task multimodal_lm` (from the model), `export_audio_encoder True`, `export_vision_encoder True`, `quantization_recipe 'dynamic_wi4b32_afp32'`. |
| no `--task` | Loads the checkpoint, then fails in the first export stage: `RuntimeError: Only Tensors of floating point and complex dtype can require gradients`, raised where the Gemma 4 patch builds the fused q/k/v `torch.nn.Linear` with the dtype of the loaded `q_proj` weight (`model_ext/gemma4/patch.py` L292). |
| `--task=text_generation` | `NotImplementedError: Quantized checkpoint is not supported yet.` from `load_model` (`export_lib.py` L795 → L135). |

litert-torch 0.10.0 from PyPI stops earlier in a fresh environment: `export_hf` imports tensorflow, which it does not install (`No module named 'tensorflow'`, [litert-torch#1345](https://github.com/google-ai-edge/litert-torch/issues/1345), fixed on main after 0.10.0); with tensorflow 2.21.0 added, the process aborts with `LLVM ERROR: inconsistency in registered CommandLine options`.

## Public threads

- [litert-torch#998, 2026-06-15](https://github.com/google-ai-edge/litert-torch/issues/998#issuecomment-4704841731), a maintainer: the prebuilt Gemma 4 `.litertlm` files are exported from variants of `google/gemma-4-E2B-it-qat-mobile-transformers`; that needs quantized safetensors conversion, which litert-torch does not support; for the time being the reply points to the post-training recipes `dynamic_wi4c_hr_afp32` and `dynamic_wi4b32_afp32`. On 2026-07-01, closing that issue: QAT safetensors support was being tracked.
- [litert-torch#1044](https://github.com/google-ai-edge/litert-torch/issues/1044) (open): the request that led to the `NotImplementedError` ([litert-torch#1050](https://github.com/google-ai-edge/litert-torch/pull/1050)); the maintainer's reply of 2026-06-15: no QAT checkpoint conversion for now.
- For the 12B, [LiteRT-LM#4185](https://github.com/google-ai-edge/LiteRT-LM/issues/4185) (open) compares `gemma-4-12B-it.litertlm` with the 12B QAT checkpoints: the file holds per-channel int4 weights, and Hugging Face has no per-channel 12B QAT checkpoint.

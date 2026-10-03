# Third-party notices

One script in this directory contains code from another project, licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). The copied code is unchanged apart from omitted docstrings; the copyright line below is the one in the source file.

| Project | Version | Source file | Copied into | What was copied | Copyright |
|---|---|---|---|---|---|
| [litert-torch](https://github.com/google-ai-edge/litert-torch) | commit 731ef0a | `litert_torch/generative/export_hf/model_ext/qwen3/qwen3_asr.py` l.106-134 | `export_bundle.py` | the body of `Qwen3AsrEncoder.forward`, as `AudioEncoderR3.encoder_body`, without the prompt-embedding concat | Copyright 2026 The LiteRT Torch Authors |

The model weights are not covered here: the converted model is distributed under the Apache License 2.0, the licence of Qwen/Qwen3-ASR-1.7B-hf.

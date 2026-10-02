## Confucius4-R2T2 (streaming speech-to-text LLM, audio-in `.litertlm`, one clip of up to 30 s per message)

`confucius4_r2t2_work/` converts NetEase Youdao's [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) (Qwen3-ASR-1.7B fine-tune: 24-layer audio encoder + 28-layer Qwen3 LM, 2.04 B parameters, 30 languages, NetEase Youdao Model Use License Agreement) into one `.litertlm` that the released LiteRT-LM runtime (`litert-lm-api` 0.17.1) drives through its **generic audio path**: the runtime frames 16 kHz PCM into 10 ms frames, the bundled encoder computes the original log-mel and the audio embeddings for a 30 s window, and the runtime splices them into the prompt at `<|audio_pad|>`. No runtime patch; the two litert-torch changes are process-local monkeypatches in the scripts. Hub repo: `<NAMESPACE>/Confucius4-R2T2`.

```sh
cd confucius4_r2t2_work
HF_HUB_DISABLE_XET=1 hf download netease-youdao/Confucius4-R2T2 --revision 185ce639118ad1362d049ca0d8ed04b6ec5cd6c9
python convert_layout.py                      # out/hf_layout: key rename to the transformers v5 -hf layout (payload byte-identical);
                                              # config / generation / processor_config from Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7
python eager_hf.py --mode full                # reference: transformers 5.14.1, fp32, the litert-torch prompt, 155 clips
PYTHONPATH=out/pyoverlay_builder017 python export_bundle_r2.py --model out/hf_layout --out out/export/c4r2_f32_30s \
  --quantization_recipe none --input_sec 30 --cache_length 1024      # fp32 export, encoder attention in the original 104-token windows
litert-lm unpack out/export/c4r2_f32_30s/model.litertlm --output-dir out/unpack/c4r2_f32_30s
python quant_sections.py --src out/unpack/c4r2_f32_30s --tag c4r2_30s --arm c4r2_30s_C_lmi8 \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack   # LM int8 dynamic + int8 embedder (+ an fp16 encoder that the next step replaces)
litert-lm unpack out/export/c4r2_30s_C_lmi8/model.litertlm --output-dir out/unpack/r3_src_C
PYTHONPATH=out/pyoverlay_builder017 python export_bundle_r3.py --window 30 --src out/unpack/r3_src_C \
  --name c4r3_30s_C_lt --prompt litert        # encoder audio [1,3000,160] -> features [1,390,2048] + mask, fp16; generic_model metadata + template; pack
python r3_render_check.py --mode expected && python r3_render_check.py --mode runtime --prompt litert \
  --bundle out/export/c4r3_30s_C_lt/model.litertlm                  # the runtime's render of the template
python r3_runtime.py --bundle out/export/c4r3_30s_C_lt/model.litertlm --clips all --prompt bundle \
  --out out/r3_rt/c4r3_30s_C_lt_full155_bundle_cpu.jsonl            # released runtime, CPU, 155 clips
python score_confucius4.py ref --hyp out/r3_rt/c4r3_30s_C_lt_full155_bundle_cpu.jsonl:text   # FLEURS en 4.43 / zh 6.40 / ja 6.21
python verify_card_snippet.py --prompt bundle                       # the card's Python block, run as written
python r3_stream.py --backend runtime --prompt litert --bundle out/export/c4r3_30s_C_lt/model.litertlm \
  --out out/r3_stream/runtime_s1_litert.jsonl                       # the vendor's streaming loop through the same API
```

Env: export, quantization and the eager reference run in one venv (python 3.14, torch 2.13, transformers 5.14.1, litert-torch 0.10.0 at 731ef0a, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0) with litert-lm-builder 0.17.0 on `PYTHONPATH` (litert-torch writes `LlmMetadata.pad_token`, which builder 0.16.1 lacks); `r3_runtime.py`, `r3_render_check.py --mode runtime`, `verify_card_snippet.py` and the streaming worker run on `litert-lm-api` 0.17.1 alone (pure Python). Fixtures: the 155-clip set of the Fun-ASR-Nano entry (FLEURS test en_us / cmn_hans_cn / ja_jp, 50 clips each, plus 5 example clips) under `$C4_FIXTURES`.

**Encoder attention windows.** Transformers attends inside 104-token (8 s) windows; litert-torch's Qwen3-ASR export patches the attention to 13-token (1 s) chunks, which cost 1.6 / 5.0 / 9.0 points (en / zh / ja, transcripts against the original on 5 s clips). `encoder_window.py` replaces that patched forward in-process with one SDPA and a constant block-diagonal mask built in numpy (so the graph gets one constant, not run-time index ops). The patched encoder equals the transformers encoder with max |Δ| 0 on 5 s and 30 s inputs.

**The encoder graph starts at raw PCM.** The generic path's built-in mel (HTK-style filterbank on the magnitude, natural log, frames not centred) is not the Qwen3-ASR / Whisper log-mel, so the bundle declares `skip_mel_spectrogram_extraction` with frame = hop = 160 samples and the encoder computes the extractor's log-mel in the graph: reflect pad + one strided conv over the windowed DFT basis (= `torch.stft(center=True)`), the extractor's Slaney filterbank, `log10`, `max − 8`, `(x + 4) / 4`. The runtime zero-fills the 30 s window after a clip, which is the input the 30 s single-window recipe was gated on. Output `features` [1, 390, 2048] plus a uint8 `mask` of ones: without a mask the runtime would take `ceil(3000 / (3000 // 390)) = 429` rows of a 390-row buffer (GENERIC_CONTRACT.md).

**Template.** The bundle renders the litert-torch prompt (`<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|><|im_start|>assistant\n`). The original repository's prompt (an empty system turn) drops FLEURS zh / ja to 10.75 / 9.33 % CER on this file (language tags `English` / `None`, Chinese numerals, one translated clip per language) and also scores worse in fp32 PyTorch (7.78 / 7.47 against 6.24 / 6.21). Text items of the user message go after `assistant\n`: that is where the original code puts a forced language (`language X<asr_text>`) and where its streaming loop puts the committed prefix.

**Quantization.** Chosen on the 155 full clips: int8 dynamic LM + int8 embedder + fp16 encoder. All-int8 translated one Mandarin clip into English (zh 9.67 %), an fp16 LM stopped after `language None` on one English clip (en 6.26 %). Mac CPU (M4 Max, 4 threads): RTF 0.124, 4.55 GB peak RSS with warm caches; LM on the Mac GPU (WebGPU over Metal, `prefill_128` and `decode` fully delegated): RTF 0.053, FLEURS 4.52 / 6.45 / 6.06. The audio encoder does not run on the GPU delegate (`BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor`).

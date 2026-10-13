## Qwen3-ASR-1.7B (speech-to-text LLM, audio-in `.litertlm`, one clip of up to 30 s per message)

`qwen3_asr17_work/` converts [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) (24-layer audio encoder + 28-layer Qwen3 LM, 2.04 B parameters, Apache 2.0) into one `.litertlm` that the released LiteRT-LM runtime (`litert-lm-api` 0.17.1) drives through its **generic audio path**: the runtime frames 16 kHz PCM into 10 ms frames, the bundled encoder computes the original log-mel and the audio embeddings for a 30 s window, and the runtime splices them into the prompt at `<|audio_pad|>`. Same recipe as the Confucius4-R2T2 entry (a fine-tune of this model with the same audio encoder), without its layout conversion. Hub repo: [litert-community/Qwen3-ASR-1.7B](https://huggingface.co/litert-community/Qwen3-ASR-1.7B).

```sh
cd qwen3_asr17_work
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-ASR-1.7B-hf --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3
python eager_hf.py --prompt litert --out out/ref_eager_litert.jsonl       # reference: transformers 5.14.1, fp32, 155 clips
python make_crops.py --samples 79999 --out_dir out/crops79999 && python encoder_window_parity.py   # 104-token windows = transformers
PYTHONPATH=out/pyoverlay_builder017 python export_fp32.py --out out/export/q17_f32_30s       # fp32 export, 30 s, KV 1024
litert-lm unpack out/export/q17_f32_30s/model.litertlm --output-dir out/unpack/q17_f32_30s
python quant_sections.py --src out/unpack/q17_f32_30s --tag q17_30s --arm q17_30s_C \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack       # LM int8 dynamic + int8 embedder
litert-lm unpack out/export/q17_30s_C/model.litertlm --output-dir out/unpack/q17_30s_C_src
PYTHONPATH=out/pyoverlay_builder017 python export_bundle.py --window 30 --src out/unpack/q17_30s_C_src \
  --name q17_30s_C_lt --prompt litert        # encoder audio [1,3000,160] -> features [1,390,2048] + mask, fp16; generic_model metadata; pack
python render_check.py --mode expected && python render_check.py --mode runtime --prompt litert \
  --bundle out/export/q17_30s_C_lt/model.litertlm                       # the runtime's render of the template
python runtime_gate.py --bundle out/export/q17_30s_C_lt/model.litertlm --clips all \
  --out out/rt/q17_30s_C_lt_full155_cpu.jsonl                          # released runtime, CPU, 155 clips
python score_fleurs.py ref --hyp out/rt/q17_30s_C_lt_full155_cpu.jsonl:text    # FLEURS en 4.43 / zh 6.19 / ja 5.73
python verify_card_snippet.py --clip zh --language Chinese             # the card's Python block, run as written
```

Env: export, quantization and the eager reference run in one venv (python 3.14, torch 2.13, transformers 5.14.1, litert-torch 0.10.0 at 731ef0a, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0) with litert-lm-builder 0.17.0 on `PYTHONPATH` (`out/pyoverlay_builder017/litert_lm_builder` → a 0.17.0 install; litert-torch writes `LlmMetadata.pad_token`, which builder 0.16.1 lacks); `runtime_gate.py`, `render_check.py --mode runtime` and `verify_card_snippet.py` run on `litert-lm-api` 0.17.1 alone (pure Python). Fixtures: the 155-clip set of the Fun-ASR-Nano entry (FLEURS test en_us / cmn_hans_cn / ja_jp, 50 clips each, plus 5 example clips) under `$C4_FIXTURES`.

**Encoder attention windows.** Transformers attends inside 104-token (8 s) windows; litert-torch's Qwen3-ASR export patches the attention to 13-token (1 s) chunks (encoder output cosine 0.81 on 5 s, 0.76 on 30 s against transformers). `encoder_window.py` replaces that patched forward in-process with one SDPA and a constant block-diagonal mask built in numpy; the patched encoder equals the transformers encoder with max |Δ| 0 on 5 s and 30 s inputs.

**The encoder graph starts at raw PCM.** The bundle declares `skip_mel_spectrogram_extraction` with frame = hop = 160 samples and the encoder computes the extractor's log-mel in the graph (reflect pad + one strided conv over the windowed DFT basis, the extractor's Slaney filterbank, `log10`, `max − 8`, `(x + 4) / 4`), then outputs `features` [1, 390, 2048] and a uint8 `mask` of ones.

**Template.** The bundle renders the litert-torch Qwen3-ASR prompt (`<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|><|im_start|>assistant\n`, no system turn) message by message and puts the user's text items after `assistant\n`, where `qwen-asr` puts a forced `language X<asr_text>`. `export_bundle.py --prompt official` bakes the checkpoint's own chat template instead (empty system turn, newlines): with the same weights it scored Mandarin / Japanese CER 6.65 / 6.06 against 6.19 / 5.73 and tagged 8 of 50 Mandarin clips `English` (fp32 PyTorch: 6.86 / 5.50 against 6.04 / 5.24, 9 of 50).

**Quantization.** int8 dynamic LM + int8 embedder + fp16 encoder (the Confucius4 choice). On the 155 clips through 0.17.1: Mac CPU en 4.43 / zh 6.19 / ja 5.73 against PyTorch fp32 with the same prompt 4.35 / 6.04 / 5.24; LM on the Mac GPU 4.52 / 6.19 / 5.32; Galaxy S26 CPU (v0.16.1 CLI, `--benchmark`, weight caches loaded) prefill 538–547 tokens/s for the 398-token prompt, decode 25.0–25.6 tokens/s.

### The 15 s file (`Qwen3-ASR-1.7B-15s.litertlm`, 2026-10-13, the speed ladder's ship form)

Same weights and recipes as the 30 s file; the encoder window is 15 s and the LM carries prefill signatures for 128 and 256 tokens. The commands below are the ones that built the shipped file (run 2026-10-13 in `qwen3_asr17_work/`, the same venv as above with litert-lm-builder 0.17.0 on `PYTHONPATH`; the lane copies of the scripts live under `speed/scripts/`):

```sh
cd qwen3_asr17_work
PYTHONPATH=out/pyoverlay_builder017 python export_fp32.py --out out/r5/f32_pb128_256 --input_sec 30 --cache_length 1024 --prefill_lengths 128,256   # fp32 LM export, KV 1024, prefill 128 and 256 (805 s wall on a shared Mac)
litert-lm unpack out/r5/f32_pb128_256/model.litertlm --output-dir out/r5/unpack/f32_pb128_256
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/quant_r5.py --src out/r5/unpack/f32_pb128_256 --tag F6_30s_pb --arm F6_C \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack            # quant_sections.py with its output root under out/r5: LM int8 dynamic + int8 embedder (the 30 s file's recipe)
litert-lm unpack out/r5/export/F6_C/model.litertlm --output-dir out/r5/unpack/F6_C_src
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/export_bundle_r3.py --window 15 --kv 1024 --enc_name enc15 --stages encoder,tflite   # the 15 s encoder: audio [1,1500,160] -> features [1,195,2048] + mask, fp16 (round 3; sha256 of enc__fp16.tflite 3f6394ff…)
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/pack_r5.py --src out/r5/unpack/F6_C_src --enc out/r3/enc/enc15/enc__fp16.tflite \
  --expect_enc_sha256 3f6394ff85db33f3c2b6560c1fefb626d3b8e2610b21667ed4f92aa462fa1385 --name F6_win15_pb128_256 --kv 1024 --prompt litert \
  --form_out speed/forms/F6_win15_pb128_256/model.litertlm                # export_bundle.pack() with the 15 s encoder: 2,701,438,880 B, sha256 fbec6d12… (the bundle's sha256 changes with each pack: the header carries a uuid and a timestamp; the sections are what to compare)
~/venvs/lt0180run/bin/python speed/scripts/runtime_gate_r3.py --bundle speed/forms/F6_win15_pb128_256/model.litertlm --clips all --backend cpu --out speed/r5/rt_F6_cpu.jsonl   # litert-lm-api 0.18.0, 155 clips
python speed/scripts/parity_r3.py --hyp speed/r5/rt_F6_cpu.jsonl --out speed/r5/parity_F6_cpu.json   # FLEURS en 4.35 / zh 6.19 / ja 5.84 (GPU fp16: 4.44 / 6.24 / 5.21)
```

Speed rows (response = audio encoder + prefill + decode, the 6.4 s English demo clip, warm): Mac M4 Max GPU 0.341 s / CPU 0.847 s (litert-lm-api 0.18.0), Galaxy S26 GPU 1.40–1.51 s / CPU 1.55 s, Pixel 8a GPU 5.03–6.33 s / CPU 5.10 s (v0.18.0 prebuilt CLI); the 30 s file on the same days 0.538 / 1.191, 2.11–2.81 / 2.11, 9.86–9.98 / 8.39 s. The full ladder with every form tried (int4 LM = parity FAIL, encoder int8 = runtime refusal unless the log-mel stays float, KV 512 = GPU parity FAIL on Japanese, prefill buckets on the 30 s window) is `qwen3_asr17_work/SPEED_LADDER.md` (rows 1–67, `speed/ROUND1–6.md`).

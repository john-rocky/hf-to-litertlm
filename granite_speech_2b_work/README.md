# granite-speech-5.0-2b → one `.litertlm` for LiteRT-LM's generic audio path

These scripts convert IBM's [granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) (Apache-2.0; a 16-layer conformer speech encoder, a 2-layer Q-Former projector and a 40-layer dense Granite language model, 2.3 B parameters) at commit `98160169` into `granite-speech-5.0-2b.litertlm`, the file of [litert-community/granite-speech-5.0-2b](https://huggingface.co/litert-community/granite-speech-5.0-2b). The released LiteRT-LM runtime (0.19.1) runs it with no change: the runtime frames 16 kHz PCM into 160-sample frames, runs the bundled audio encoder on a 36-second window, and puts its embeddings into the prompt at `<|audio|>`.

```bash
uv venv -p 3.13 .venv-gs2b
uv pip install --python .venv-gs2b/bin/python -r granite_speech_2b_work/requirements.txt
uv venv -p 3.13 .venv-gs2b-rt
uv pip install --python .venv-gs2b-rt/bin/python -r granite_speech_2b_work/requirements-runtime.txt
PY=$PWD/.venv-gs2b/bin/python RT_PY=$PWD/.venv-gs2b-rt/bin/python \
  bash granite_speech_2b_work/convert_granite_speech_2b.sh out/granite-speech-5.0-2b
```

`REF=<the published granite-speech-5.0-2b.litertlm>` adds two comparisons: every section's bytes and header items, and the transcripts of the same clips. The steps, the environments and what each step checks are in the header of `convert_granite_speech_2b.sh`; REPRODUCE.md has the expected sections.

| file | does |
|---|---|
| `convert_granite_speech_2b.sh` | the steps, in order, with a `### CMD:` log (`OUTDIR/recipe.log`) |
| `build_lm.py` | the language model as a standalone `GraniteForCausalLM` checkpoint (the export's input) |
| `collect_lm_export.py` | takes the LM export's graphs and executor metadata out of litert-torch's work directory |
| `port.py` | the audio path (front end, conformer, Q-Former, projector) as one PyTorch module |
| `export_encoder.py` | exports `port.py` to TFLite (fp32), then the fp16-weight file |
| `build_bundle.py` | the `.litertlm`: metadata, template, tokenizer, the three graphs, with checks |
| `fetch_clips.py`, `transcribe.py`, `reference_hf.py`, `compare.py` | the transcription check: six clips through the file and through the source model in transformers |
| `requirements.txt`, `requirements-runtime.txt` | the two environments (the export, and litert-lm 0.19.1) |

## The audio encoder graph (`port.py`)

The runtime hands the encoder 36 s of framed PCM, `audio` [1, 3600, 160], with the rest of the window zero, and no clip length. The graph computes what transformers computes on the unpadded clip:

- **Clip length.** n = the index of the last non-zero sample + 1. Exact digital silence at the end of a clip therefore counts as padding.
- **Front end.** torchaudio's MelSpectrogram as the processor configures it (n_fft 512, window 400, hop 160, 80 mels, centered, reflect padding, power 2), with the STFT as a matrix product over the window and torchaudio's own Hann window and mel filter bank as constants. The centre padding reflects the clip's own samples at both ends, not the window's zeros. Then log10 (floor 1e-10), the maximum over the clip's M = n // 160 + 1 frames only, a floor at maximum − 8, /4 + 1, and pairs of frames stacked into E = M // 2 rows of 160.
- **Conformer.** Block attention in 200-row blocks with the keys at or after E masked, as transformers masks its zero-padded last block; Shaw relative positions; the convolution module sees zeros after the clip end (transformers' depthwise convolution pads with zeros); the mid-layer CTC softmax added back after block 8.
- **Q-Former.** The encoder rows at or after E are set to zero before the projector, as transformers zero-pads its last 15-row window; 3 queries per window, then the linear layer 1024 → 2048.
- **Token count.** `mask` = 1 for the first ceil(E / 15) × 3 tokens (the processor's `_get_num_audio_features`); LiteRT-LM keeps the embeddings up to the last non-zero entry. The graph decides it without a division: the converter turns a division by a constant into a multiplication by its float32 reciprocal, and float32(1/15) is above 1/15, so ceil(E × float32(1/15)) is one too high when E is a multiple of 15.
- **×12.** `features` is the projector output itself. The language model's `embedding_multiplier` 12 is applied inside the exported prefill / decode graph to every input row, audio rows included, which is where transformers applies it too (after it scatters the projector output into `inputs_embeds`).

The converter names every tensor after its module path (`port.GraniteSpeechAudioGraph/port.Encoder_encoder/…`), so a renamed file, class or attribute gives a TFLite file with other bytes.

## The bundle

`generic_model` with `audio_enabled`, `delimiter_regex` `(<\|audio\|>)`, `audio_token_regex` `<\|audio\|>`, `skip_mel_spectrogram_extraction` (the front end is in the graph), frame and hop 160, `audio_input_scale` 1.0 (LiteRT-LM's defaults are 512 / 160 / 32768 when unset). No start token: the source's tokenizer has `add_bos_token` false and its BOS is `<|end_of_text|>`. Stop token 100257 `<|end_of_text|>`. The template renders the repo's `chat_template.jinja` on string content and also takes the list of content parts LiteRT-LM passes for a message with audio (`USER: <|audio|>{text}\n ASSISTANT:`). The tokenizer section holds the repo's `tokenizer.json` bytes. fp32 activations are declared on every TFLite section (`prefer_activation_type`), which LiteRT-LM's GPU executor follows when the app sets no activation type.

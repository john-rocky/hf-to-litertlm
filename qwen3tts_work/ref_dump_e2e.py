"""Full greedy E2E reference dump (qtts venv): codes, hiddens, prefill embeds, wav.

Hooks talker.generate to capture the exact prefill inputs the host pipeline must
reproduce, and uses the inner model.generate to get codes + hidden states.

~/qtts/bin/python qwen3tts_work/ref_dump_e2e.py
"""
import numpy as np
import torch
import soundfile as sf

torch.manual_seed(0)

from qwen_tts import Qwen3TTSModel

TEXT = "Hello! This is a small test of speech synthesis running on device."
LANG = "English"

wrap = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu", dtype=torch.float32,
)
model = wrap.model

# speaker embedding (x-vector only mode)
items = wrap.create_voice_clone_prompt(
    ref_audio="qwen3tts_work/assets/clone.wav", ref_text=None, x_vector_only_mode=True,
)
vc = wrap._prompt_items_to_voice_clone_prompt(items)
spk = items[0].ref_spk_embedding
print("spk_emb", spk.shape, float(spk.norm()))

input_ids = wrap._tokenize_texts([wrap._build_assistant_text(TEXT)])
print("input_ids", input_ids[0].shape, input_ids[0][0, :6].tolist(), "...", input_ids[0][0, -6:].tolist())

captured = {}
orig_gen = model.talker.generate
def hook_gen(*args, **kwargs):
    captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().numpy()
    captured["attention_mask"] = kwargs["attention_mask"].detach().numpy()
    captured["trailing_text_hidden"] = kwargs["trailing_text_hidden"].detach().numpy()
    captured["tts_pad_embed"] = kwargs["tts_pad_embed"].detach().numpy()
    return orig_gen(*args, **kwargs)
model.talker.generate = hook_gen

with torch.no_grad():
    codes_list, hiddens_list = model.generate(
        input_ids=input_ids,
        voice_clone_prompt=vc,
        languages=[LANG.lower()],
        do_sample=False, top_k=1, temperature=1.0,
        subtalker_dosample=False, subtalker_top_k=1, subtalker_temperature=1.0,
        max_new_tokens=512,
    )

codes = codes_list[0].numpy()      # [T,16]
hiddens = hiddens_list[0].numpy()  # [T,1024]
print("codes", codes.shape, "hiddens", hiddens.shape)
print("cb0 head:", codes[:8, 0].tolist())

with torch.no_grad():
    wavs, out_sr = model.speech_tokenizer.decode({"audio_codes": [torch.tensor(codes)]})
wav = np.asarray(wavs[0])
assert out_sr == 24000
sf.write("qwen3tts_work/ref/ref_e2e.wav", wav, 24000)

np.savez(
    "qwen3tts_work/ref/e2e_ref.npz",
    spk_emb=spk.detach().numpy(),
    input_ids=input_ids[0].numpy(),
    prefill_embeds=captured["inputs_embeds"],
    attention_mask=captured["attention_mask"],
    trailing_text_hidden=captured["trailing_text_hidden"],
    tts_pad_embed=captured["tts_pad_embed"],
    codes=codes,
    hiddens=hiddens,
    wav=wav,
)
print("E2E_REF_DUMPED", wav.shape)

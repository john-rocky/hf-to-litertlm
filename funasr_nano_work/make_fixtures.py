"""Build the round-1 fixtures: 16 kHz mono PCM16 wav under out/fixtures/ + fixtures/meta.json.

(a) the 5 official example mp3s (FunAudioLLM/Fun-ASR-Nano-2512 @ 272c57b8, identical bytes in the
    -vllm repo) -> ffmpeg -ar 16000 -ac 1 -sample_fmt s16 (smoke; no reference text except zh).
(b) 20 LibriSpeech dev-clean clips (granite_speech_work/fixtures/clipNN.flac, 16 kHz s16 mono
    already) -> same ffmpeg command (no resampling happens), reference text from
    granite_speech_work/fixtures/meta.json.
Run with the funasr_oracle venv (needs soundfile).
"""
import json
import os
import subprocess

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "fixtures")
EX_DIR = os.path.join(HERE, "out", "hf_official", "example")
GRANITE = os.path.join(REPO, "granite_speech_work", "fixtures")
ZH_EXPECTED = "开饭时间早上九点至下午五点。"  # -vllm repo README / MODEL_PROVENANCE.json validation.expected_text


def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst], check=True)
    info = sf.info(dst)
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16", info
    return info.frames


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for lang in ["zh", "en", "ja", "ko", "yue"]:
        fid = f"example_{lang}"
        dst = os.path.join(OUT, fid + ".wav")
        n = to_wav(os.path.join(EX_DIR, lang + ".mp3"), dst)
        row = {"id": fid, "file": f"out/fixtures/{fid}.wav", "text": None,
               "duration_s": round(n / 16000, 4), "n_samples": n, "n_samples_mod_960": n % 960,
               "lang": lang,
               "source": f"FunAudioLLM/Fun-ASR-Nano-2512@272c57b82523ada6fd87095e955f8e29100979ab example/{lang}.mp3 "
                         "(ffmpeg -ar 16000 -ac 1 -sample_fmt s16)",
               "license": "Apache-2.0 (model repo README frontmatter; the -vllm repo ships the LICENSE text)"}
        if lang == "zh":
            row["expected_text"] = ZH_EXPECTED
        meta.append(row)
    gmeta = json.load(open(os.path.join(GRANITE, "meta.json")))
    assert len(gmeta) == 20
    for i, g in enumerate(gmeta):
        fid = f"en_clip{i:02d}"
        src = os.path.join(GRANITE, f"clip{i:02d}.flac")
        assert os.path.basename(g["file"]) == f"clip{i:02d}.wav", g
        dst = os.path.join(OUT, fid + ".wav")
        n = to_wav(src, dst)
        meta.append({"id": fid, "file": f"out/fixtures/{fid}.wav", "text": g["text"],
                     "duration_s": round(n / 16000, 4), "n_samples": n, "n_samples_mod_960": n % 960,
                     "lang": "en", "source_id": g["id"],
                     "source": f"LibriSpeech dev-clean {g['id']} via hf-internal-testing/librispeech_asr_dummy "
                               f"(granite_speech_work/fixtures/clip{i:02d}.flac)",
                     "license": "CC BY 4.0"})
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    with open(os.path.join(HERE, "fixtures", "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    for r in meta:
        print(r["id"], r["n_samples"], r["duration_s"], "mod960", r["n_samples_mod_960"])


if __name__ == "__main__":
    main()

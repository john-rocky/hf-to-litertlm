"""Audio gate: ASR transcript error (whisper) + speaker similarity (TitaNet) + mel/duration stats,
oracle wav vs candidate wav for every case.  Two venvs are involved, so this script has two entry points:

  ~/venvs/ltconv040dev/bin/python3 audio8_tts_work/gate_audio.py asr <dir> [<dir>...]      -> <dir>/asr.json (openai-whisper turbo)
  ~/parakeet-env/bin/python3      audio8_tts_work/gate_audio.py spk <dir> [<dir>...]      -> <dir>/spk.json (NeMo TitaNet-L)
  python3                          audio8_tts_work/gate_audio.py report <dir> [<dir>...]  -> table (uses the two json files)
<dir> = out/oracle or out/e2e/<tag>. WER (en, word) / CER (ja, char) against the input sentence; speaker cos
against the reference clip (ref cases) and against the oracle wav of the same case (all cases).
"""
import os, sys, json, re, glob, unicodedata
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "turbo")


def load_wav16(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return a


def norm_en(t):
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def norm_ja(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\s。、．，「」『』！？!?・…—\-\.,]+", "", t)
    return list(t)


def edit_distance(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return int(d[len(b)])


def cmd_asr(dirs):
    import whisper, torch
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    model = whisper.load_model(WHISPER_MODEL, device="cpu")
    for d in dirs:
        res = {}
        for case_id, lang, ref_key, text, seed in C.cases():
            p = os.path.join(d, f"{case_id}.wav")
            if not os.path.exists(p):
                continue
            a = load_wav16(p)
            if a.size < 1600:
                res[case_id] = dict(hyp="", ref=text, err=len(norm_en(text) if lang == "en" else norm_ja(text)), n=0, dur=a.size / 16000); continue
            out = model.transcribe(a, language=lang, temperature=0.0, fp16=False, condition_on_previous_text=False)
            hyp = out["text"].strip()
            r, h = (norm_en(text), norm_en(hyp)) if lang == "en" else (norm_ja(text), norm_ja(hyp))
            res[case_id] = dict(hyp=hyp, ref=text, err=edit_distance(r, h), n=len(r), dur=a.size / 16000)
            print(f"[{os.path.basename(d)}] {case_id}: err {res[case_id]['err']}/{len(r)} | {hyp}", flush=True)
        json.dump(dict(model=WHISPER_MODEL, cases=res), open(os.path.join(d, "asr.json"), "w"), ensure_ascii=False, indent=1)


def cmd_spk(dirs):
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    m = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large", map_location="cpu").eval()

    def emb(path):
        a = load_wav16(path)
        with torch.no_grad():
            x = torch.tensor(a)[None]; L = torch.tensor([a.size])
            _, e = m.forward(input_signal=x, input_signal_length=L)
        e = e[0].numpy(); return e / (np.linalg.norm(e) + 1e-9)

    ref_emb = {k: emb(os.path.join(C.FIX, f"ref_{k}_44k.wav")) for k in C.REFS}
    oracle_emb = {}
    for case_id, lang, ref_key, text, seed in C.cases():
        p = os.path.join(C.OUT, "oracle", f"{case_id}.wav")
        if os.path.exists(p):
            oracle_emb[case_id] = emb(p)
    for d in dirs:
        res = {}
        for case_id, lang, ref_key, text, seed in C.cases():
            p = os.path.join(d, f"{case_id}.wav")
            if not os.path.exists(p) or os.path.getsize(p) < 4000:
                continue
            e = emb(p)
            res[case_id] = dict(cos_ref=float(e @ ref_emb[ref_key]) if ref_key else None,
                                cos_oracle=float(e @ oracle_emb[case_id]) if case_id in oracle_emb else None)
            print(f"[{os.path.basename(d)}] {case_id}: cos_ref {res[case_id]['cos_ref']} cos_oracle {res[case_id]['cos_oracle']:.3f}", flush=True)
        json.dump(dict(model="nvidia/speakerverification_en_titanet_large", cases=res), open(os.path.join(d, "spk.json"), "w"), indent=1)


def cmd_report(dirs):
    rows = []
    for d in dirs:
        asr = json.load(open(os.path.join(d, "asr.json")))["cases"] if os.path.exists(os.path.join(d, "asr.json")) else {}
        spk = json.load(open(os.path.join(d, "spk.json")))["cases"] if os.path.exists(os.path.join(d, "spk.json")) else {}
        for lang in ("en", "ja"):
            ids = [c[0] for c in C.cases() if c[1] == lang and c[0] in asr]
            err = sum(asr[i]["err"] for i in ids); n = sum(asr[i]["n"] for i in ids)
            cr = [spk[i]["cos_ref"] for i in ids if i in spk and spk[i]["cos_ref"] is not None]
            co = [spk[i]["cos_oracle"] for i in ids if i in spk and spk[i]["cos_oracle"] is not None]
            dur = sum(asr[i]["dur"] for i in ids)
            rows.append((os.path.relpath(d, C.OUT), lang, len(ids), f"{100*err/max(n,1):.1f}% ({err}/{n})",
                         f"{np.mean(cr):.3f}" if cr else "-", f"{np.mean(co):.3f}" if co else "-", f"{dur:.1f}s"))
    print(f"{'dir':<14} {'lang':<4} {'n':>2} {'WER/CER':>16} {'spk cos vs ref':>14} {'cos vs oracle':>14} {'audio':>7}")
    for r in rows:
        print(f"{r[0]:<14} {r[1]:<4} {r[2]:>2} {r[3]:>16} {r[4]:>14} {r[5]:>14} {r[6]:>7}")


if __name__ == "__main__":
    cmd, dirs = sys.argv[1], sys.argv[2:]
    {"asr": cmd_asr, "spk": cmd_spk, "report": cmd_report}[cmd](dirs)

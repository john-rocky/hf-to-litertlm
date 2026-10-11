"""EMA Lightning on LiteRT: the host loop that ships with the graphs. No PyTorch needed.

    pip install ai-edge-litert normalizer-tr numpy
    python ema_host.py "Merhaba, bu ses telefonunuzun üzerinde üretiliyor." --out merhaba.wav [--seed 0] [--speed 1.0]

Reads config.json next to this file (vocabulary, buckets, step times, decoder stage lengths), runs normalizer-tr and the
package's alphabet step, cuts the text into pieces, and drives the four graphs: text -> (host timeline) -> align ->
step x 4 (host Euler) -> decoder windows -> 48 kHz float32 audio. The text handling, the chunker and the frame timeline
follow the ema-lightning package (Apache-2.0, canberk7/ema-lightning); the noise comes from torch.Generator when torch
is installed (bit-for-bit the package's take for a seed) and from numpy otherwise (a different take for the same seed).
"""
import argparse, json, re, sys, time, unicodedata, wave
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
RATE, HOP, LATENT, D = CFG["sample_rate"], CFG["hop"], CFG["latent_dim"], CFG["d"]
TIMES = [float(t) for t in CFG["times"]]
NEG = -1e4
W = CFG["windows"]
LIM = CFG["limits"]

# ----------------------------------------------------------------------------------------------- text frontend
TURKISH = frozenset("çğıöşüÇĞİÖŞÜ")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                            "«": '"', "»": '"', "–": "-", "—": "-", "−": "-", "…": "..."})
UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]")
BLOCK_BYTES = 8 * 1024


class Frontend:
    """Any written Turkish -> text in the model's alphabet (normalizer-tr 'fallback' policy, Turkish lowercase)."""

    def __init__(self, vocab):
        from normalizer_tr import Normalizer
        self.vocab = frozenset(vocab)
        self.normalizer = Normalizer()

    def __call__(self, text):
        text = UNSAFE.sub(" ", text.encode("utf-8", "ignore").decode("utf-8"))
        if not text.strip():
            return ""
        return self.alphabet(" ".join(self.spoken(block) for block in blocks(text)))

    def spoken(self, text):
        try:
            return self.normalizer.normalize(text, ambiguity_policy="fallback").normalized_text
        except Exception:
            return text

    def alphabet(self, text):
        text = text.translate(TYPOGRAPHY).replace("İ", "i").replace("I", "ı").lower()
        out = []
        for ch in text:
            if ch not in TURKISH:
                ch = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(ch if ch and all(c in self.vocab for c in ch) else " ")
        return re.sub(r"\s+", " ", "".join(out)).strip()


def blocks(text):
    words, block, size = text.split(), [], 0
    for word in words:
        n = len(word.encode("utf-8")) + 1
        if block and size + n > BLOCK_BYTES:
            yield " ".join(block)
            block, size = [], 0
        block.append(word)
        size += n
    if block:
        yield " ".join(block)


# ---------------------------------------------------------------------------------------------------- chunker
LETTERS_PER_SECOND, MAX_SECONDS = 18.0, 10.0
SENTENCE_PAUSE, CLAUSE_PAUSE = 0.25, 0.12
CUTS = ((re.compile(r"[.!?]+[\"')]*(?= )"), SENTENCE_PAUSE), (re.compile(r"[,;:](?= )"), CLAUSE_PAUSE),
        (re.compile(r"\S(?= )"), CLAUSE_PAUSE))
LETTER = re.compile(r"[^\W\d_]")


def chunk(text, speed):
    """[(piece, seconds of silence after it)], each piece ending in terminal punctuation."""
    limit = int(min(LIM["letters_per_piece"], LETTERS_PER_SECOND * MAX_SECONDS * speed))
    pieces, rest = [], text.strip()
    while rest:
        cut, pause = len(rest), 0.0
        if len(rest) > limit:
            cut = limit
            for pattern, gap in CUTS:
                ends = [m.end() for m in pattern.finditer(rest, 0, limit + 1)]
                if ends:
                    cut, pause = ends[-1], gap
                    break
        piece, rest = rest[:cut].strip(), rest[cut:].strip()
        if LETTER.search(piece):
            pieces.append((piece if piece.rstrip("\"')")[-1:] in (".", "!", "?") else piece.rstrip(",;:- ") + ".", pause))
    if pieces:
        pieces[-1] = (pieces[-1][0], 0.0)
    return pieces


# ------------------------------------------------------------------------------------------- host bookkeeping
def piece_ids(text, stoi):
    ids = [stoi.get(ch, 1) for ch in text]
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return np.array(ids, np.int32), np.array(cw, np.int32), np.array(wstart, np.int32)


def timeline(dur, cw, wstart, speed=1.0):
    dur = dur.astype(np.float64) / speed
    n_words = int(cw[-1]) + 1
    counts = np.zeros(n_words)
    np.add.at(counts, cw, dur)
    counts = np.clip(np.round(counts), 1, LIM["max_word_frames"]).astype(np.int64)
    frames = min(int(counts.sum()), LIM["max_frames"])
    fw = np.repeat(np.arange(n_words), counts)[:frames]
    starts = np.cumsum(counts) - counts
    fp = ((np.arange(frames) - starts[fw]) / counts[fw]).astype(np.float32)
    c = np.maximum(dur, 1e-4).astype(np.float32)
    done = np.cumsum(c)
    before = done - c
    total = np.zeros(n_words, np.float32)
    np.add.at(total, cw, c)
    cp = np.clip((done - before[wstart] - 0.5 * c) / np.maximum(total[cw], 1e-8), 0, 1).astype(np.float32)
    wlen = np.maximum(np.bincount(cw, minlength=n_words).astype(np.float32), 1)
    woff = np.cumsum(wlen) - wlen
    return (fw.astype(np.int32), fp, (woff[cw] + cp * wlen[cw]).astype(np.float32), (woff[fw] + fp * wlen[fw]).astype(np.float32))


def text_bucket(L):
    return next(b for b in CFG["text_buckets"] if L <= b)


def bucket_for(L, T):
    for lb, tb in CFG["sound_buckets"]:
        if L <= lb and T <= tb:
            return lb, tb
    raise ValueError(f"piece too long for the buckets: L={L} T={T}")


def p_bucket(n):
    return next(p for p in CFG["decoder_buckets"] if n <= p)


def sound_inputs(h, cg, fg, cw, fw, Lb, Tb, noise):
    L, T = len(cg), len(fg)
    a = CFG["aligner"]
    H = np.zeros((1, Lb, D), np.float32); H[0, :L] = h
    CG = np.zeros((1, Lb), np.float32); CG[0, :L] = cg
    FG = np.zeros((1, Tb), np.float32); FG[0, :T] = fg
    CW = np.full(Lb, -1, np.int64); CW[:L] = cw
    FW = np.full(Tb, -1, np.int64); FW[:T] = fw
    rel = CW[None, :] - FW[:, None]
    allow = ((rel >= -a["lookback"]) & (rel <= a["lookahead"]) & (CW[None, :] >= 0) & (FW[:, None] >= 0)).astype(np.float32)[None, None]
    fbias = np.full((1, 1, 1, Tb), NEG, np.float32); fbias[..., :T] = 0.0
    N = np.zeros((1, noise.shape[0], Tb, LATENT), np.float32); N[0, :, :T] = noise
    return dict(h=H, cg=CG, fg=FG, allow=allow, fbias=fbias, noise=N)


def decoder_masks(n, P):
    full, real = CFG["decoder_stage_lengths"][str(P)], stage_lengths(n)
    out = {}
    for i, (f, r) in enumerate(zip(full, real)):
        m = np.zeros((1, 1, f), np.float32); m[..., :r] = 1.0
        out[f"m{i}"] = m
    return out


def stage_lengths(n):  # the decoder's length recurrence: rates (8,6,5,2,2,2), kernels (16,12,10,4,4,4), pads (k-r)//2
    out, length = [n], n
    for r, k in zip((8, 6, 5, 2, 2, 2), (16, 12, 10, 4, 4, 4)):
        length = (length - 1) * r - 2 * ((k - r) // 2) + k
        out.append(length)
    return out


def decode_windows(frames):
    spans, s = [], 0
    while s < frames:
        e = min(frames, s + (W["first"] if s == 0 else W["later"]))
        spans.append((max(0, s - W["context"]), min(frames, e + W["context"]), s, e))
        s = e
    return spans


def noise_for(seed, frames):
    try:
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn((len(TIMES), frames, LATENT), generator=g).numpy()
    except ImportError:
        return np.random.default_rng(seed).standard_normal((len(TIMES), frames, LATENT)).astype(np.float32)


# ------------------------------------------------------------------------------------------------- the graphs
class Graph:
    def __init__(self, path, signature, threads=4):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.sig = self.it.get_signature_runner(signature)
        self.inputs = self.sig.get_input_details()

    def __call__(self, **kw):
        return self.sig(**{k: np.ascontiguousarray(v, dtype=self.inputs[k]["dtype"]) for k, v in kw.items()})


class EmaLiteRT:
    def __init__(self, threads=4):
        self.files = {f["kind"]: f["file"] for f in CFG["files"]}
        self.frontend = Frontend(CFG["vocab"])
        self.stoi = {ch: i for i, ch in enumerate(CFG["vocab"])}
        tab = np.load(HERE / self.files["table"])
        self.table = [(float(t), tab["bc"][k:k + 1], tab["sg"][k:k + 1]) for k, t in enumerate(tab["times"])]
        self.threads, self.graphs = threads, {}

    def graph(self, kind, signature):
        key = (kind, signature)
        if key not in self.graphs:
            self.graphs[key] = Graph(HERE / self.files[kind], signature, self.threads)
        return self.graphs[key]

    def piece(self, text, seed, speed=1.0):
        t0 = time.perf_counter()
        ids, cw, wstart = piece_ids(text, self.stoi)
        L = len(ids)
        Lb = text_bucket(L)
        IDS = np.zeros((1, Lb), np.int32); IDS[0, :L] = ids
        MASK = np.zeros((1, Lb), np.float32); MASK[0, :L] = 1
        o = self.graph("text", f"ema_text_L{Lb}")(ids=IDS, mask=MASK)
        h, dur = o["output_0"][0, :L], o["output_1"][0, :L]
        fw, fp, cg, fg = timeline(dur, cw, wstart, speed)
        T = len(fw)
        Lb2, Tb = bucket_for(L, T)
        kw = sound_inputs(h, cg, fg, cw, fw, Lb2, Tb, noise_for(seed, T))
        cond = self.graph("align", f"ema_align_L{Lb2}_T{Tb}")(h=kw["h"], cg=kw["cg"], fg=kw["fg"], allow=kw["allow"])["output_0"]
        step = self.graph("step", f"ema_step_T{Tb}")
        noise = kw["noise"][0]  # [4, Tb, 64]: one draw per step
        x = noise[0:1]
        for k, (t, bc, sg) in enumerate(self.table):
            v = step(x=x, cond=cond, bc=bc, sg=sg, fbias=kw["fbias"])["output_0"]
            x1 = x + (1 - t) * v  # the package's sampler: x1 = x + (1 - t) v; x = (1 - t') noise' + t' x1
            if k + 1 < len(TIMES):
                x = (1 - TIMES[k + 1]) * noise[k + 1:k + 2] + TIMES[k + 1] * x1
        lat = x1[0, :T]
        audio, first = [], None
        for (A, B, s, e) in decode_windows(T):
            n = B - A
            P = p_bucket(n)
            Z = np.zeros((1, LATENT, P), np.float32); Z[0, :, :n] = lat[A:B].T
            y = self.graph("decoder", f"ema_decoder_P{P}")(z=Z, **decoder_masks(n, P))["output_0"][0]
            audio.append(y[(s - A) * HOP:(e - A) * HOP])
            first = first or (time.perf_counter() - t0)
        return np.concatenate(audio), dict(letters=L, frames=T, total_s=time.perf_counter() - t0, first_audio_s=first)

    def say(self, text, seed=0, speed=1.0):
        out, stats = [], []
        for i, (piece, pause) in enumerate(chunk(self.frontend(text), speed)):
            audio, st = self.piece(piece, (seed * 1_000_003 + i) % 2**63, speed)
            st["text"] = piece
            out.append(audio)
            if pause:
                out.append(np.zeros(int(pause * RATE), np.float32))
            stats.append(st)
        return np.concatenate(out) if out else np.zeros(0, np.float32), stats


def write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE); f.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    tts = EmaLiteRT(a.threads)
    audio, stats = tts.say(a.text, a.seed, a.speed)
    write_wav(a.out, audio)
    secs = len(audio) / RATE
    tot = sum(s["total_s"] for s in stats)
    print(f"{a.out}: {secs:.2f} s of audio in {tot * 1e3:.0f} ms (RTF {tot / max(secs, 1e-9):.3f}), first audio {stats[0]['first_audio_s'] * 1e3:.0f} ms; "
          f"pieces: {[s['text'] for s in stats]}")


if __name__ == "__main__":
    main()

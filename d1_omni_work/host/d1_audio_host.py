"""d1-omni audio host: one 16 kHz mono clip -> the audio graph's five inputs -> the prefix rows (numpy only).

    from d1_audio_host import prepare, prefix_rows
    x, info = prepare(samples)            # samples: int16 PCM or float in [-1, 1], 16 kHz mono, 1-D
    out = audio_graph(**x)["prefix"]      # signature audio_<T_b>: prefix float32 [1, T3, 1024]
    prefix = prefix_rows(out, info)       # [P, 1024] -> the decision graph's `prefix` input (media = 1 on 0 .. P-1)

The provider's audio.py (pinned snapshot 414f8d64), step by step:
  waveform()   int16 -> float32 / 32768, anything else -> float32; cut to 30 s; zero-padded to 8,000 samples (0.5 s)
  mel()        MelFrontend (always float32 in the provider): frames = n // 160 valid frames; preemphasis y[0] = x[0],
               y[t] = x[t] - 0.97 x[t-1]; STFT n_fft 512, hop 160, Hann 400 (periodic=False) centred in the 512-sample
               frame (56 zeros each side), center=True with 256 zeros each side -> T = n // 160 + 1 frames;
               |X|^2 as the provider computes it (sqrt(re^2 + im^2), squared); Slaney mel 128 (slaney_filterbank,
               copied verbatim); log(x + 2^-24); per mel bin mean / std over the valid frames only (std divides by
               count - 1, then + 1e-5; a NaN std counts as 0); the invalid frames (t >= frames, here only the last
               STFT frame) are 0.
  bucket       the smallest T_b in T_BUCKETS with T <= T_b (5 / 10 / 20 / 30 s); the mel is zero-padded on the right
  inputs       mel [1, 128, T_b], mel_valid [1, T_b] = (t < frames), v1 [1, T1], v2 [1, T2], v3 [1, T3] = (t < L_k);
               L_{k+1} = (L_k + 2 - 3) // 2 + 1 from L_0 = frames (the provider's subsampling lengths) and
               T_{k+1} = (T_k + 2 - 3) // 2 + 1 from T_0 = T_b (Conv2d k3 s2 p1 output sizes); P = L_3
  prefix_rows  out[0, :P]

mel(precision="float32") keeps every step in float32 like the provider (numpy's float32 FFT and matmul);
mel(precision="float64") runs every step in float64 and casts once at the end. Neither is bit-equal to torch.stft;
both were measured against the provider's own output when this host was written.
"""
from __future__ import annotations

import sys

import numpy as np

sys.dont_write_bytecode = True

SAMPLE_RATE, MIN_SAMPLES, MAX_SECONDS = 16000, 8000, 30
N_FFT, WIN, HOP, N_MELS = 512, 400, 160, 128
LOG_GUARD = 2.0 ** -24
STD_EPS = 1e-5
T_BUCKETS = (501, 1001, 2001, 3001)   # STFT frames of a 5 / 10 / 20 / 30 s clip (n // 160 + 1)
D = 1024


def waveform(audio) -> np.ndarray:
    """The provider's waveform(): 1-D int16 or float -> float32 [N], cut to 30 s, zero-padded to 0.5 s."""
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be mono: a 1-D array of 16 kHz samples")
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    x = x.astype(np.float32) / np.float32(32768.0) if x.dtype == np.int16 else x.astype(np.float32)
    if len(x) < MIN_SAMPLES:
        x = np.pad(x, (0, MIN_SAMPLES - len(x)))
    return x


def slaney_filterbank(sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = 128) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, norm="slaney") in float32, computed as librosa computes it."""
    # verbatim copy of the provider's audio.slaney_filterbank (numpy only)
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)[()]
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = f_sp * m
        high = m >= min_log_mel
        f[high] = min_log_hz * np.exp(logstep * (m[high] - min_log_mel))
        return f

    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, np.fft.rfftfreq(n=n_fft, d=1.0 / sr))
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


def hann_window_f32() -> np.ndarray:
    """torch.hann_window(400, periodic=False) in float32: arange * float32(2 pi / 399) -> cos -> * -0.5 + 0.5
    (ATen hamming_window with alpha = beta = 0.5); the cos is taken in float64 and rounded to float32."""
    arg = np.arange(WIN, dtype=np.float32) * np.float32(np.pi * 2.0 / (WIN - 1))
    c = np.cos(arg.astype(np.float64)).astype(np.float32)
    return c * np.float32(-0.5) + np.float32(0.5)


def frame_count(n: int) -> tuple[int, int]:
    """-> (valid frames = n // 160, STFT frames T = n // 160 + 1) for n samples (the provider's floor_divide)."""
    frames = (n + N_FFT // 2 * 2 - N_FFT) // HOP
    return frames, 1 + n // HOP


_FB = None


def mel_stages(x: np.ndarray, precision: str = "float32", window: np.ndarray | None = None) -> dict:
    """waveform() output float32 [n] -> every stage of the front end (the check script compares them one by one):
    pre [n] (float32), power [257, T], lin [128, T], log [128, T], mean / std [128], out [128, T] (precision dtype),
    frames, T. `window` (float32 [400]) replaces hann_window_f32() (the check's identity run uses torch's)."""
    global _FB
    if _FB is None:
        _FB = slaney_filterbank(n_fft=N_FFT, n_mels=N_MELS)
    dt = {"float32": np.float32, "float64": np.float64}[precision]
    x = np.asarray(x, np.float32)
    n = x.shape[0]
    frames, T = frame_count(n)
    y = np.empty(n, np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - np.float32(0.97) * x[:-1]          # float32, as the provider (0.97 rounds to float32 there too)
    w = np.zeros(N_FFT, dt)
    off = (N_FFT - WIN) // 2
    w[off:off + WIN] = (hann_window_f32() if window is None else np.asarray(window, np.float32)).astype(dt)
    p = np.pad(y.astype(dt), (N_FFT // 2, N_FFT // 2))   # center=True, pad_mode="constant"
    assert 1 + (p.shape[0] - N_FFT) // HOP == T
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(T)[:, None]
    spec = np.fft.rfft(p[idx] * w, axis=1)                # [T, 257], complex64 for float32 input (numpy >= 2)
    re, im = spec.real.astype(dt), spec.imag.astype(dt)
    power = np.sqrt(re * re + im * im)
    power = (power * power).T                              # the provider squares the magnitude; [257, T]
    lin = _FB.astype(dt) @ power                           # [128, T]
    m = np.log(lin + dt(LOG_GUARD))
    valid = np.arange(T) < frames
    count = int(valid.sum())
    mean = np.where(valid[None], m, dt(0)).sum(1, dtype=dt) / dt(count)
    var = np.square(np.where(valid[None], m - mean[:, None], dt(0))).sum(1, dtype=dt) / dt(count - 1.0)
    std = np.sqrt(var)
    std = np.where(np.isnan(std), dt(0), std)
    out = (m - mean[:, None]) / (std + dt(STD_EPS))[:, None]
    out = np.where(valid[None], out, dt(0))
    return {"pre": y, "power": power, "lin": lin, "log": m, "mean": mean, "std": std, "out": out,
            "frames": frames, "T": T}


def mel(x: np.ndarray, precision: str = "float32") -> tuple[np.ndarray, int]:
    """waveform() output float32 [n] -> (mel float32 [128, T], frames)."""
    s = mel_stages(x, precision)
    return s["out"].astype(np.float32), s["frames"]


def sub_len(n: int) -> int:
    """Conv2d(k 3, stride 2, pad 1): output size; the provider's lengths rule (lengths + 2 - 3) // 2 + 1."""
    return (n + 2 - 3) // 2 + 1


def lengths(frames: int) -> tuple[int, int, int]:
    l1 = sub_len(frames)
    l2 = sub_len(l1)
    return l1, l2, sub_len(l2)


def dims(T: int) -> tuple[int, int, int]:
    """Bucket T_b -> (T1, T2, T3) of the graph's mask inputs and output rows."""
    return lengths(T)


def bucket_for(T: int, buckets=T_BUCKETS) -> int:
    for b in buckets:
        if T <= b:
            return b
    raise ValueError(f"{T} STFT frames exceed the largest bucket ({buckets[-1]}) — waveform() caps a clip at 30 s")


def build_inputs(mel_ft: np.ndarray, frames: int, T_b: int) -> dict:
    """mel [128, T] (T = frames + 1) -> the five inputs of signature audio_<T_b> (float32, batch 1)."""
    T = mel_ft.shape[1]
    if T > T_b:
        raise ValueError(f"{T} frames do not fit bucket {T_b}")
    x = np.zeros((1, N_MELS, T_b), np.float32)
    x[0, :, :T] = mel_ft
    T1, T2, T3 = dims(T_b)
    L1, L2, L3 = lengths(frames)
    return {"mel": x,
            "mel_valid": (np.arange(T_b) < frames).astype(np.float32)[None],
            "v1": (np.arange(T1) < L1).astype(np.float32)[None],
            "v2": (np.arange(T2) < L2).astype(np.float32)[None],
            "v3": (np.arange(T3) < L3).astype(np.float32)[None]}


def prepare(audio, precision: str = "float32", bucket: int | None = None) -> tuple[dict, dict]:
    """One clip -> (graph inputs, info: n, frames, T, T_b, L1-L3 = P)."""
    w = waveform(audio)
    m, frames = mel(w, precision)
    T = m.shape[1]
    T_b = bucket or bucket_for(T)
    L = lengths(frames)
    return build_inputs(m, frames, T_b), {"n": int(w.shape[0]), "frames": int(frames), "T": int(T), "T_b": int(T_b),
                                          "T123": list(dims(T_b)), "L123": list(L), "P": int(L[2]),
                                          "precision": precision}


def read_audio(path) -> np.ndarray:
    """A 16 kHz mono file -> int16 samples, read as the model card reads its clip (soundfile.read(..., dtype="int16")).
    Another rate or more than one channel is refused: the model takes 16 kHz mono, and resampling is not part of the
    provider's code."""
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="int16")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz; the model takes {SAMPLE_RATE} Hz mono (resample it first)")
    if samples.ndim != 1:
        raise ValueError(f"{path}: {samples.shape[1]} channels; the model takes one (mono)")
    return samples


def prefix_rows(out: np.ndarray, info: dict) -> np.ndarray:
    """The graph's prefix [1, T3, 1024] -> the P valid rows [P, 1024]."""
    out = np.asarray(out, np.float32)
    assert out.shape == (1, dims(info["T_b"])[2], D), (out.shape, info["T_b"])
    return out[0, :info["P"]].copy()


if __name__ == "__main__":
    print(__doc__)

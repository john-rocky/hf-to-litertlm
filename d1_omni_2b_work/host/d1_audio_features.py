"""The audio path of the d1-Omni-2B host, in numpy: what the provider does to a clip before its audio encoder.

The provider (`media.Reader._decoded` + `Lfm3OmniProcessor`) reads a clip (`media.read_audio`: 16-bit PCM WAV or a
NumPy file; imported from the snapshot, it needs no torch), averages its channels and resamples it to 16 kHz in torch
(`media.resampled`), and hands it to `Lfm3OmniFeatureExtractor` (the snapshot's vendored copy): right pad to whole
frames, periodic Hann window of 400, STFT (n_fft 400, hop 160, no centering), |X|^2, slaney mel filters (128), log10,
floor -12, (x + 4) / 4. This module ports the torch parts to numpy and reproduces the provider's float32 arithmetic on
this machine bit for bit (test_host.py `audio_features` checks every step against the provider's own code):

- `mel_filter_bank`: transformers.audio_utils' (slaney scale and norm), copied: numpy float64, the same values.
- the window: `torch.hann_window(400)` = cos(k * float32(2 pi / 400)) * -0.5 + 0.5 in float32, the cosine as
  PyTorch computes it on this CPU: SLEEF's `cosf_u1` (vectorized; AT_BUILD_ARM_VEC256_WITH_SLEEF), ported below. A
  correctly rounded cosine differs from SLEEF's in 21 of the 401 values.
- the STFT: frames times the window (float32 products), then the real FFT as PyTorch runs it without MKL: pocketfft's
  `rfftp` (radix 4, 4, 5, 5 for n = 400) in float32, with the multiply-adds the compiler fused (`a*b + c` in one
  expression is one FMA: clang's -ffp-contract=on). numpy's own FFT computes other roundings (its float32 FFT differs
  from torch's in 93 % of the bins by up to 4e-7 relative), and a mel band of the int16 noise floor next to loud bins
  turns that into up to 1.7e-4 on a speech clip's log-mel: the bar is 1e-5, so the port follows torch's arithmetic.
- |X|: `torch.abs` of a complex64 = hypotf (= np.hypot in float32; np.abs of a complex64 rounds differently).
- the mel product: float32 [128, 201] x [201, F] (Accelerate in numpy and torch: the same sums, measured).
- log10: SLEEF's `log10f_u1` (torch's float32 log10 on this CPU), ported below; a correctly rounded log10 differs
  in 0.1 % of the values by one ulp.
- the resample (`media.resampled`, torchaudio's default): the sinc kernel in float32 with SLEEF's `sinf_u1` /
  `cosf_u1`, and the strided convolution as PyTorch's slow_conv2d runs it: the kernel [new, K] times the im2col
  columns [K, L] (one float32 GEMM in the same operand order, Accelerate).

The multiply-adds are emulated exactly: `fma32` rounds a*b + c once to float32 (float64 product, TwoSum for the sum,
the one double-rounding case resolved).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

F32 = np.float32
FLT_MIN = F32(1.17549435e-38)


# --------------------------------------------------------------------------- #
# float32 fused multiply-add, exactly
# --------------------------------------------------------------------------- #


def fma32(a, b, c) -> np.ndarray:
    """a * b + c rounded once to float32 (what one FMA instruction returns), for float32 operands.

    The float64 product of two float32 values is exact; s = p + c is the float64 sum, err its rounding error (TwoSum).
    Rounding s to float32 is then the correctly rounded result unless s sits exactly on a float32 midpoint while err
    is not zero: then the true sum lies beyond the midpoint on err's side, and s is moved one float64 step that way
    first. Exact for results in float32's normal range (a subnormal result can be off by one subnormal step)."""
    a = np.asarray(a, dtype=F32).astype(np.float64)
    b = np.asarray(b, dtype=F32).astype(np.float64)
    c = np.asarray(c, dtype=F32).astype(np.float64)
    p = a * b
    s = p + c
    bv = s - p
    err = (p - (s - bv)) + (c - bv)
    s = np.asarray(s)
    mid = (s.view(np.uint64) & np.uint64(0x1FFFFFFF)) == np.uint64(0x10000000)
    fix = mid & (err != 0)
    if np.any(fix):
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return s.astype(F32)


# --------------------------------------------------------------------------- #
# SLEEF's float32 sin / cos / log10 (u1, the vectorized functions PyTorch calls), ported
# --------------------------------------------------------------------------- #

_PI_A2 = F32(3.1414794921875)
_PI_B2 = F32(0.00011315941810607910156)
_PI_C2 = F32(1.9841872589410058936e-09)
_M_1_PI = F32(0.318309886183790671537767526745028724)
_TRIG_RANGE = F32(125.0)  # TRIGRANGEMAX2f: beyond it SLEEF reduces with Payne-Hanek (not ported)


def _dfadd2_ff(x, y):
    s = x + y
    v = s - x
    return s, (x - (s - v)) + (y - v)


def _dfadd2_f2f(xx, xy, y):
    s = xx + y
    v = s - xx
    return s, ((xx - (s - v)) + (y - v)) + xy


def _dfadd_f2f(xx, xy, y):
    s = xx + y
    return s, ((xx - s) + y) + xy


def _dfadd_ff(x, y):
    s = x + y
    return s, (x - s) + y


def _dfadd_ff2(x, yx, yy):
    s = x + yx
    return s, ((x - s) + yx) + yy


def _dfadd_f2f2(xx, xy, yx, yy):
    s = xx + yx
    return s, (((xx - s) + yx) + xy) + yy


def _dfsqu(xx, xy):
    s = xx * xx
    return s, fma32(xx + xx, xy, fma32(xx, xx, -s))


def _dfmul_f2f2(xx, xy, yx, yy):
    s = xx * yx
    return s, fma32(xx, yy, fma32(xy, yx, fma32(xx, yx, -s)))


def _dfmul_f2f(xx, xy, y):
    s = xx * y
    return s, fma32(xy, y, fma32(xx, y, -s))


def _dfmul_scalar(xx, xy, yx, yy):
    return fma32(xx, yx, fma32(xy, yx, xx * yy))


def _dfdiv(nx, ny, dx, dy):
    t = F32(1) / dx
    s = nx * t
    u = fma32(t, nx, -s)
    v = fma32(-dy, t, fma32(-dx, t, F32(1)))
    return s, fma32(s, v, fma32(ny, t, u))


def _trig_tail(sx, sy, tx, ty):
    u = F32(2.6083159809786593541503e-06)
    u = fma32(u, sx, F32(-0.0001981069071916863322258))
    u = fma32(u, sx, F32(0.00833307858556509017944336))
    ax, ay = _dfadd_ff(F32(-0.166666597127914428710938), u * sx)
    mx, my = _dfmul_f2f2(ax, ay, sx, sy)
    xx, xy = _dfadd_ff2(F32(1), mx, my)
    return _dfmul_scalar(tx, ty, xx, xy)


def _trig_input(d) -> np.ndarray:
    d = np.asarray(d, dtype=F32)
    if d.size and not np.all(np.abs(d) < _TRIG_RANGE):
        raise ValueError("the SLEEF port covers |x| < 125 (its Payne-Hanek reduction for larger values is not ported)")
    return d


def sleef_sinf_u1(d) -> np.ndarray:
    """SLEEF xsinf_u1 (1.0-ulp sinf) for |d| < 125, with FMA: torch.sin of a float32 tensor on this CPU."""
    d = _trig_input(d)
    u = np.rint(d * _M_1_PI).astype(F32)
    q = u.astype(np.int32)
    v = fma32(u, -_PI_A2, d)
    sx, sy = _dfadd2_ff(v, u * -_PI_B2)
    sx, sy = _dfadd_f2f(sx, sy, u * -_PI_C2)
    tx, ty = sx, sy
    sx, sy = _dfsqu(sx, sy)
    r = _trig_tail(sx, sy, tx, ty)
    r = np.where((q & 1) == 1, -r, r).astype(F32)
    return np.where((d == 0) & np.signbit(d), d, r).astype(F32)


def sleef_cosf_u1(d) -> np.ndarray:
    """SLEEF xcosf_u1 (1.0-ulp cosf) for |d| < 125, with FMA: torch.cos of a float32 tensor on this CPU."""
    d = _trig_input(d)
    dq = fma32(np.rint(fma32(d, _M_1_PI, F32(-0.5))).astype(F32), F32(2), F32(1))
    q = dq.astype(np.int32)
    sx, sy = _dfadd2_ff(d, dq * (-_PI_A2 * F32(0.5)))
    sx, sy = _dfadd2_f2f(sx, sy, dq * (-_PI_B2 * F32(0.5)))
    sx, sy = _dfadd2_f2f(sx, sy, dq * (-_PI_C2 * F32(0.5)))
    tx, ty = sx, sy
    sx, sy = _dfsqu(sx, sy)
    r = _trig_tail(sx, sy, tx, ty)
    return np.where((q & 2) == 0, -r, r).astype(F32)


def sleef_log10f_u1(d) -> np.ndarray:
    """SLEEF xlog10f_u1 (1.0-ulp log10f) with FMA: torch.log10 of a float32 tensor on this CPU (0 -> -inf, a negative
    value or NaN -> NaN, +inf -> +inf; subnormals scaled by 2^64 first, as SLEEF does)."""
    d = np.asarray(d, dtype=F32)
    small = d < FLT_MIN
    x = np.where(small, d * F32(2.0 ** 64), d).astype(F32)
    e = ((x * F32(1.0 / 0.75)).view(np.int32) >> 23 & 0xFF) - 0x7F
    m = (x.view(np.int32) + (-e << 23)).view(F32)
    e = np.where(small, e - 64, e)
    ax, ay = _dfadd2_ff(F32(-1), m)
    bx, by = _dfadd2_ff(F32(1), m)
    xx, xy = _dfdiv(ax, ay, bx, by)
    x2 = xx * xx
    t = fma32(F32(0.1314289868e+0), x2, F32(0.1735493541e+0))
    t = fma32(t, x2, F32(0.2895309627e+0))
    sx, sy = _dfmul_f2f(F32(0.30103001), F32(-1.432098889e-08), e.astype(F32))
    px, py = _dfmul_f2f2(xx, xy, F32(0.868588984), F32(-2.170757285e-08))
    sx, sy = _dfadd_f2f2(sx, sy, px, py)
    sx, sy = _dfadd_f2f(sx, sy, (x2 * xx) * t)
    r = (sx + sy).astype(F32)
    r = np.where(np.isposinf(d), F32(np.inf), r)
    r = np.where((d < 0) | np.isnan(d), F32(np.nan), r)
    return np.where(d == 0, F32(-np.inf), r).astype(F32)


# --------------------------------------------------------------------------- #
# pocketfft's real FFT (rfftp, forward, float32), as PyTorch runs it
# --------------------------------------------------------------------------- #


def _factorize(n: int) -> list[int]:
    """rfftp::factorize: fours first, a two at the front, then odd factors."""
    fact = []
    while n % 4 == 0:
        fact.append(4)
        n //= 4
    if n % 2 == 0:
        n //= 2
        fact.append(2)
        fact[0], fact[-1] = fact[-1], fact[0]
    d = 3
    while d * d <= n:
        while n % d == 0:
            fact.append(d)
            n //= d
        d += 2
    if n > 1:
        fact.append(n)
    return fact


def _largest_prime_factor(n: int) -> int:
    res = 1
    while n % 2 == 0:
        res, n = 2, n // 2
    x = 3
    while x * x <= n:
        while n % x == 0:
            res, n = x, n // x
        x += 2
    return n if n > 1 else res


def _twiddles(n: int):
    """sincos_2pibyn<float>(n): two tables of double cos / sin (libm, as std::cos / std::sin), combined in double and
    rounded to float32."""
    ang = 0.25 * math.pi / n

    def calc(x: int):
        x <<= 3
        if x < 4 * n:
            if x < 2 * n:
                if x < n:
                    return math.cos(x * ang), math.sin(x * ang)
                return math.sin((2 * n - x) * ang), math.cos((2 * n - x) * ang)
            x -= 2 * n
            if x < n:
                return -math.sin(x * ang), math.cos(x * ang)
            return -math.cos((2 * n - x) * ang), math.sin((2 * n - x) * ang)
        x = 8 * n - x
        if x < 2 * n:
            if x < n:
                return math.cos(x * ang), -math.sin(x * ang)
            return math.sin((2 * n - x) * ang), -math.cos((2 * n - x) * ang)
        x -= 2 * n
        if x < n:
            return -math.sin(x * ang), -math.cos(x * ang)
        return -math.cos((2 * n - x) * ang), -math.sin((2 * n - x) * ang)

    nval = (n + 2) // 2
    shift = 1
    while (1 << shift) * (1 << shift) < nval:
        shift += 1
    mask = (1 << shift) - 1
    v1 = [(1.0, 0.0)] + [calc(i) for i in range(1, mask + 1)]
    v2 = [(1.0, 0.0)] + [calc(i * (mask + 1)) for i in range(1, (nval + mask) // (mask + 1))]

    def at(idx: int) -> tuple[np.float32, np.float32]:
        sign = 1.0
        if 2 * idx > n:
            idx, sign = n - idx, -1.0
        x1, x2 = v1[idx & mask], v2[idx >> shift]
        return F32(x1[0] * x2[0] - x1[1] * x2[1]), F32(sign) * F32(x1[0] * x2[1] + x1[1] * x2[0])

    return at


_HSQT2 = F32(0.707106781186547524400844362104849)
_TAUR, _TAUI = F32(-0.5), F32(0.8660254037844386467637231707529362)
_TR11, _TI11 = F32(0.3090169943749474241022934171828191), F32(0.9510565162951535721164393333793821)
_TR12, _TI12 = F32(-0.8090169943749474241022934171828191), F32(0.5877852522924731291687059546390728)


def _mulpm(wr, wi, e, f):
    """MULPM: (a, b) = conj(wr + i wi) * (e + i f), as compiled: a = fma(wr, e, wi*f), b = fma(wr, f, -(wi*e))."""
    return fma32(wr, e, wi * f), fma32(wr, f, -(wi * e))


class RealFFT:
    """pocketfft_r<float>(n).exec(forward) for the lengths whose factors are 2, 3, 4 and 5 (rfftp; the generic radix
    and Bluestein's plan are not ported), over the last axis of a float32 array."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("an FFT length is at least 1")
        lpf = 0 if n < 50 else _largest_prime_factor(n)
        self.n, self.fact = n, _factorize(n) if n > 1 else []
        if lpf * lpf > n or any(f > 5 for f in self.fact):
            raise NotImplementedError(f"pocketfft's plan for length {n} (a prime factor above 5) is not ported")
        twid = _twiddles(n)
        self.tw = []
        l1 = 1
        for k, ip in enumerate(self.fact):
            ido = n // (l1 * ip)
            tw = None
            if k < len(self.fact) - 1:  # the last factor needs no twiddles
                tw = np.zeros((ip - 1) * (ido - 1), dtype=F32)
                for j in range(1, ip):
                    for i in range(1, (ido - 1) // 2 + 1):
                        tw[(j - 1) * (ido - 1) + 2 * i - 2], tw[(j - 1) * (ido - 1) + 2 * i - 1] = twid(j * l1 * i)
            self.tw.append(tw)
            l1 *= ip

    def halfcomplex(self, x: np.ndarray) -> np.ndarray:
        """rfftp::exec(c, 1, true): r0, r1, i1, r2, i2, ... (the forward transform in pocketfft's packed order)."""
        c = np.ascontiguousarray(x, dtype=F32)
        if c.shape[-1] != self.n:
            raise ValueError(f"this plan transforms {self.n} values, not {c.shape[-1]}")
        nf, l1 = len(self.fact), self.n
        for k1 in range(nf):
            k = nf - k1 - 1
            ip = self.fact[k]
            ido = self.n // l1
            l1 //= ip
            c = {2: self._radf2, 3: self._radf3, 4: self._radf4, 5: self._radf5}[ip](ido, l1, c, self.tw[k])
        return c

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The one-sided spectrum (real, imaginary) [..., n // 2 + 1], as `general_r2c` writes it."""
        r = self.halfcomplex(x)
        n, m = self.n, (self.n - 1) // 2
        shape = r.shape[:-1] + (n // 2 + 1,)
        re, im = np.zeros(shape, dtype=F32), np.zeros(shape, dtype=F32)
        re[..., 0] = r[..., 0]
        re[..., 1:m + 1] = r[..., 1:2 * m:2]
        im[..., 1:m + 1] = r[..., 2:2 * m + 1:2]
        if n % 2 == 0:
            re[..., n // 2] = r[..., n - 1]
        return re, im

    # each pass reads CC(a, b, c) = cc[a + ido * (b + l1 * c)] and writes CH(a, b, c) = ch[a + ido * (b + ip * c)]

    @staticmethod
    def _radf2(ido, l1, cc, wa):
        lead = cc.shape[:-1]
        CC = cc.reshape(lead + (2, l1, ido))
        ch = np.empty(lead + (l1, 2, ido), dtype=F32)
        c0, c1 = CC[..., 0, :, 0], CC[..., 1, :, 0]
        ch[..., :, 0, 0] = c0 + c1
        ch[..., :, 1, ido - 1] = c0 - c1
        if ido % 2 == 0:
            ch[..., :, 1, 0] = -CC[..., 1, :, ido - 1]
            ch[..., :, 0, ido - 1] = CC[..., 0, :, ido - 1]
        if ido <= 2:
            return ch.reshape(lead + (-1,))
        i = np.arange(2, ido, 2)
        ic = ido - i
        row0, row1 = CC[..., 0, :, :], CC[..., 1, :, :]
        tr2, ti2 = _mulpm(wa[i - 2], wa[i - 1], row1[..., i - 1], row1[..., i])
        a, b = row0[..., i - 1], row0[..., i]
        v = ch[..., :, 0, :]
        v[..., i - 1], v[..., i] = a + tr2, ti2 + b
        v = ch[..., :, 1, :]
        v[..., ic - 1], v[..., ic] = a - tr2, ti2 - b
        return ch.reshape(lead + (-1,))

    @staticmethod
    def _radf3(ido, l1, cc, wa):
        lead = cc.shape[:-1]
        CC = cc.reshape(lead + (3, l1, ido))
        ch = np.empty(lead + (l1, 3, ido), dtype=F32)
        c0, c1, c2 = (CC[..., j, :, 0] for j in range(3))
        cr2 = c1 + c2
        ch[..., :, 0, 0] = c0 + cr2
        ch[..., :, 2, 0] = _TAUI * (c2 - c1)
        ch[..., :, 1, ido - 1] = fma32(_TAUR, cr2, c0)
        if ido == 1:
            return ch.reshape(lead + (-1,))
        i = np.arange(2, ido, 2)
        ic = ido - i
        rows = [CC[..., j, :, :] for j in range(3)]
        dr2, di2 = _mulpm(wa[i - 2], wa[i - 1], rows[1][..., i - 1], rows[1][..., i])
        dr3, di3 = _mulpm(wa[i - 2 + (ido - 1)], wa[i - 1 + (ido - 1)], rows[2][..., i - 1], rows[2][..., i])
        dr2, di2, dr3, di3 = dr2 + dr3, di2 + di3, di2 - di3, dr3 - dr2
        a, b = rows[0][..., i - 1], rows[0][..., i]
        tr2, ti2 = fma32(_TAUR, dr2, a), fma32(_TAUR, di2, b)
        tr3, ti3 = _TAUI * dr3, _TAUI * di3
        v = ch[..., :, 0, :]
        v[..., i - 1], v[..., i] = a + dr2, b + di2
        v = ch[..., :, 2, :]
        v[..., i - 1], v[..., i] = tr2 + tr3, ti3 + ti2
        v = ch[..., :, 1, :]
        v[..., ic - 1], v[..., ic] = tr2 - tr3, ti3 - ti2
        return ch.reshape(lead + (-1,))

    @staticmethod
    def _radf4(ido, l1, cc, wa):
        lead = cc.shape[:-1]
        CC = cc.reshape(lead + (4, l1, ido))
        ch = np.empty(lead + (l1, 4, ido), dtype=F32)
        c0, c1, c2, c3 = (CC[..., j, :, 0] for j in range(4))
        tr1, tr2 = c3 + c1, c0 + c2
        ch[..., :, 2, 0] = c3 - c1
        ch[..., :, 1, ido - 1] = c0 - c2
        ch[..., :, 0, 0] = tr2 + tr1
        ch[..., :, 3, ido - 1] = tr2 - tr1
        if ido % 2 == 0:
            e0, e1, e2, e3 = (CC[..., j, :, ido - 1] for j in range(4))
            ti1 = (-_HSQT2) * (e1 + e3)
            tr1 = _HSQT2 * (e1 - e3)
            ch[..., :, 0, ido - 1] = e0 + tr1
            ch[..., :, 2, ido - 1] = e0 - tr1
            ch[..., :, 3, 0] = ti1 + e2
            ch[..., :, 1, 0] = ti1 - e2
        if ido <= 2:
            return ch.reshape(lead + (-1,))
        i = np.arange(2, ido, 2)
        ic = ido - i
        rows = [CC[..., j, :, :] for j in range(4)]

        def mp(x, j):
            return _mulpm(wa[i - 2 + x * (ido - 1)], wa[i - 1 + x * (ido - 1)], rows[j][..., i - 1], rows[j][..., i])

        cr2, ci2 = mp(0, 1)
        cr3, ci3 = mp(1, 2)
        cr4, ci4 = mp(2, 3)
        tr1, tr4 = cr4 + cr2, cr4 - cr2
        ti1, ti4 = ci2 + ci4, ci2 - ci4
        a, b = rows[0][..., i - 1], rows[0][..., i]
        tr2, tr3 = a + cr3, a - cr3
        ti2, ti3 = b + ci3, b - ci3
        v = ch[..., :, 0, :]
        v[..., i - 1], v[..., i] = tr2 + tr1, ti1 + ti2
        v = ch[..., :, 3, :]
        v[..., ic - 1], v[..., ic] = tr2 - tr1, ti1 - ti2
        v = ch[..., :, 2, :]
        v[..., i - 1], v[..., i] = tr3 + ti4, tr4 + ti3
        v = ch[..., :, 1, :]
        v[..., ic - 1], v[..., ic] = tr3 - ti4, tr4 - ti3
        return ch.reshape(lead + (-1,))

    @staticmethod
    def _radf5(ido, l1, cc, wa):
        lead = cc.shape[:-1]
        CC = cc.reshape(lead + (5, l1, ido))
        ch = np.empty(lead + (l1, 5, ido), dtype=F32)
        c0, c1, c2, c3, c4 = (CC[..., j, :, 0] for j in range(5))
        cr2, ci5 = c4 + c1, c4 - c1
        cr3, ci4 = c3 + c2, c3 - c2
        ch[..., :, 0, 0] = c0 + cr2 + cr3
        ch[..., :, 1, ido - 1] = fma32(_TR12, cr3, fma32(_TR11, cr2, c0))
        ch[..., :, 2, 0] = fma32(_TI11, ci5, _TI12 * ci4)
        ch[..., :, 3, ido - 1] = fma32(_TR11, cr3, fma32(_TR12, cr2, c0))
        ch[..., :, 4, 0] = fma32(_TI12, ci5, -(_TI11 * ci4))
        if ido == 1:
            return ch.reshape(lead + (-1,))
        i = np.arange(2, ido, 2)
        ic = ido - i
        rows = [CC[..., j, :, :] for j in range(5)]

        def mp(x, j):
            return _mulpm(wa[i - 2 + x * (ido - 1)], wa[i - 1 + x * (ido - 1)], rows[j][..., i - 1], rows[j][..., i])

        dr2, di2 = mp(0, 1)
        dr3, di3 = mp(1, 2)
        dr4, di4 = mp(2, 3)
        dr5, di5 = mp(3, 4)
        dr2, di2, dr5, di5 = dr2 + dr5, di2 + di5, di2 - di5, dr5 - dr2
        dr3, di3, dr4, di4 = dr3 + dr4, di3 + di4, di3 - di4, dr4 - dr3
        a, b = rows[0][..., i - 1], rows[0][..., i]
        v = ch[..., :, 0, :]
        v[..., i - 1], v[..., i] = a + dr2 + dr3, b + di2 + di3
        tr2 = fma32(_TR12, dr3, fma32(_TR11, dr2, a))
        ti2 = fma32(_TR12, di3, fma32(_TR11, di2, b))
        tr3 = fma32(_TR11, dr3, fma32(_TR12, dr2, a))
        ti3 = fma32(_TR11, di3, fma32(_TR12, di2, b))
        tr5 = fma32(_TI11, dr5, _TI12 * dr4)
        ti5 = fma32(_TI11, di5, _TI12 * di4)
        tr4 = fma32(_TI12, dr5, -(_TI11 * dr4))
        ti4 = fma32(_TI12, di5, -(_TI11 * di4))
        v = ch[..., :, 2, :]
        v[..., i - 1], v[..., i] = tr2 + tr5, ti5 + ti2
        v = ch[..., :, 1, :]
        v[..., ic - 1], v[..., ic] = tr2 - tr5, ti5 - ti2
        v = ch[..., :, 4, :]
        v[..., i - 1], v[..., i] = tr3 + tr4, ti4 + ti3
        v = ch[..., :, 3, :]
        v[..., ic - 1], v[..., ic] = tr3 - tr4, ti4 - ti3
        return ch.reshape(lead + (-1,))


# --------------------------------------------------------------------------- #
# transformers.audio_utils.mel_filter_bank (slaney), copied (numpy float64)
# --------------------------------------------------------------------------- #


def hertz_to_mel(freq, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(num_frequency_bins: int, num_mel_filters: int, min_frequency: float, max_frequency: float,
                    sampling_rate: int, norm: str | None = None, mel_scale: str = "htk",
                    triangularize_in_mel_space: bool = False) -> np.ndarray:
    """transformers.audio_utils.mel_filter_bank (5.19.0), copied: [num_frequency_bins, num_mel_filters] float64."""
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')
    if num_frequency_bins < 2:
        raise ValueError(f"Require num_frequency_bins: {num_frequency_bins} >= 2")
    if min_frequency > max_frequency:
        raise ValueError(f"Require min_frequency: {min_frequency} <= max_frequency: {max_frequency}")
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)
    if triangularize_in_mel_space:
        fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
    if norm is not None and norm == "slaney":
        enorm = 2.0 / (filter_freqs[2: num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)
    if (mel_filters.max(axis=0) == 0.0).any():
        raise ValueError(f"a mel filter has all zero values ({num_mel_filters} filters, {num_frequency_bins} bins)")
    return mel_filters


# --------------------------------------------------------------------------- #
# torch's periodic Hann window and torchaudio's default resample, as the provider computes them
# --------------------------------------------------------------------------- #


def hann_window(n: int) -> np.ndarray:
    """torch.hann_window(n) (periodic, float32): arange(n + 1) * float32(2 pi / n), SLEEF's cos, * -0.5 + 0.5, the
    first n."""
    k = np.arange(n + 1, dtype=F32) * F32(math.pi * 2.0 / n)
    return (sleef_cosf_u1(k) * F32(-0.5) + F32(0.5))[:n].astype(F32)


def resample_kernel(old: int, new: int, zeros: int, cut: float, width: int) -> np.ndarray:
    """`media.resampled`'s filter [new, 2 * width + old], each torch step in float32 (scalars as float32)."""
    t = np.arange(0, -new, -1, dtype=F32)[:, None] / F32(new)
    t = np.clip((t + np.arange(-width, width + old, dtype=F32)[None, :] / F32(old)) * F32(cut),
                F32(-zeros), F32(zeros)).astype(F32)
    window = sleef_cosf_u1(t * F32(math.pi) / F32(zeros) / F32(2))
    window = window * window
    t = t * F32(math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(t == 0, F32(1.0), sleef_sinf_u1(t) / t).astype(F32)
    return (sinc * (window * F32(cut / old))).astype(F32)


def resampled(wave, rate: int, target: int, zeros: int = 6, rolloff: float = 0.99) -> np.ndarray:
    """media.resampled in numpy: a mono float32 waveform at `target` (channels averaged, the provider's own numpy
    lines), the windowed-sinc filter applied as PyTorch's strided convolution computes it (the kernel times the
    im2col columns, one float32 GEMM: conv1d(pad(x, (width, width + old)), kernel, stride=old), phases interleaved)."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 2:  # (channels, samples) or (samples, channels): a clip never has more channels than samples
        wave = (wave.T if wave.shape[0] > wave.shape[1] else wave).mean(0)
    if rate == target:
        return wave
    gcd = math.gcd(int(rate), int(target))
    old, new = int(rate) // gcd, int(target) // gcd
    cut = min(old, new) * rolloff
    width = math.ceil(zeros * old / cut)
    kernel = resample_kernel(old, new, zeros, cut, width)
    size = kernel.shape[1]
    x = np.zeros(len(wave) + 2 * width + old, dtype=F32)
    x[width:width + len(wave)] = wave
    count = (len(x) - size) // old + 1
    cols = np.ascontiguousarray(np.lib.stride_tricks.as_strided(x, shape=(size, count), strides=(4, 4 * old)))
    out = kernel @ cols  # [new, count]: torch's output channels x positions
    return np.ascontiguousarray(out.T).reshape(-1)[:math.ceil(new * len(wave) / old)]


# --------------------------------------------------------------------------- #
# Lfm3OmniFeatureExtractor, one clip
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtractorSettings:
    feature_size: int = 128
    sampling_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    log_floor: float = -12.0
    normalization_offset: float = 4.0
    normalization_divisor: float = 4.0
    encoder_stride: int = 8
    padding_value: float = 0.0


class FeatureExtractor:
    """Lfm3OmniFeatureExtractor with the snapshot's processor_config.json, for one mono clip at its sampling rate."""

    def __init__(self, snapshot_dir: str | None = None, settings: ExtractorSettings | None = None):
        if settings is None:
            with open(os.path.join(snapshot_dir, "processor_config.json")) as f:
                doc = json.load(f).get("feature_extractor")
            if doc is None:
                raise ValueError("this checkpoint has no feature extractor (it takes no audio)")
            if doc.get("feature_extractor_type") != "Lfm3OmniFeatureExtractor":
                raise ValueError(f"the host ports Lfm3OmniFeatureExtractor, not {doc.get('feature_extractor_type')}")
            if doc.get("padding_side", "right") != "right":
                raise ValueError("the host ports right padding")
            names = ExtractorSettings.__dataclass_fields__
            settings = ExtractorSettings(**{k: doc[k] for k in names if k in doc})
        s = self.settings = settings
        if s.win_length != s.n_fft:
            raise NotImplementedError("the host ports a window as long as the FFT (win_length == n_fft)")
        self.fft = RealFFT(s.n_fft)
        self.window = hann_window(s.win_length)
        self.mel_filters = mel_filter_bank(num_frequency_bins=s.n_fft // 2 + 1, num_mel_filters=s.feature_size,
                                           min_frequency=0.0, max_frequency=s.sampling_rate / 2,
                                           sampling_rate=s.sampling_rate, norm="slaney", mel_scale="slaney")
        self.filters_t = self.mel_filters.astype(F32).T  # [mels, bins], the transposed view torch multiplies with

    @property
    def sampling_rate(self) -> int:
        return self.settings.sampling_rate

    def pad_length(self, num_samples: int) -> int:
        s = self.settings
        if num_samples < s.n_fft:
            return s.n_fft
        return num_samples + (-(num_samples - s.n_fft) % s.hop_length)

    def num_frames(self, num_samples: int) -> int:
        return 1 + (self.pad_length(num_samples) - self.settings.n_fft) // self.settings.hop_length

    def num_encoder_tokens(self, num_frames: int) -> int:
        return -(num_frames // -self.settings.encoder_stride)

    def power(self, padded: np.ndarray) -> np.ndarray:
        """|STFT|^2 [bins, frames] of a padded clip (torch.stft, center False, the periodic window)."""
        s = self.settings
        frames = 1 + (len(padded) - s.n_fft) // s.hop_length
        x = np.lib.stride_tricks.sliding_window_view(padded, s.n_fft)[::s.hop_length][:frames] * self.window
        re, im = self.fft(x)
        return np.square(np.hypot(re, im)).T  # [bins, frames], a transposed view as torch's stft returns it

    def log_mel(self, padded: np.ndarray) -> np.ndarray:
        """_log_mel: float32 [mels, frames]."""
        s = self.settings
        mel = self.filters_t @ self.power(padded)
        log_mel = np.maximum(sleef_log10f_u1(mel), F32(s.log_floor))
        return ((log_mel + F32(s.normalization_offset)) / F32(s.normalization_divisor)).astype(F32)

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """`__call__` on one mono clip: right padded with zeros to whole frames, its log-mel `input_features`
        [feature_size, frames] (its `feature_attention_mask` is all ones)."""
        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim != 1:
            raise ValueError(f"Lfm3OmniFeatureExtractor expects mono clips, got a clip of shape {tuple(clip.shape)}")
        padded = np.full(self.pad_length(clip.shape[0]), F32(self.settings.padding_value), dtype=F32)
        padded[:clip.shape[0]] = clip
        return self.log_mel(padded)


def read_audio(media_module, path: str, rate: int):
    """media.read_audio for the formats it reads without torch (16-bit PCM WAV, .npy); any other format goes through
    transformers' load_audio in the provider, which the host does not have."""
    if not (path.endswith(".wav") or path.endswith(".npy")):
        raise NotImplementedError(f"the host reads 16-bit PCM WAV and .npy clips (media.read_audio), not {path!r}")
    return media_module.read_audio(path, rate)

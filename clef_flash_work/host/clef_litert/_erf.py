"""erf in float64 over NumPy arrays (NumPy has none): FreeBSD msun s_erf.c as ported by musl and Rust `libm`,
vectorized region by region. Used by the head's GELU (the author's torch.nn.GELU / F.gelu, the exact erf form).

 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ===================================================="""
from __future__ import annotations

import math

import numpy as np

ERX = 8.45062911510467529297e-01
EFX8 = 1.02703333676410069053e+00
PP = (1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02,
      -5.77027029648944159157e-03, -2.37630166566501626084e-05)
QQ = (3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03,
      1.32494738004321644526e-04, -3.96022827877536812320e-06)
PA = (-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01,
      3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02,
      -2.16637559486879084300e-03)
QA = (1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02,
      1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02)
RA = (-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01,
      -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02,
      -8.12874355063065934246e+01, -9.81432934416914548592e+00)
SA = (1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02,
      6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02,
      6.57024977031928170135e+00, -6.04244152148580987438e-02)
RB = (-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01,
      -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03,
      -4.83519191608651397019e+02)
SB = (3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03,
      3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02,
      -2.24409524465858183362e+01)
SQRT1_2 = math.sqrt(0.5)


def _poly(z, c):
    """c[0] + z * (c[1] + z * (... + z * c[-1])), Horner from the last coefficient (the C source's nesting)."""
    r = np.full_like(z, c[-1])
    for a in c[-2::-1]:
        r = a + z * r
    return r


def _one_plus_z_poly(z, c):
    """1 + z * (c[0] + z * (c[1] + ...)) (the denominators Q / S)."""
    return 1.0 + z * _poly(z, c)


def erf(x) -> np.ndarray:
    """Elementwise erf of a float64 array (NaN -> NaN, +-inf -> +-1)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    ax = np.abs(x)
    hi = (ax.view(np.uint64) >> np.uint64(32)).astype(np.uint32)       # high word of |x|
    sign = np.signbit(x)

    m = hi < 0x3e300000                                                 # |x| < 2**-28
    out[m] = 0.125 * (8.0 * x[m] + EFX8 * x[m])
    m = (hi >= 0x3e300000) & (hi < 0x3feb0000)                          # |x| < 0.84375
    xm = x[m]
    z = xm * xm
    out[m] = xm + xm * (_poly(z, PP) / _one_plus_z_poly(z, QQ))
    m = (hi >= 0x3feb0000) & (hi < 0x3ff40000)                          # 0.84375 <= |x| < 1.25
    s = ax[m] - 1.0
    erfc1 = 1.0 - ERX - _poly(s, PA) / _one_plus_z_poly(s, QA)
    out[m] = np.where(sign[m], -(1.0 - erfc1), 1.0 - erfc1)
    m = (hi >= 0x3ff40000) & (hi < 0x40180000)                          # 1.25 <= |x| < 6
    if m.any():
        a = ax[m]
        h = hi[m]
        s = 1.0 / (a * a)
        near = h < 0x4006db6d                                           # |x| < 1 / 0.35
        r = np.where(near, _poly(s, RA), _poly(s, RB))
        big_s = np.where(near, _one_plus_z_poly(s, SA), _one_plus_z_poly(s, SB))
        zz = (a.view(np.uint64) & np.uint64(0xFFFFFFFF00000000)).view(np.float64)   # low word cleared
        erfc2 = np.exp(-zz * zz - 0.5625) * np.exp((zz - a) * (zz + a) + r / big_s) / a
        out[m] = np.where(sign[m], -(1.0 - erfc2), 1.0 - erfc2)
    m = hi >= 0x40180000                                                # |x| >= 6, inf, NaN
    y = 1.0 - math.ldexp(1.0, -1022)
    out[m] = np.where(np.isnan(x[m]), np.nan, np.where(sign[m], -y, y))
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, the exact form x / 2 * (1 + erf(x / sqrt 2)), evaluated in float64 and rounded to x's float32."""
    xd = np.asarray(x, dtype=np.float64)
    return (0.5 * xd * (1.0 + erf(xd * SQRT1_2))).astype(np.float32)

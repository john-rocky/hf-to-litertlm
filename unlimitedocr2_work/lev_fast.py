#!/usr/bin/env python3
"""Exact Levenshtein distance by Hyyro's bit-parallel form of Myers' algorithm on Python ints (one pass over the
longer string, the shorter one as the bit pattern): the same value as the plain O(n*m) table, checked by
`python lev_fast.py` on random strings. score_uocr.py uses it for the CER (speed only)."""
import random


def lev_fast(a, b):
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq = {}
    for i, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << i)
    full = (1 << m) - 1
    top = 1 << (m - 1)
    pv, mv, score = full, 0, m
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & full
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = (mh | ~(xv | ph)) & full
        mv = ph & xv
    return score


if __name__ == "__main__":
    import os
    import sys
    def lev(a, b):  # the plain O(n*m) table
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    rnd = random.Random(0)
    alpha = "ab cde\n0123{}\\漢字"
    n = 0
    for _ in range(3000):
        a = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 40)))
        b = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 40)))
        assert lev_fast(a, b) == lev(a, b), (a, b, lev_fast(a, b), lev(a, b))
        n += 1
    for _ in range(20):
        a = "".join(rnd.choice(alpha) for _ in range(rnd.randint(300, 900)))
        b = "".join(rnd.choice(alpha) for _ in range(rnd.randint(300, 900)))
        assert lev_fast(a, b) == lev(a, b)
        n += 1
    print(f"lev_fast == lev on {n} random pairs")

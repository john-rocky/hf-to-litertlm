"""Rank metrics with numpy only (no scipy): Spearman with average ranks for ties, average precision."""
import numpy as np


def rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def average_precision(scores, labels):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y = np.asarray(labels)[order]
    hits = np.cumsum(y)
    prec = hits / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / max(1, y.sum()))

"""Parsing and matching helpers shared by run_arms.py and score_arms.py.

Coordinates stay in the model's bins (0..1000 on each axis), so arms that saw the image at different resolutions are
compared on the same scale. Parsing is the package's own: `evaluation.utils.parser.parse_prediction` (shapes),
`evaluation.metrics.counting_metric.predicted_count` (counts), `wevisone_vl.engine.parse_seg_boxes` (boxes before
`<OBJ>`); the in-order word list for OCR walks the same `REF_RE` the parser uses.
"""
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.metrics.counting_metric import predicted_count
from evaluation.utils.parser import BIN_RE, REF_RE, parse_prediction
from wevisone_vl.engine import parse_seg_boxes

COORD_FIRST, COORD_LAST = 151669, 152669          # <0> .. <1000>
EOS_IDS = (151645, 151643)                         # <|im_end|>, <|endoftext|>
POINT_TASKS = ("point", "count_point")
MULTI_LABEL_TASKS = ("detection", "point")         # match within a label; other tasks pool every shape


def strip_answer(raw):
    return (raw or "").split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def is_coord(token_id):
    return COORD_FIRST <= int(token_id) <= COORD_LAST


def coord_bin(token_id):
    return int(token_id) - COORD_FIRST


def label_key(name):
    return re.sub(r"[\s_]+", "", str(name).lower())


def ordered_shapes(text, pointing):
    """[(label, kind, coords)] in output order (bins); kind = box / point / polygon."""
    out = []
    for label, kind, body in REF_RE.findall(text):
        for segment in body.strip().rstrip(",").split(","):
            values = [int(v) for v in BIN_RE.findall(segment)]
            if kind == "point":
                for k in range(0, len(values) - 1, 2):
                    out.append((label.strip(), "point", [float(values[k]), float(values[k + 1])]))
            elif kind == "polygon":
                if values:
                    out.append((label.strip(), "polygon", [float(v) for v in values]))
            else:
                for k in range(0, len(values) - 3, 4):
                    x0, y0, x1, y1 = values[k:k + 4]
                    if x1 > x0 and y1 > y0:
                        out.append((label.strip(), "box", [float(x0), float(y0), float(x1), float(y1)]))
    if not out and pointing:
        shapes = parse_prediction(text, 1000, 1000, pointing=True)
        for label, items in shapes.items():
            for item in items:
                kind = "point" if len(item) == 2 else ("box" if len(item) == 4 else "polygon")
                out.append((label, kind, [float(v) for v in item]))
    return out


def parse(raw, task):
    text = strip_answer(raw)
    pointing = task in POINT_TASKS
    shapes = parse_prediction(text, 1000, 1000, pointing=pointing)
    ordered = ordered_shapes(text, pointing)
    n_box = sum(1 for _, k, _ in ordered if k == "box")
    n_point = sum(1 for _, k, _ in ordered if k == "point")
    return {
        "answer": text,
        "shapes": shapes,
        "ordered": [[lab, kind, coords] for lab, kind, coords in ordered],
        "count_author": int(predicted_count({"extracted_predictions": shapes, "raw_response": text})),
        "count_shapes": n_box + n_point,
        # The answer's own "Total number: N." (count_point answers end with it; added after the first numbers,
        # the author's box count reads every point answer as 0).
        "count_stated": (int(m[-1]) if (m := re.findall(r"Total number:\s*(\d+)", text)) else None),
        "n_box": n_box,
        "n_point": n_point,
        "seg_boxes_bins": [[round(v * 1000) for v in b] for b in parse_seg_boxes(text)],
        "n_obj_tokens": text.count("<OBJ>"),
        # absent target: the checkpoint answers `<ref>…</ref><box>none</box>` (lowercase)
        "is_none": (not ordered) and bool(re.search(r"(?i)\bnone\b", text)),
        "words": [lab for lab, kind, _ in ordered if kind == "box"],
    }


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_boxes(ref, test, thr=0.75):
    """Hungarian on IoU (pairs below thr weigh 0). Returns (matched ref boxes, ious of matched pairs)."""
    if not ref or not test:
        return 0, []
    w = np.array([[iou(r, t) for t in test] for r in ref])
    w[w < thr] = 0.0
    ri, ci = linear_sum_assignment(w, maximize=True)
    got = [float(w[i, j]) for i, j in zip(ri, ci) if w[i, j] > 0]
    return len(got), got


def match_points(ref, test, thr=20.0):
    """Hungarian on the per-axis (Chebyshev) bin distance; a pair counts when both axes are within thr bins
    (2 % of the side). Returns (matched ref points, distances of matched pairs)."""
    if not ref or not test:
        return 0, []
    d = np.array([[max(abs(r[0] - t[0]), abs(r[1] - t[1])) for t in test] for r in ref])
    w = np.where(d <= thr, thr + 1.0 - d, 0.0)
    ri, ci = linear_sum_assignment(w, maximize=True)
    got = [float(d[i, j]) for i, j in zip(ri, ci) if w[i, j] > 0]
    return len(got), got


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _stated_or_shapes(parsed):
    stated = parsed.get("count_stated")
    if stated is None:
        stated = re.findall(r"Total number:\s*(\d+)", parsed["answer"])
        stated = int(stated[-1]) if stated else None
    return stated if stated is not None else parsed["count_shapes"]


def grouped(ordered, kind, task):
    """{label key: [coords]} for one shape kind; tasks outside MULTI_LABEL_TASKS pool every label into ''."""
    out = {}
    for lab, k, coords in ordered:
        if k != kind:
            continue
        key = label_key(lab) if task in MULTI_LABEL_TASKS else ""
        out.setdefault(key, []).append(coords)
    return out


def compare_rows(ref_parsed, test_parsed, task):
    """Structured comparison of two parsed answers to the same request (ref = the arm compared against)."""
    out = {}
    rb, tb = grouped(ref_parsed["ordered"], "box", task), grouped(test_parsed["ordered"], "box", task)
    n_ref_box = sum(len(v) for v in rb.values())
    matched, ious = 0, []
    for key, boxes in rb.items():
        m, got = match_boxes(boxes, tb.get(key, []))
        matched += m
        ious += got
    out["boxes"] = {"n_ref": n_ref_box, "n_test": sum(len(v) for v in tb.values()), "matched": matched,
                    "mean_iou_matched": (sum(ious) / len(ious)) if ious else None}
    rp, tp = grouped(ref_parsed["ordered"], "point", task), grouped(test_parsed["ordered"], "point", task)
    n_ref_pt = sum(len(v) for v in rp.values())
    matched_pt, dists = 0, []
    for key, pts in rp.items():
        m, got = match_points(pts, tp.get(key, []))
        matched_pt += m
        dists += got
    out["points"] = {"n_ref": n_ref_pt, "n_test": sum(len(v) for v in tp.values()), "matched": matched_pt,
                     "max_dist_matched": max(dists) if dists else None}
    out["count"] = {"ref": ref_parsed["count_author"], "test": test_parsed["count_author"],
                    "equal": ref_parsed["count_author"] == test_parsed["count_author"],
                    "ref_shapes": ref_parsed["count_shapes"], "test_shapes": test_parsed["count_shapes"],
                    "equal_shapes": ref_parsed["count_shapes"] == test_parsed["count_shapes"],
                    "ref_stated": ref_parsed.get("count_stated"), "test_stated": test_parsed.get("count_stated"),
                    # stated count when the answer states one, else the number of shapes
                    "equal_stated_or_shapes": _stated_or_shapes(ref_parsed) == _stated_or_shapes(test_parsed)}
    rw, tw = " ".join(ref_parsed["words"]), " ".join(test_parsed["words"])
    out["ocr"] = {"ref_text": rw, "test_text": tw, "edits": levenshtein(rw, tw), "ref_len": len(rw),
                  "cer": (levenshtein(rw, tw) / len(rw)) if rw else (0.0 if not tw else 1.0)}
    out["none"] = {"ref": ref_parsed["is_none"], "test": test_parsed["is_none"],
                   "equal": ref_parsed["is_none"] == test_parsed["is_none"]}
    out["answer_equal"] = ref_parsed["answer"] == test_parsed["answer"]
    return out

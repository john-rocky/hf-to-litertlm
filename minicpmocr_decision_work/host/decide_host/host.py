"""The scoring host: picture path, token row, M-RoPE tables, tower and decision calls,
the pooled row, the head in numpy. No torch, torchvision or transformers. The backends pad and pick the bucket; the host never
pads.

    tower(patches [n, 1536], pos [n, 1024], cos [n, 64], sin [n, 64]) -> rows [n / 4, 2048]     tower.max_n
    decide(embeds [T, 2048], cos [T, 64], sin [T, 64], planes [3, T]) -> hidden [T, 2048]       decide.max_len

    h = Host(tower, decide, snapshot)
    h.score(image, render, text) -> {"status": "ok", "prob_1", "logits", "T", "n", ...}
    h.best_of(image, [(render, text), ...]) -> the rows of each candidate and the index of the highest prob_1; the original's
                                                tower rows are computed once
A refused request comes back as {"status": <why>, ...}: see REFUSE. A defect (the prompt template, the pooled id) raises.
contract.json at the repository root lists every input, output, token id and bucket."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from . import formulas as F
from . import picture as P

TEMPLATE_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"
TOKENIZER_SHA256 = "06b95093"                 # first 8 hex of the source's tokenizer.json, unchanged in this repository
MAX_ASPECT = 199.0
EMBED_KEY = "model.language_model.embed_tokens.weight"
POS_KEY = "model.visual.pos_embed.weight"
SCORE_KEY = "score.weight"
NOT_8_BIT = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F", "1;16")
REFUSE = ("literal_pad_token", "not_8_bit", "unreadable_picture", "too_large")


class Refused(Exception):
    def __init__(self, status: str, **info):
        super().__init__(status)
        self.doc = {"status": status, **info}


def _assert_template() -> None:
    """The author's template, its placeholders not expanded: score.py's md5 gate."""
    s = (f"<|im_start|>user\n{F.IMAGE_PLACEHOLDER}\nOCR文本：\n{{text}}\n{F.IMAGE_PLACEHOLDER}\n"
         f"{F.INSTRUCTION}{F.ASSISTANT_SUFFIX}")
    got = hashlib.md5(s.encode("utf-8")).hexdigest()
    if got != TEMPLATE_MD5:
        raise RuntimeError(f"prompt template md5 {got}, want {TEMPLATE_MD5}")


_assert_template()


def load_picture(image) -> np.ndarray:
    """uint8 RGB [H, W, 3] of a path or a PIL image: the author's pad of an extreme aspect, no EXIF step, no resize."""
    try:
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        w, h = img.size
        mode = img.mode
        if w <= 0 or h <= 0:
            raise Refused("unreadable_picture", why="a side of 0")
        if mode in NOT_8_BIT:
            raise Refused("not_8_bit", mode=mode)
        # Pillow opens a 16-bit RGB, RGBA or gray + alpha PNG in an 8-bit mode and keeps the high bytes; only the raw mode tells
        # (a picture that is already decoded has no tile: pass a path or an unloaded image).
        raw = [t[3] for t in getattr(img, "tile", ()) if len(t) > 3 and isinstance(t[3], str) and ";16" in t[3]]
        if raw:
            raise Refused("not_8_bit", mode=mode, raw=raw[0])
        rgb = img.convert("RGB")
    except Refused:
        raise
    except Exception as e:
        raise Refused("unreadable_picture", why=f"{type(e).__name__}: {e}"[:200]) from e
    if max(w, h) / min(w, h) > MAX_ASPECT:
        if w >= h:
            nw, nh = w, max(h, int(w / MAX_ASPECT + 0.999999))
        else:
            nw, nh = max(w, int(h / MAX_ASPECT + 0.999999)), h
        canvas = Image.new("RGB", (nw, nh), "white")
        canvas.paste(rgb, ((nw - w) // 2, (nh - h) // 2))
        rgb = canvas
    return np.asarray(rgb, dtype=np.uint8)


@dataclass
class Picture:
    gh: int
    gw: int
    n: int
    patches: np.ndarray    # [n, 1536]
    pos: np.ndarray        # [n, 1024]
    cos: np.ndarray        # [n, 64]
    sin: np.ndarray        # [n, 64]


class Host:
    def __init__(self, tower, decide, snapshot, tables=None):
        """`snapshot` holds tokenizer.json. `tables` is a safetensors file with the three bf16 tables (this repository's
        tables/tables.safetensors); by default the source's model.safetensors in `snapshot`. Read by header offset, bytes as they are."""
        snapshot = Path(snapshot)
        tok_path = snapshot / "tokenizer.json"
        got = hashlib.sha256(tok_path.read_bytes()).hexdigest()
        if not got.startswith(TOKENIZER_SHA256):
            raise RuntimeError(f"tokenizer.json sha256 {got}, want {TOKENIZER_SHA256}...")
        self.tokenizer_sha256 = got
        self.tok = Tokenizer.from_file(str(tok_path))
        src = tables or snapshot / "model.safetensors"
        self.embed = F.Bf16Table(src, EMBED_KEY)
        self.pos_table = F.Bf16Table(src, POS_KEY).rows()
        self.score_w = F.Bf16Table(src, SCORE_KEY).rows()
        assert self.embed.shape == (248320, 2048) and self.pos_table.shape == (2304, 1024) and self.score_w.shape == (2, 2048)
        self.tower, self.decide = tower, decide

    def picture(self, image) -> Picture:
        u8 = load_picture(image)
        H, W = P.smart_resize(*u8.shape[:2])
        patches = P.to_pixel_values(P.resize_uint8(u8, H, W))
        gh, gw = H // P.PATCH, W // P.PATCH
        cos, sin = F.vision_rotary(gh, gw)
        return Picture(gh, gw, gh * gw, patches, F.pos_rows(self.pos_table, gh, gw), cos, sin)

    def token_row(self, text: str, m1: int, m2: int) -> np.ndarray:
        ids = np.asarray(self.tok.encode(F.build_prompt(text, m1, m2), add_special_tokens=False).ids, dtype=np.int64)
        if ids[-1] != F.POOLED:
            raise RuntimeError(f"pooled id {int(ids[-1])}, want {F.POOLED}")
        return ids

    def rows_of(self, pic: Picture) -> np.ndarray:
        return self.tower(pic.patches, pic.pos, pic.cos, pic.sin)

    def _score(self, a: Picture, b: Picture, text: str, rows_a=None, rows_b=None, ms=None) -> dict:
        ms = {} if ms is None else ms
        if "<|image_pad|>" in text or "<|video_pad|>" in text:
            raise Refused("literal_pad_token")
        t0 = time.perf_counter()
        ids = self.token_row(text, a.n // 4, b.n // 4)
        T = len(ids)
        limit = {"max_n": getattr(self.tower, "max_n", None), "max_len": getattr(self.decide, "max_len", None)}
        if (limit["max_n"] and max(a.n, b.n) > limit["max_n"]) or (limit["max_len"] and T > limit["max_len"]):
            raise Refused("too_large", n=[a.n, b.n], T=T, limit=limit)
        ms["tokens"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        tower_ms = []
        for name, pic, rows in (("a", a, rows_a), ("b", b, rows_b)):
            if rows is None:
                ts = time.perf_counter()
                rows = self.rows_of(pic)
                tower_ms.append((time.perf_counter() - ts) * 1000)
                if name == "a":
                    rows_a = rows
                else:
                    rows_b = rows
        ms["tower"] = tower_ms
        t2 = time.perf_counter()
        embeds = F.fill_embeds(ids, T, self.embed, [rows_a, rows_b])[0][0]
        pos3 = F.mrope_planes(ids, [(a.gh, a.gw), (b.gh, b.gw)])
        cos, sin = F.mrope_tables(pos3)
        ms["rows_and_tables"] = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()
        hidden = self.decide(embeds, cos[0], sin[0], pos3)
        ms["decide"] = (time.perf_counter() - t3) * 1000
        t4 = time.perf_counter()
        pooled = T - 1
        logits, p1 = F.softmax_prob1(hidden[pooled], self.score_w)
        logits_prev, p1_prev = F.softmax_prob1(hidden[pooled - 1], self.score_w)
        ms["head"] = (time.perf_counter() - t4) * 1000
        ms["tower_and_host_gap"] = (t2 - t1) * 1000 - sum(tower_ms)
        return {"status": "ok", "prob_1": p1, "logits": [float(x) for x in logits], "prob_1_prev": p1_prev,
                "logits_prev": [float(x) for x in logits_prev], "T": T, "N": [a.n, b.n], "n": [a.n, b.n],
                "grids": [[a.gh, a.gw], [b.gh, b.gw]], "pooled_id": int(ids[pooled]),
                "ids_sha": hashlib.sha256(ids.tobytes()).hexdigest()[:16], "hidden_pooled": hidden[pooled].copy(),
                "ms": ms, "_rows": (rows_a, rows_b)}

    def score(self, image, render, text: str) -> dict:
        ms = {}
        try:
            t0 = time.perf_counter()
            a, b = self.picture(image), self.picture(render)
            ms["pictures"] = (time.perf_counter() - t0) * 1000
            out = self._score(a, b, text, ms=ms)
        except Refused as r:
            return r.doc
        out.pop("_rows")
        ms["total"] = (time.perf_counter() - t0) * 1000
        return out

    def best_of(self, image, candidates: list) -> dict:
        """candidates = [(render, text), ...]. The original goes through the tower once."""
        ms = {}
        t0 = time.perf_counter()
        try:
            a = self.picture(image)
        except Refused as r:
            return r.doc
        rows_a = None
        out = []
        for render, text in candidates:
            try:
                b = self.picture(render)
                one = self._score(a, b, text, rows_a=rows_a)
            except Refused as r:
                out.append(r.doc)
                continue
            rows_a = one.pop("_rows")[0]
            out.append(one)
        ok = [(r["prob_1"], i) for i, r in enumerate(out) if r["status"] == "ok"]
        ms["total"] = (time.perf_counter() - t0) * 1000
        return {"status": "ok" if ok else "refused", "best": max(ok)[1] if ok else None, "candidates": out, "ms": ms}

    def close(self) -> None:
        for backend in (self.decide, self.tower):
            if backend is not None and hasattr(backend, "close"):
                backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

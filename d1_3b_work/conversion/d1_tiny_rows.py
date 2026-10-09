"""Stand-ins that let the host run end to end on the round-2 tiny graphs (no torch here; host dependencies only).

The tiny graphs have a 256-token vocabulary, so the host's real token ids (up to 125,016) are folded into it:
`TinyTokenizer.encode` returns [i % 255 for i in the real ids] (0..254; 255 is the tiny pad id and never a real token).
The host derives option codes and read-out groups from the same tokenizer, so they fold the same way (a fold can merge
two codes; the host then picks fallback codes, the same on both sides of the comparison). The read-out table is random
float32 rows [255, d] (seed 1, N(0, 1)) for ids 0..254, written once to cache/tiny/readout_tiny.safetensors.
E2E_RECORDS = the fixture requests of the end-to-end test: every row fits the tiny graphs (L64 / L256), the three
question types, one-question and several-question requests, a row of exactly 64 tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import d1_litert as H  # noqa: E402

TINY_MOD = 255
TINY_PAD_ID = 255
TABLE = K / "cache/tiny/readout_tiny.safetensors"
TABLE_SEED = 1
E2E_RECORDS = ("card_text_001", "tv4x_emotion_00", "tv4x_qnli_07", "tv4_000", "tv4s_00", "own_fiveq_09", "own_sensor_08")


class TinyTokenizer(H.D1Tokenizer):
    def encode(self, text: str):
        return [i % TINY_MOD for i in super().encode(text)]


def tokenizer() -> TinyTokenizer:
    return TinyTokenizer(K / "hf_small" / H.TOKENIZER_FILE)


def requests() -> list[tuple[str, dict]]:
    recs = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    return [(rid, {"state": recs[rid]["request"]["state"], "questions": recs[rid]["request"]["questions"]})
            for rid in E2E_RECORDS]


def make_table(d: int) -> dict:
    from safetensors.numpy import save_file

    assert not TABLE.exists(), f"refusing to overwrite {TABLE}"
    rows = np.random.default_rng(TABLE_SEED).standard_normal((TINY_MOD, d)).astype(np.float32)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    save_file({"ids": np.arange(TINY_MOD, dtype=np.int64), "rows": rows}, str(TABLE))
    return {"file": str(TABLE.relative_to(K)), "shape": list(rows.shape)}


def table() -> H.ReadoutTable:
    return H.ReadoutTable.from_file(TABLE)

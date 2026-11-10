#!/usr/bin/env python3
"""Dimensions of tencent/WeVisOne-VL-4B, read from the pinned snapshot's config.json.

The WeVisOne-VL-2B tools (wevisone_work/tools/) hold the 2B dimensions as literals. The copies in this directory read
them from here instead, so the snapshot decides them all:

    WEVISONE_SNAPSHOT=<snapshot dir>   (default: the Hugging Face cache path of tencent/WeVisOne-VL-4B at 4a7bed92)

Derived values (docs/deepstack-wide-embedding.md):
    HID        text hidden size                               (2B 2048, 4B 2560)
    WIDE       4 * HID + 4 = width of an image row / embedding (2B 8196, 4B 10244)
    FLAG       4 * HID = the image-token flag channel          (2B 8192, 4B 10240)
    MLP        text intermediate size                         (2B 6144, 4B 9728)
    LAYERS     text decoder layers                            (2B 28, 4B 36)
    KV_STATES  2 * LAYERS = K + V state buffers               (2B 56, 4B 72)
    OUT_HID    vision merger output (out_hidden_size) = HID, asserted

    [WEVISONE_SNAPSHOT=DIR] python3 tools/wv_dims.py [--json]
"""
import json
import os
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]                       # wevisone4b_work
REPO_ID = "tencent/WeVisOne-VL-4B"
REVISION = "4a7bed92750ac9d8b8a124cede14759036932dc2"
SNAP = Path(os.environ.get("WEVISONE_SNAPSHOT", "~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-4B/"
                           "snapshots/" + REVISION)).expanduser()
# the WeVisOne-VL-2B test pictures (wevisone_work/fixtures/fetch_public.sh writes fixtures/img/), read only
FIX = K.parent / "wevisone_work" / "fixtures"
# tokenizer.json of the snapshot (the same file as WeVisOne-VL-2B's at 3dad9ce8)
TOKENIZER_SHA256 = "3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e"

CFG = json.loads((SNAP / "config.json").read_text())
TEXT = CFG["text_config"]
VISION = CFG["vision_config"]

HID = int(TEXT["hidden_size"])
LAYERS = int(TEXT["num_hidden_layers"])
HEADS = int(TEXT["num_attention_heads"])
KV_HEADS = int(TEXT["num_key_value_heads"])
HEAD_DIM = int(TEXT["head_dim"])
MLP = int(TEXT["intermediate_size"])
VOCAB = int(TEXT["vocab_size"])
MROPE_SECTION = tuple(int(x) for x in TEXT["rope_parameters"]["mrope_section"])
ROPE_THETA = float(TEXT["rope_parameters"]["rope_theta"])
WIDE = 4 * HID + 4
FLAG = 4 * HID
KV_STATES = 2 * LAYERS

V_HID = int(VISION["hidden_size"])
V_DEPTH = int(VISION["depth"])
V_HEADS = int(VISION["num_heads"])
V_MLP = int(VISION["intermediate_size"])
V_PATCH = int(VISION["patch_size"])
V_MERGE = int(VISION["spatial_merge_size"])
V_TEMPORAL = int(VISION["temporal_patch_size"])
DEEPSTACK = tuple(int(x) for x in VISION["deepstack_visual_indexes"])
OUT_HID = int(VISION["out_hidden_size"])
POS_EMB = int(VISION["num_position_embeddings"])
assert OUT_HID == HID, (OUT_HID, HID)          # the wide layout puts merger outputs in HID-wide slots
assert len(DEEPSTACK) == 3, DEEPSTACK           # three DeepStack slots, added after decoder layers 0, 1, 2
assert TEXT["rope_parameters"].get("mrope_interleaved") is True
assert not CFG.get("tie_word_embeddings") and not TEXT.get("tie_word_embeddings")


def summary():
    return {"snap": str(SNAP), "revision": SNAP.name, "HID": HID, "WIDE": WIDE, "FLAG": FLAG, "MLP": MLP,
            "LAYERS": LAYERS, "KV_STATES": KV_STATES, "HEADS": HEADS, "KV_HEADS": KV_HEADS, "HEAD_DIM": HEAD_DIM,
            "VOCAB": VOCAB, "MROPE_SECTION": list(MROPE_SECTION), "ROPE_THETA": ROPE_THETA, "OUT_HID": OUT_HID,
            "V_HID": V_HID, "V_DEPTH": V_DEPTH, "V_HEADS": V_HEADS, "V_MLP": V_MLP, "V_PATCH": V_PATCH,
            "V_MERGE": V_MERGE, "V_TEMPORAL": V_TEMPORAL, "DEEPSTACK": list(DEEPSTACK), "POS_EMB": POS_EMB}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=None if "--json" in sys.argv else 1))

"""Shared paths and loaders for the d1-omni lane scripts (reference venv side).

The provider's code is read from the pinned HF snapshot, never edited: `prompt.py` and `encoder.py` are imported as
plain modules (neither has a relative import); `modeling_d1.py` (relative imports) only through transformers'
remote-code loader.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Importing the provider's .py files straight from the HF snapshot would leave __pycache__/ inside the shared cache
# (seen in round 1, removed); no bytecode files from any import in these scripts.
sys.dont_write_bytecode = True

REPO = "LiquidAI/d1-omni-600M"
REV = "414f8d6438174f5b2133a9c21a478fc42625e308"
K = Path(__file__).resolve().parents[1]
SNAP = Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-omni-600M/snapshots" / REV
WEIGHTS = SNAP / "model.safetensors"

# token ids the launch measured from tokenizer.json (asserted against convert_tokens_to_ids in step 2)
TOKEN_IDS = {"<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 7, "<|mask|>": 16, "<|reserved_7|>": 17,
             "<|reserved_8|>": 18, "<|reserved_9|>": 19, "<|reserved_10|>": 20, "<|reserved_11|>": 21}
ROLE = {"<|pad|>": "pad", "<|startoftext|>": "bos", "<|im_end|>": "eos", "<|mask|>": "marker",
        "<|reserved_7|>": "state", "<|reserved_8|>": "q", "<|reserved_9|>": "opt", "<|reserved_10|>": "opt_end",
        "<|reserved_11|>": "decide"}

# modeling_d1.py constants (read from the file; asserted equal to the imported config in ref_probs.py)
MAX_LENGTH, IMAGE_TEXT_LENGTH, AUDIO_TEXT_LENGTH = 16384, 896, 15360
YES_NO = {"false": "no", "true": "yes"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def provider_prompt():
    return load_module("d1_provider_prompt", SNAP / "prompt.py")


def provider_encoder():
    return load_module("d1_provider_encoder", SNAP / "encoder.py")


def config():
    return json.loads((SNAP / "config.json").read_text())


def tokenizer():
    from transformers import AutoTokenizer

    # trust_remote_code=True as modeling_d1.D1OmniModel.tokenizer does (the config's auto_map names its code;
    # without it transformers asks on stdin)
    return AutoTokenizer.from_pretrained(REPO, revision=REV, trust_remote_code=True)


def mode_of(record):
    """The provider's per-request settings (modeling_d1.probabilities_batch), from the record's media kind."""
    media = record.get("media")
    kind = None if media is None else media["kind"]
    if kind is None:
        return {"mode": "text", "max_len": MAX_LENGTH, "noul_default": None, "calibrate": True, "audio": False}
    if kind == "image":
        return {"mode": "image", "max_len": IMAGE_TEXT_LENGTH, "noul_default": YES_NO, "calibrate": False,
                "audio": False}
    if kind == "audio":
        return {"mode": "audio", "max_len": AUDIO_TEXT_LENGTH, "noul_default": YES_NO, "calibrate": False,
                "audio": True}
    raise ValueError(kind)


def state_of(record, mode):
    """modeling_d1: audio turns a None state into {}; then None -> ''."""
    state = record["request"]["state"]
    if mode["audio"] and state is None:
        state = {}
    return "" if state is None else state

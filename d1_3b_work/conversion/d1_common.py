"""Shared paths, the provider's code and the provider's prompt settings for the d1-3B LiteRT conversion.

The provider's files (LiquidAI/d1-3B at REV) sit in `hf_small/` (everything but the weights). Their modules use
relative imports, so they are loaded as one synthetic package, `d1_provider`, whose path is that folder; byte-code
writing is switched off first so the folder stays exactly the Hub's files.

`engine_settings` gives the arguments `runner.SystemOne.__init__` passes to `prompt.render` (`D1Model.engine` builds
SystemOne with the defaults): bos = the tokenizer's `bos_token`, lead = `default_lead(config.model_type)`, state style
`DEFAULT_STATE_STYLE`, system `DEFAULT_SYSTEM`, option style "desc".
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

K = Path(__file__).resolve().parents[1]
REPO = "LiquidAI/d1-3B"
REV = "da1fe36a861f24690f27f622dca1d8688503d113"
HF_SMALL = K / "hf_small"
FIXTURES = K / "fixtures/requests.json"
ROWS = K / "fixtures/rows.json"
PROBES = K / "fixtures/token_probes.json"
MODEL_TYPE = "lfm2_vl"   # config.json model_type; SystemOne reads it for the lead
PACKAGE = "d1_provider"
TOKENIZER_SHA256 = "8096ecb9f54599d756c8de728a598a340bc1e43c0deb77ddd62456c38349fcee"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def provider(module: str, src: Path = HF_SMALL) -> types.ModuleType:
    """`prompt`, `api` (standard library only) or `hybrid`, `lfm2_vl`, `runner`, `modeling_d1` (torch, transformers)."""
    sys.dont_write_bytecode = True
    pkg = sys.modules.get(PACKAGE)
    if pkg is None:
        pkg = types.ModuleType(PACKAGE)
        pkg.__path__ = [str(src)]
        sys.modules[PACKAGE] = pkg
    assert list(pkg.__path__) == [str(src)], (pkg.__path__, src)
    return importlib.import_module(f"{PACKAGE}.{module}")


def load_tokenizer(src: Path = HF_SMALL):
    """The tokenizer `D1Model.engine` loads: `AutoTokenizer.from_pretrained(<model dir>)`."""
    from transformers import AutoTokenizer

    assert sha256_file(src / "tokenizer.json") == TOKENIZER_SHA256
    return AutoTokenizer.from_pretrained(str(src))


def engine_settings(tokenizer, prompt: types.ModuleType) -> dict:
    """What SystemOne(model=..., tokenizer=...) passes to prompt.render, with its defaults."""
    bos = getattr(tokenizer, "bos_token", None)
    return {"bos": bos if isinstance(bos, str) else "", "lead": prompt.default_lead(MODEL_TYPE),
            "style": prompt.DEFAULT_STATE_STYLE, "system": prompt.DEFAULT_SYSTEM, "option_style": "desc"}


def render_row(prompt: types.ModuleType, tokenizer, settings: dict, state: Any, q, images: str = "") -> str:
    """`SystemOne.render(state, q)` (images="") or the text of `SystemOne._request` for one question with pictures."""
    s = settings
    return prompt.render(tokenizer, state, q, s["bos"], s["lead"], s["style"], s["system"], s["option_style"], images)


def split_texts(prompt: types.ModuleType, tokenizer, settings: dict, state: Any, q, images: str = "") -> tuple[str, str]:
    """The trunk and branch texts `SystemOne._request` encodes separately for a request of several questions."""
    s = settings
    prefix = prompt.prefix_text(tokenizer, state, s["bos"], s["style"], s["system"], images)
    suffix = prompt.suffix_text(tokenizer, q, s["lead"], s["option_style"])
    return prefix, suffix


def encode(tokenizer, text: str) -> list[int]:
    """Every encode in the provider's code: `tokenizer.encode(text, add_special_tokens=False)`."""
    return tokenizer.encode(text, add_special_tokens=False)


def load_fixtures(path: Path = FIXTURES) -> dict:
    return json.loads(path.read_text())


def questions_of(prompt: types.ModuleType, record: dict) -> list[tuple[str, Any, str | None]]:
    """[(qid, Question or None, error)] in request order; `prompt.as_question` is the provider's parser."""
    out = []
    for qid, q in record["request"]["questions"].items():
        try:
            out.append((qid, prompt.as_question(q), None))
        except Exception as e:   # the provider's schema: instructions and criteria are required
            out.append((qid, None, f"{type(e).__name__}: {e}"))
    return out

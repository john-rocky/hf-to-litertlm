"""The provider's tokenizer on the `tokenizers` library alone.

d1-2B's engine holds `AutoTokenizer.from_pretrained(snapshot)`, which in transformers 5.19.0 is a `TokenizersBackend`
over the snapshot's tokenizer.json (its backend JSON equals `Tokenizer.from_file(tokenizer.json)`'s; test_host.py checks
it). `TokenizerAdapter` offers what the provider's prompt / runner / media / pointer / attention code calls on it, with
the same results:

- `encode(text, add_special_tokens=...)` -> ids;
- `tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)` -> {"input_ids", "attention_mask",
  "offset_mapping"} (character offsets, as the fast tokenizer returns them);
- `decode(ids, skip_special_tokens=False)` (no clean-up of spaces: `clean_up_tokenization_spaces` is False);
- `bos_token`, `eos_token`, `pad_token`, `pad_token_id`, `all_special_tokens`, `added_tokens_decoder` (id ->
  `tokenizers.AddedToken`, whose `content` / `special` `prompt._specials_cached` reads), the processor's token names
  (`image_token`, `video_token`, `vision_start_token`, `vision_end_token`) and `convert_tokens_to_ids`;
- the `d1_*` attributes the engine sets (`d1_codes`, `d1_numbers`, `d1_box_edges`, `d1_prompt`; D1Host sets them).

`apply_chat_template` is not offered: the provider reads it only for a model type with no registered lead
(`prompt.template_lead`), and lfm3_vl has one.

The adapter has no attribute named `tokenizer`: `prompt._specials_cached` would take it for a processor's own.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence


def _content(value: Any) -> str | None:
    """A special token as tokenizer_config.json writes it: a string, or an AddedToken dict with its `content`."""
    if isinstance(value, dict):
        return value.get("content")
    return value


class TokenizerAdapter:
    def __init__(self, snapshot_dir: str):
        from tokenizers import Tokenizer

        self.name_or_path = snapshot_dir
        self._backend = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
        with open(os.path.join(snapshot_dir, "tokenizer_config.json")) as f:
            config = json.load(f)
        self.bos_token = _content(config.get("bos_token"))
        self.eos_token = _content(config.get("eos_token"))
        self.unk_token = _content(config.get("unk_token"))
        self.pad_token = _content(config.get("pad_token"))
        self.clean_up_tokenization_spaces = bool(config.get("clean_up_tokenization_spaces", False))
        extra = config.get("extra_special_tokens") or []
        extra = list(extra.values()) if isinstance(extra, dict) else list(extra)
        specific = config.get("model_specific_special_tokens") or {}
        for name, token in specific.items():  # image_token, video_token, vision_start_token, vision_end_token
            setattr(self, name, _content(token))
        named = [self.bos_token, self.eos_token, self.unk_token, self.pad_token]
        self.all_special_tokens = list(dict.fromkeys(
            [t for t in named if t] + [_content(t) for t in extra] + [_content(t) for t in specific.values()]))
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token) if self.pad_token else None
        self.bos_token_id = self.convert_tokens_to_ids(self.bos_token) if self.bos_token else None
        # the engine's settings (`SlotReader._serve` sets them on the tokenizer); D1Host overwrites them
        self.d1_codes, self.d1_numbers, self.d1_box_edges, self.d1_prompt = 1, False, "median", 1

    # ------------------------------------------------------------------ vocabulary

    @property
    def added_tokens_decoder(self) -> dict:
        return self._backend.get_added_tokens_decoder()

    def convert_tokens_to_ids(self, tokens: str | Iterable[str]):
        if isinstance(tokens, str):
            return self._backend.token_to_id(tokens)
        return [self._backend.token_to_id(t) for t in tokens]

    def __len__(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    # ------------------------------------------------------------------ encode / decode

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._backend.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False,
                 **unused) -> dict:
        if not isinstance(text, str):
            raise TypeError("the host's tokenizer takes one text at a time")
        if unused:
            raise TypeError(f"unsupported tokenizer arguments: {sorted(unused)}")
        enc = self._backend.encode(text, add_special_tokens=add_special_tokens)
        out = {"input_ids": enc.ids, "attention_mask": enc.attention_mask}
        if return_offsets_mapping:
            out["offset_mapping"] = [tuple(o) for o in enc.offsets]
        return out

    def decode(self, ids: int | Sequence[int], skip_special_tokens: bool = False, **unused) -> str:
        if isinstance(ids, int):
            ids = [ids]
        text = self._backend.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)
        if self.clean_up_tokenization_spaces:
            raise NotImplementedError("clean_up_tokenization_spaces is set in tokenizer_config.json: not ported")
        return text

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("the host renders d1's own prompt; a model type with no registered lead "
                                  "(prompt.LEADS) would need the chat template, which the host does not read")

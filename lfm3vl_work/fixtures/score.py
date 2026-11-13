#!/usr/bin/env python3
"""Scoring rules of the LFM3-VL-2B fixtures (fixed before any converted form was run).

same32(a, b): the first min(32, len a, len b) generated token ids are equal, and when either side has
    fewer than 32 ids both have the same length (one stopping on EOS while the other goes on = different).
answer_part(text): the text after the last "</think>" if there is one, else the whole text.
correct(text, expect): answer_part lowercased, number words zero..twenty -> digits, then expect (lowercased)
    searched as a regex with word boundaries; any hit = correct. expect None = not scored.
text gate: the 8 QUESTIONS of scripts/verify_quality.py (the question + " Answer briefly.", scored by its
    answer_ok on its strip_think) + France: "What is the capital of France? Answer in one word." -> \\bparis\\b.
"""
import importlib.util
import os
import re

N32 = 32
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
       "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
FRANCE = ("france=Paris", "What is the capital of France? Answer in one word.", r"\bparis\b")


def _load_vq():
  p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "verify_quality.py")
  spec = importlib.util.spec_from_file_location("lfm3vl_verify_quality", p)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_VQ = _load_vq()
TEXT_QUESTIONS = [(lab, q + _VQ.SUFFIX, pat) for lab, q, pat in _VQ.QUESTIONS]   # the 8
TEXT_ROWS = TEXT_QUESTIONS + [FRANCE]                                             # + France


def same32(a, b):
  a, b = list(a), list(b)
  n = min(N32, len(a), len(b))
  if a[:n] != b[:n]:
    return False
  if len(a) < N32 or len(b) < N32:
    return len(a) == len(b)
  return True


EOS = 124900


def split_eos(ids):
  """(content ids without a trailing EOS, ended_on_eos)."""
  ids = list(ids)
  return (ids[:-1], True) if ids and ids[-1] == EOS else (ids, False)


def same32_eos(a_ids, a_eos, b_ids, b_eos):
  """same32 for sequences whose EOS is a flag (the runtime strips its stop token from the text it returns):
  content ids compared on the first min(32, ...); a side shorter than 32 must match the other's length and EOS."""
  a, b = list(a_ids), list(b_ids)
  n = min(N32, len(a), len(b))
  if a[:n] != b[:n]:
    return False
  if len(a) < N32 or len(b) < N32:
    return len(a) == len(b) and bool(a_eos) == bool(b_eos)
  return True


def answer_part(text):
  return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def normalize(text):
  t = answer_part(text).lower()
  for i, w in enumerate(NUM):
    t = re.sub(rf"\b{w}\b", str(i), t)
  return t


def correct(text, expect):
  if expect is None:
    return None
  return re.search(rf"\b{re.escape(expect.lower())}\b", normalize(text)) is not None


def text_pass(text, pattern):
  return _VQ.answer_ok(pattern, _VQ.strip_think(text))

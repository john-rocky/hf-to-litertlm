#!/usr/bin/env python3
"""Desk renderer for the LiteRT-LM conversation prefix contract (no engine, no weights).

Renders a bundle's chat template the way litert-lm 0.18.0 does before every message
(runtime/conversation/conversation.cc, runtime/components/prompt_template.cc,
runtime/components/rust/minijinja_template.rs; tag v0.18.0):
  * the template source first goes through EditTemplateForMinijinja (regex rewrites of
    Python idioms, prompt_template.cc l.40-74);
  * the environment keeps trailing newlines, trims blocks and lstrips blocks
    (minijinja_template.rs l.182-184) and adds strftime_now, raise_exception, tojson
    (", " / ": " spacing), lstrip, rstrip and a `none` test that is true for an
    undefined value only (l.185-190);
  * the context is messages, tools (only when not null), add_generation_prompt, now,
    bos_token = eos_token = "" (prompt_template.h l.92-98), then the extra_context keys
    (enable_thinking) at top level (minijinja_template.rs l.235-255);
  * every message goes through NormalizeMessageContent first (string content -> one text
    part; data_utils.cc l.66-84), and a reply carries `channels` and `reasoning_content`
    next to its content (channel_util.cc l.114-139).

Two engines, same inputs:
  jinja2     jinja2 (ImmutableSandboxedEnvironment) with the runtime's filters / tests added,
             plus minijinja's `split`, `items`, `startingwith`, `endingwith`; Python methods on
             str / dict / list raise (minijinja has no methods without pycompat).
  minijinja  minijinja-py, the same crate version as the runtime (=2.14.0, Cargo.toml l.19),
             pycompat off; `is none` / `is not none` are rewritten to `is undefined` /
             `is not undefined` because the Python binding hands undefined and none to a
             Python test as the same None (the runtime's `none` test is is_undefined).
"""
import datetime
import json
import re

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

# 2026-01-01T00:00:00Z: a fixed `now` so that strftime_now renders the same every time.
NOW_TS = 1767225600

# prompt_template.cc v0.18.0 l.42-72, in order (RE2::GlobalReplace == re.sub here: no
# pattern uses a construct the two engines read differently).
MINIJINJA_EDITS = (
    (r"\.startswith\((.*?)\)", r" is startingwith \1"),
    (r"\.endswith\((.*?)\)", r" is endingwith \1"),
    (r"\.replace\((.*?),(.*?)\)", r" | replace(\1,\2)"),
    (r"\.split\((.*?)\)\[0\]", r" | split(\1) | first"),
    (r"\.split\((.*?)\)\[-1\]", r" | split(\1) | last"),
    (r"\.split\((.*?)\)", r" | split(\1)"),
    (r"\.join\((.*?)\)", r" | join(\1)"),
    (r"\.lstrip\(\)", " | lstrip"),
    (r"\.lstrip\((.*?)\)", r" | lstrip(\1)"),
    (r"\.rstrip\(\)", " | rstrip"),
    (r"\.rstrip\((.*?)\)", r" | rstrip(\1)"),
    (r"\.strip\(\)", " | trim"),
    (r"\.strip\((.*?)\)", r" | trim(\1)"),
    (r"\.items\(\)", " | items"),
    (r"{% generation %}", ""),
    (r"{% endgeneration %}", ""),
)


def edit_for_minijinja(src):
  for pat, rep in MINIJINJA_EDITS:
    src = re.sub(pat, rep, src)
  return src


def tojson_space(value, *args, **kwargs):
  """The runtime's tojson (minijinja_template.rs l.109-151): no extra arguments."""
  if args or kwargs:
    raise jinja2.TemplateError("tojson: too many arguments")
  if isinstance(value, jinja2.Undefined):
    raise jinja2.TemplateError("tojson: undefined value")
  return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _lstrip(s, chars=None):
  return str(s).lstrip(chars)


def _rstrip(s, chars=None):
  return str(s).rstrip(chars)


def _raise_exception(msg):
  raise jinja2.TemplateError(str(msg))


def _strftime_now(fmt):
  return datetime.datetime.fromtimestamp(NOW_TS, datetime.timezone.utc).strftime(fmt)


# ---------------------------------------------------------------- jinja2 engine

class _MinijinjaLikeEnv(ImmutableSandboxedEnvironment):
  """minijinja has no methods on str / map / seq without pycompat: refuse them here too."""

  def call(__self, __context, __obj, *args, **kwargs):  # noqa: N805 (jinja2 signature)
    owner = getattr(__obj, "__self__", None)
    if isinstance(owner, (str, dict, list, tuple)):
      raise jinja2.TemplateError(
          f"unknown method: {type(owner).__name__} has no method named "
          f"{getattr(__obj, '__name__', '?')}")
    return super().call(__context, __obj, *args, **kwargs)


def _split(s, sep=None, maxsplits=None):
  s = str(s)
  if maxsplits is None:
    return s.split(sep)
  return s.split(sep, int(maxsplits))


def _items(v):
  if isinstance(v, dict):
    return list(v.items())
  raise jinja2.TemplateError("items: value is not a map")


def make_jinja2_env(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True):
  env = _MinijinjaLikeEnv(trim_blocks=trim_blocks, lstrip_blocks=lstrip_blocks,
                          keep_trailing_newline=keep_trailing_newline)
  env.globals["raise_exception"] = _raise_exception
  env.globals["strftime_now"] = _strftime_now
  env.filters["tojson"] = tojson_space
  env.filters["lstrip"] = _lstrip
  env.filters["rstrip"] = _rstrip
  env.filters["split"] = _split
  env.filters["items"] = _items
  env.tests["none"] = lambda v: isinstance(v, jinja2.Undefined)
  env.tests["startingwith"] = lambda s, p: str(s).startswith(str(p))
  env.tests["endingwith"] = lambda s, p: str(s).endswith(str(p))
  return env


# ------------------------------------------------------------- minijinja engine

_NONE_TEST = re.compile(r"\bis(\s+not)?\s+none\b")


def none_test_to_undefined(src):
  return _NONE_TEST.sub(lambda m: "is" + (m.group(1) or "") + " undefined", src)


def make_minijinja_env(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True):
  import minijinja  # minijinja-py; only the venv that has it can use this engine
  env = minijinja.Environment()
  env.pycompat = False
  env.trim_blocks = trim_blocks
  env.lstrip_blocks = lstrip_blocks
  env.keep_trailing_newline = keep_trailing_newline
  env.add_function("raise_exception", _raise_exception)
  env.add_function("strftime_now", _strftime_now)
  env.add_filter("tojson", tojson_space)
  env.add_filter("lstrip", _lstrip)
  env.add_filter("rstrip", _rstrip)
  return env


def have_minijinja():
  try:
    import minijinja  # noqa: F401
    return True
  except ImportError:
    return False


class Renderer:
  """render(messages, add_generation_prompt, tools=None, extra_context=None) -> str."""

  def __init__(self, source, engine="jinja2", trim_blocks=True, lstrip_blocks=True,
               keep_trailing_newline=True, edit=True):
    self.engine = engine
    src = edit_for_minijinja(source) if edit else source
    if engine == "jinja2":
      self._env = make_jinja2_env(trim_blocks, lstrip_blocks, keep_trailing_newline)
      self._tmpl = self._env.from_string(src)
    elif engine == "minijinja":
      self._env = make_minijinja_env(trim_blocks, lstrip_blocks, keep_trailing_newline)
      self._env.add_template("template", none_test_to_undefined(src))
    else:
      raise ValueError(engine)

  def render(self, messages, add_generation_prompt, tools=None, extra_context=None):
    ctx = {"messages": messages}
    if tools is not None:
      ctx["tools"] = tools
    ctx["add_generation_prompt"] = bool(add_generation_prompt)
    ctx["now"] = NOW_TS
    ctx["bos_token"] = ""
    ctx["eos_token"] = ""
    for k, v in (extra_context or {}).items():
      ctx[k] = v
    if self.engine == "jinja2":
      return self._tmpl.render(**ctx)
    return self._env.render_template("template", **ctx)


# ------------------------------------------------------------ message shapes

def parts(text):
  return [{"type": "text", "text": text}]


def user_msg(text, form="rt"):
  if form == "v0171":
    return {"role": "user", "content": text}
  return {"role": "user", "content": parts(text)}


def assistant_msg(content, thought=None, form="rt"):
  """One assistant reply in one of the history shapes.

  rt     the 0.18.0 runtime message: content parts + channels + reasoning_content
         (channels only when the reply had channel text; channel_util.cc l.114-139)
  i      the thought inline in the content: "<think>\\n{thought}\\n</think>\\n\\n{content}"
         (content with its leading newlines removed), content parts
  ii     content parts + reasoning_content
  iii    content parts only
  v0171  what 0.17.1 handed the template: {role, content: string}
         (GenericDataProcessor::MessageToTemplateInput v0.17.1 l.95-116 drops every
         other key when the content is one text part)
  """
  has_thought = bool(thought)
  if form == "rt":
    m = {"role": "assistant", "content": parts(content)}
    if has_thought:
      m["channels"] = {"thought": thought}
      m["reasoning_content"] = thought
    return m
  if form == "i":
    text = (f"<think>\n{thought}\n</think>\n\n" + content.lstrip("\n")) if has_thought else content
    return {"role": "assistant", "content": parts(text)}
  if form == "ii":
    m = {"role": "assistant", "content": parts(content)}
    if has_thought:
      m["reasoning_content"] = thought
    return m
  if form == "iii":
    return {"role": "assistant", "content": parts(content)}
  if form == "v0171":
    return {"role": "assistant", "content": content}
  raise ValueError(form)


# ------------------------------------------------------------ prefix checks

def divergence(old, new):
  """Index of the first differing character; len(old) when old is a prefix of new."""
  n = min(len(old), len(new))
  i = 0
  while i < n and old[i] == new[i]:
    i += 1
  return i


def around(old, new, i, w=40):
  """The 40 characters before the split point and the next 40 of old / new (repr)."""
  return {"before": old[max(0, i - w):i], "old_next": old[i:i + w], "new_next": new[i:i + w]}


def prefix_result(old, new):
  i = divergence(old, new)
  ok = i == len(old)
  r = {"pass": ok, "old_len": len(old), "new_len": len(new)}
  if not ok:
    r["at"] = i
    r.update(around(old, new, i))
  return r


def history(turns, k, form):
  """[u1, a1, ..., u_k, a_k] from turns = [{"user", "content", "thought"}, ...]."""
  msgs = []
  for t in turns[:k]:
    msgs.append(user_msg(t["user"], form))
    msgs.append(assistant_msg(t["content"], t.get("thought"), form))
  return msgs


def runtime_refills(form, turns, k):
  """Whether 0.18.0 rewinds and refills at turn k+1: the reply of turn k carried
  `channels` (conversation.cc l.711-717, l.620-626)."""
  return form == "rt" and bool(turns[k - 1].get("thought"))


def step_checks(renderer, turns, k, form, extra, tools=None):
  """Turn k+1 of a conversation whose first k turns are done.

  step   GetSingleTurnTextFromFullHistory: old = render(history_k, gen=false),
         new = render(history_k + user_{k+1}, gen=true) (conversation.cc l.300-321)
  refill RewindAndGetInputDataVector: old = render(history[0:ckpt], gen=false) with
         ckpt = index of user_k (empty string when ckpt == 0: include_preface),
         new = render(history_k, gen=false) (conversation.cc l.1067-1100, l.961-1045)
  """
  hk = history(turns, k, form)
  nxt = user_msg(turns[k]["user"], form)
  old = renderer.render(hk, False, tools, extra)
  new = renderer.render(hk + [nxt], True, tools, extra)
  step = prefix_result(old, new)
  ckpt = 2 * (k - 1)
  r_old = renderer.render(hk[:ckpt], False, tools, extra) if ckpt > 0 else ""
  r_new = old  # render(history_k, gen=false) is the same render as the step's old
  refill = prefix_result(r_old, r_new)
  refill["runtime_runs_it"] = runtime_refills(form, turns, k)
  refill["refill_text_len"] = len(r_new) - len(r_old) if refill["pass"] else None
  return {"step": step, "refill": refill, "old": old, "new": new,
          "refill_old": r_old, "refill_new": r_new}


THINKING = (("unset", None), ("true", True), ("false", False))


def extra_for(thinking_value):
  return {} if thinking_value is None else {"enable_thinking": thinking_value}


def three_turn_table(renderer, turns, forms, steps=(1, 2), tools=None):
  rows = []
  for form in forms:
    for tname, tval in THINKING:
      for k in steps:
        try:
          c = step_checks(renderer, turns, k, form, extra_for(tval), tools)
          rows.append({"form": form, "enable_thinking": tname, "step": k,
                       "step_check": c["step"], "refill_check": c["refill"]})
        except Exception as e:  # noqa: BLE001 (template text is data; any error is a row)
          rows.append({"form": form, "enable_thinking": tname, "step": k,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"})
  return rows


def cell(row):
  if "error" in row:
    return "ERROR"
  s = "PASS" if row["step_check"]["pass"] else "FAIL"
  r = "PASS" if row["refill_check"]["pass"] else "FAIL"
  return f"{s} / {r}"


def first_fail(rows):
  for row in rows:
    if "error" in row:
      return row
    if not row["step_check"]["pass"] or not row["refill_check"]["pass"]:
      return row
  return None


def fmt_split(chk):
  return f"{chk['before']!r} ‖ old {chk['old_next']!r} / new {chk['new_next']!r}"

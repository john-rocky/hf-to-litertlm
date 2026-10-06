#!/usr/bin/env python3
"""Offline checks of the capabilities.thinking.control derivation in
make_manifest.py: one LlmMetadata per case, built in memory, no network.

    python manifest/test_thinking_control.py
"""
import pathlib
import sys
import unittest
import unittest.mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_manifest as mm  # noqa: E402
from litert_lm_builder.runtime.proto import llm_metadata_pb2  # noqa: E402

TURNS = (
    "{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
    "{%- if m.content is string -%}{{ m.content }}"
    "{%- else -%}{%- for p in m.content -%}{%- if p.type == 'text' -%}{{ p.text }}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}<|im_end|>\n{%- endfor -%}")

def gen(tail=""):
  """The generation prompt, ending with `tail` exactly (whitespace kept)."""
  return ("{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n"
          + tail.replace("\n", "\\n") + "' }}{%- endif -%}")

def bundle(jinja=None, channel=("<think>\n", "\n</think>"), model_prefix=None,
           model_type="generic_model", user=("", ""), more_channels=()):
  m = llm_metadata_pb2.LlmMetadata()
  if jinja is not None:
    m.jinja_prompt_template = jinja
  if model_prefix is not None:
    m.prompt_templates.model.prefix = model_prefix
    m.prompt_templates.user.prefix, m.prompt_templates.user.suffix = user
  for markers in ([channel] if channel else []) + list(more_channels):
    c = m.channels.add()
    c.channel_name, c.start, c.end = "thought", markers[0], markers[1]
  if model_type:
    getattr(m.llm_model_type, model_type).SetInParent()
  return m

def control(m):
  return mm.derive_thinking_control(m)[0]

class ThinkingControl(unittest.TestCase):

  def test_switch_needs_a_prompt_that_changes(self):
    on_off = (TURNS + "{%- if enable_thinking | default(true) -%}" + gen("<think>\n")
              + "{%- else -%}" + gen("<think>\n\n</think>\n\n") + "{%- endif -%}")
    self.assertEqual(control(bundle(on_off)), "switch")
    # the default differs, the answer does not
    self.assertEqual(control(bundle(on_off.replace("default(true)", "default(false)"))), "switch")
    # a switch works with or without a declared channel
    self.assertEqual(control(bundle(on_off, channel=None)), "switch")

  def test_a_switch_has_to_hold_in_both_states(self):
    def switch(on, off, **kw):
      return control(bundle(TURNS + "{%- if enable_thinking | default(true) -%}" + gen(on)
                            + "{%- else -%}" + gen(off) + "{%- endif -%}", **kw))
    bare = ("<think>", "</think>")
    # either state may leave the block to the model
    self.assertEqual(switch("", "<think>\n\n</think>\n\n"), "switch")
    self.assertEqual(switch("<think>\n", ""), "switch")
    self.assertEqual(switch("Reasoning: on\n", "Reasoning: off\n"), "switch")
    # on that closes the block, or off that opens it, is no switch
    self.assertIsNone(switch("<think>\n\n</think>\n\n", "<think>\n"))
    self.assertIsNone(switch("<think>\n\n</think>\n\n", ""))
    self.assertIsNone(switch("<think>\n", " <think>\n"))
    # the runtime has to read each state the same way (exact markers)
    self.assertIsNone(switch("<think>", "<think>\n\n</think>\n\n"))
    self.assertIsNone(switch("<think>", "<think></think>", channel=("<think>", "\n</think>")))
    self.assertEqual(switch("<think>", "<think></think>", channel=bare), "switch")

  def test_reading_the_flag_without_changing_the_prompt_is_not_derived(self):
    dead = TURNS + "{%- set unused = enable_thinking -%}" + gen()
    self.assertIsNone(control(bundle(dead)))

  def test_the_flag_inside_a_string_or_comment_is_not_a_read(self):
    quoted = TURNS + "{#- enable_thinking is not supported -#}{{ 'enable_thinking' if false }}" + gen()
    self.assertEqual(control(bundle(quoted)), mm.UNSWITCHED)

  def test_always_never_and_unswitched_follow_the_end_of_the_prompt(self):
    self.assertEqual(control(bundle(TURNS + gen("<think>\n"))), "always")
    self.assertEqual(control(bundle(TURNS + gen("<think>\n\n</think>\n\n"))), "never")
    self.assertEqual(control(bundle(TURNS + gen())), mm.UNSWITCHED)
    # the end of the prompt is compared without whitespace
    bare = ("<think>", "</think>")
    self.assertEqual(control(bundle(TURNS + gen("<think>\n"), channel=bare)), "always")
    self.assertEqual(control(bundle(TURNS + gen("<think></think>\n"), channel=bare)), "never")

  def test_the_runtime_has_to_read_the_prompt_the_same_way(self):
    # the runtime looks for the exact markers: "<think>" does not open "<think>\n"
    self.assertIsNone(control(bundle(TURNS + gen("<think>"))))
    # a start marker earlier in the prompt leaves the channel open for the runtime
    mention = "Reason inside <think> tags.\n" + TURNS + gen()
    self.assertIsNone(control(bundle(mention, channel=("<think>", "</think>"))))
    self.assertEqual(control(bundle(mention)), mm.UNSWITCHED)
    # an end marker the runtime cannot find leaves a closed block open
    self.assertIsNone(control(bundle(TURNS + gen("<think></think>"), channel=("<think>", "\n</think>"))))

  def test_markers_that_cannot_be_compared_are_not_derived(self):
    for channel in (("", "</think>"), ("<think>", ""), (" \n", "</think>")):
      self.assertIsNone(control(bundle(TURNS + gen("<think>"), channel=channel)), channel)
    # which of several channels a prompt leaves open is not derived
    tool = ("<tool_call>", "</tool_call>")
    self.assertIsNone(control(bundle(TURNS + gen("<think>\n"), more_channels=[tool])))
    self.assertIsNone(control(bundle(TURNS + gen("<think>\n"), channel=tool,
                                     more_channels=[("<think>\n", "\n</think>")])))

  def test_no_declared_channel_is_never_whatever_the_prompt_prints(self):
    self.assertEqual(control(bundle(TURNS + gen(), channel=None)), "never")
    self.assertEqual(control(bundle(TURNS + gen("<think>\n"), channel=None)), "never")

  def test_the_channel_markers_are_the_bundle_s_own(self):
    mistral = "{%- for m in messages -%}[INST]{{ m.content }}[/INST]{%- endfor -%}"
    self.assertEqual(control(bundle(mistral, channel=("[THINK]", "[/THINK]"))), mm.UNSWITCHED)
    self.assertEqual(control(bundle(mistral + "[THINK]", channel=("[THINK]", "[/THINK]"))), "always")

  def test_a_system_message_flag_is_not_a_switch(self):
    soft = ("{%- set ns = namespace(off=false) -%}"
            "{%- if messages[0].role == 'system' and '/no_think' in messages[0].content -%}{%- set ns.off = true -%}{%- endif -%}"
            + TURNS + "{%- if ns.off -%}" + gen("<think>\n\n</think>\n") + "{%- else -%}" + gen() + "{%- endif -%}")
    self.assertEqual(control(bundle(soft)), mm.UNSWITCHED)

  def test_a_bundle_without_jinja_is_read_from_its_model_prefix(self):
    self.assertEqual(control(bundle(model_prefix="<|im_start|>assistant\n<think>\n", model_type="qwen3")), "always")
    self.assertEqual(control(bundle(model_prefix="<|im_start|>assistant\n", model_type="qwen3")), mm.UNSWITCHED)
    self.assertEqual(control(bundle(model_prefix="<|im_start|>assistant\n", channel=None, model_type="qwen2p5")), "never")
    self.assertEqual(control(bundle(model_prefix="<|im_start|>assistant\n<think>\n\n</think>\n")), "never")
    self.assertIsNone(control(bundle(model_prefix="<|im_start|>assistant\n<think>")))
    known = llm_metadata_pb2.LlmMetadata().llm_model_type.DESCRIPTOR.fields_by_name
    for model_type in ("generic_model", "qwen3", "qwen2p5", "gemma4", "fast_vlm", "minicpmv4"):
      if model_type in known:  # the installed builder may predate a type
        self.assertEqual(control(bundle(model_prefix="<|im_start|>assistant\n<think>\n", model_type=model_type)),
                         "always", model_type)

  def test_the_affixes_are_rendered_the_way_the_runtime_renders_them(self):
    # the whole turn is read, not the model prefix alone
    self.assertIsNone(control(bundle(model_prefix="<|im_start|>assistant\n", user=("<think>\n", ""))))
    self.assertEqual(control(bundle(model_prefix="A:", user=("<think>", "</think>"), channel=("<think>", "</think>"))),
                     mm.UNSWITCHED)
    # an affix is template source: the newline after a block tag is trimmed,
    # so this user suffix no longer carries the channel's end marker ...
    self.assertIsNone(control(bundle(model_prefix="A:", user=("<think>", "\n</think>\n"),
                                     channel=("<think>", "\n</think>"))))
    # ... and an expression in it is evaluated
    self.assertEqual(control(bundle(model_prefix="{{ '<think>' ~ '\n' }}")), "always")

  def test_no_template_in_the_bundle_is_not_derived(self):
    self.assertIsNone(control(bundle()))
    # for these the runtime renders a template of its own, or none at all
    for model_type in ("gemma3", "gemma3n", "function_gemma", "lfm2", None):
      self.assertIsNone(control(bundle(model_prefix="<start_of_turn>model\n", model_type=model_type)), model_type)

  def test_a_template_that_does_not_parse_or_render_is_not_derived(self):
    self.assertIsNone(control(bundle("{%- if -%}")))
    self.assertIsNone(control(bundle("{{ raise_exception('no') }}")))
    # errors that are not jinja2's own
    self.assertIsNone(control(bundle("{% for a in messages %}" * 25 + "{% endfor %}" * 25)))

  def test_the_template_cannot_reach_outside_the_sandbox(self):
    self.assertIsNone(control(bundle("{{ ''.__class__.__mro__[1].__subclasses__() }}")))
    self.assertIsNone(control(bundle("{%- set _ = messages.append(1) -%}" + TURNS + gen())))

  def test_one_content_shape_is_enough_two_have_to_agree(self):
    # a string-only template raises on a list of parts
    string_only = "{%- for m in messages -%}{{ '<|im_start|>user\n' + m.content }}{%- endfor -%}" + gen("<think>\n")
    self.assertEqual(control(bundle(string_only)), "always")
    by_shape = (TURNS + "{%- if messages[0].content is string -%}" + gen("<think>\n")
                + "{%- else -%}" + gen() + "{%- endif -%}")
    self.assertIsNone(control(bundle(by_shape)))

  def test_the_runtime_s_template_helpers_are_there(self):
    helpers = ("{{ strftime_now('%Y') }}{{ messages | tojson }}{{ ' a ' | lstrip | rstrip }}"
               "{{ bos_token + eos_token }}{% generation %}{% endgeneration %}"
               + TURNS + gen("<think>\n"))
    self.assertEqual(control(bundle(helpers)), "always")

  def test_the_environment_is_set_the_way_the_runtime_sets_it(self):
    render = lambda source, **kw: mm.template_env().from_string(source).render(**kw)
    # lstrip_blocks, trim_blocks, keep_trailing_newline
    self.assertEqual(render("  {% if true %}\nx{% endif %}y\n"), "xy\n")
    # `none` is true for an undefined value only
    self.assertEqual(render("{{ missing is none }} {{ none is none }} {{ given is none }}", given=None),
                     "True False False")
    self.assertEqual(render("{{ value | tojson }}", value={"a": "é", "b": [1, "<"]}), '{"a": "é", "b": [1, "<"]}')

  def test_one_value_per_model(self):
    same = {"a.litertlm": ("always", "x"), "b.litertlm": ("always", "y")}
    self.assertEqual(mm.model_thinking_control(same), "always")
    self.assertIsNone(mm.model_thinking_control({"a.litertlm": (None, "x")}))
    for other in ("never", mm.UNSWITCHED, None):
      with self.assertRaises(SystemExit):
        mm.model_thinking_control({**same, "c.litertlm": (other, "z")})

  def test_the_unswitched_value_is_one_constant(self):
    unswitched = {"a.litertlm": (mm.UNSWITCHED, "x")}
    self.assertEqual(mm.model_thinking_control(unswitched), "model")
    with unittest.mock.patch.object(mm, "CONTROL_UNSWITCHED", None):
      self.assertIsNone(mm.model_thinking_control(unswitched))

if __name__ == "__main__":
  unittest.main()

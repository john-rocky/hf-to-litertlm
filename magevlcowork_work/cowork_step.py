#!/usr/bin/env python3
"""One step of a GUI agent with Mage-VL-Cowork's LiteRT-LM bundle: a screenshot and an instruction in, the model's next
action out (fast_vlm, one 1920 x 1088 picture per request).

  python cowork_step.py Mage-VL-Cowork.litertlm screenshot.png "Open the Downloads settings"
      [--previous "Step 1: Click the Settings icon in the dock."]... [--backend gpu|cpu] [--vision-backend gpu|cpu]
      [--max-output-tokens N] [--cache-dir DIR] [--keep-picture OUT.png] [--json]

Needs litert-lm (pip install litert-lm) and Pillow. What it sends is what the vendor's GUI agent sends at a step
(microsoft/Mage-VL-Cowork, osworld_eval/mm_agents/mage_gui_agent.py): the agent's system prompt (below, verbatim), then
one user turn = [screenshot, "Please generate the next move ... Instruction: ... Previous actions: ..."]. Earlier steps
go in as one line each under "Previous actions:" (the agent folds its history the same way); the bundle holds one
screenshot per request.

The picture: the script applies the EXIF orientation and stretches the screenshot to 1920 x 1088 (bicubic, no
padding) itself, the size the bundle reads (68 x 120 patches of 16 = 2,040 image tokens after the 2 x 2 merge), so
the runtime gets the pixels the model card's tests used. The model answers in normalized coordinates
[0, 999]; the script converts the first tool call's point back to the pixels of your screenshot the way the vendor's
agent does (x / 999 * (width - 1), rounded) and prints it after the reply. It reads the tool call leniently: the model
can write a typographic quote (U+201C / U+201D) where JSON needs a straight one.

Executing the action is up to your app. The model card asks for an isolated sandbox and a human confirmation before
irreversible actions (microsoft/Mage-VL-Cowork, "Responsible AI & Safe Deployment").
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

from PIL import Image, ImageOps

W, H = 1920, 1088   # the bundle's picture: FastVlm image_tensor_width x image_tensor_height

# Verbatim from microsoft/Mage-VL-Cowork (Apache-2.0), osworld_eval/mm_agents/mage_gui_agent.py, MAGE_GUI_SYSTEM_PROMPT.
SYSTEM = (
    'You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n'
    '# Tools\n\nYou have access to the following functions:\n\n<tools>\n'
    '{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer screen.", '
    '"parameters": {"properties": {"action": {"description": "* `click`: Click at (x,y). Use `button` for left/right/middle.\\n'
    '* `double_click`: Double click at (x,y).\\n* `triple_click`: Triple click at (x,y) to select line/paragraph.\\n'
    '* `move`: Move cursor to (x,y) without clicking.\\n* `drag`: Drag from `start` to `end` coordinates.\\n'
    '* `type`: Type text string.\\n* `keypress`: Press key(s). Single key or combo.\\n'
    '* `scroll`: Scroll at optional (x,y) position. scroll_x/scroll_y in [-999,999], |999|=full screen. Negative scroll_y=down, negative scroll_x=left.\\n'
    '* `wait`: Wait specified seconds.\\n* `terminate`: End the task.", '
    '"enum": ["click", "double_click", "triple_click", "move", "drag", "type", "keypress", "scroll", "wait", "terminate"], "type": "string"}, '
    '"coordinate": {"description": "(x,y) in [0,999].", "type": "array"}, '
    '"button": {"description": "Mouse button for click: left/right/middle.", "type": "string", "enum": ["left", "right", "middle"]}, '
    '"start": {"description": "(x,y) drag start in [0,999].", "type": "array"}, '
    '"end": {"description": "(x,y) drag end in [0,999].", "type": "array"}, '
    '"keys": {"description": "Key list for keypress.", "type": "array"}, '
    '"text": {"description": "Text for action=type.", "type": "string"}, '
    '"scroll_x": {"description": "Horizontal scroll [-999,999]. Negative=left.", "type": "number"}, '
    '"scroll_y": {"description": "Vertical scroll [-999,999]. Negative=down.", "type": "number"}, '
    '"time": {"description": "Seconds for wait.", "type": "number"}, '
    '"status": {"description": "success/failure for terminate.", "type": "string", "enum": ["success", "failure"]}, '
    '"msg_to_user": {"description": "Optional message for terminate.", "type": "string"}}, "required": ["action"], "type": "object"}}}\n'
    '</tools>\n\n'
    'For every step, reply in EXACTLY this format with NO suffix:\n\n'
    '<think>\nyour reasoning paragraph describing the current screen state, how it connects to the task goal, and the rationale for the chosen action\n</think>\n\n'
    'Action: one short imperative sentence describing the next move.\n\n'
    '<tool_call>\n{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1"}}\n</tool_call>\n\n'
    '<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n'
    '- Reasoning goes inside <think>...</think> BEFORE the Action line; nothing after the last tool_call\n'
    '- You may output at most 5 <tool_call> blocks per response\n'
    '- Use `type` for text input, `keypress` for shortcuts and special keys\n</IMPORTANT>\n\n'
    '# Response format\n\nFor every step:\n1) <think>...</think> block with your reasoning.\n2) Action: a short imperative sentence.\n'
    '3) One or more <tool_call>...</tool_call> blocks (max 5).\n\n'
    "Rules:\n- The screen's resolution is 1000x1000. Coordinates are in [0, 999] range.\n"
    '- Scroll amounts are normalized: |999| = one full screen width/height.\n'
    '- For scroll, you may include coordinate to scroll at that position.\n'
    '- For drag, specify start and end coordinates.\n- Be brief: one sentence for Action.\n'
    '- Do not output anything after the last tool call.\n- If finishing, use action=terminate.'
)
SYSTEM_SHA256 = "e771df81f39e1782aa0034902f0f4f91e6a22d00b9d4d9d9c9c57c5e630a617c"   # 3,388 bytes, UTF-8


def user_text(instruction, previous):
  """The vendor agent's user text at a step (MAGEGUIAgent._build_messages)."""
  prev = "\n".join(previous) if previous else "None"
  return ("\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
          f"Instruction: {instruction}\n\nPrevious actions:\n{prev}")


def first_tool_call(text):
  """The first <tool_call> block as a dict, read leniently (typographic quotes straightened), or None."""
  m = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S)
  if not m:
    return None
  body = m.group(1).replace("“", '"').replace("”", '"')
  try:
    obj = json.loads(body)
    return obj if isinstance(obj, dict) else None
  except ValueError:
    c = re.search(r'"coordinate"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', body)
    a = re.search(r'"action"\s*:\s*"(\w+)"', body)
    if not (c or a):
      return None
    args = {"action": a.group(1) if a else None}
    if c:
      args["coordinate"] = [float(c.group(1)), float(c.group(2))]
    return {"name": "computer_use", "arguments": args, "lenient": True}


def to_px(coord, width, height):
  """[0, 999] -> pixels of the original screenshot, as mage_gui_agent.MAGEGUIAgent._norm_to_px."""
  x = max(0.0, min(999.0, float(coord[0])))
  y = max(0.0, min(999.0, float(coord[1])))
  return int(round(x / 999.0 * (width - 1))), int(round(y / 999.0 * (height - 1)))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("bundle")
  ap.add_argument("screenshot")
  ap.add_argument("instruction")
  ap.add_argument("--previous", action="append", default=[],
                  help='an earlier step as one line, e.g. "Step 1: Click the Settings icon." (repeat in order)')
  ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
  ap.add_argument("--vision-backend", choices=("cpu", "gpu"), default=None, help="default: the --backend value")
  ap.add_argument("--max-output-tokens", type=int, default=540,
                  help="the KV cache holds 3,584 tokens; the screenshot and the prompt take about 2,990 of them "
                       "(+ about 13 per --previous line)")
  ap.add_argument("--cache-dir", default=None, help="where the runtime writes its caches (default: beside the bundle)")
  ap.add_argument("--keep-picture", default=None, help="also save the 1920 x 1088 picture the bundle reads")
  ap.add_argument("--json", action="store_true", help="print one JSON object instead of the reply and the point")
  a = ap.parse_args()

  if hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest() != SYSTEM_SHA256:
    sys.exit("the embedded system prompt is not the vendor's text (edited by hand?)")
  if not os.path.isfile(a.bundle):
    sys.exit(f"no such bundle: {a.bundle!r}")
  try:
    shot = ImageOps.exif_transpose(Image.open(a.screenshot)).convert("RGB")
  except (OSError, ValueError) as e:
    sys.exit(f"cannot read the screenshot {a.screenshot!r}: {e}")
  picture = shot.resize((W, H), Image.BICUBIC)

  import litert_lm
  backend = {"cpu": litert_lm.interfaces.Backend.CPU, "gpu": litert_lm.interfaces.Backend.GPU}
  text = ""
  with tempfile.TemporaryDirectory() as tmp:
    path = a.keep_picture or os.path.join(tmp, "picture.png")
    picture.save(path, format="PNG")
    engine = litert_lm.Engine(a.bundle, backend=backend[a.backend](),
                              vision_backend=backend[a.vision_backend or a.backend](),
                              max_num_images=1, cache_dir=a.cache_dir)
    conv = engine.create_conversation(messages=[{"role": "system", "content": SYSTEM}])
    message = {"role": "user", "content": [{"type": "image", "path": path},
                                           {"type": "text", "text": user_text(a.instruction, a.previous)}]}
    print(f"[{shot.width}x{shot.height} -> {W}x{H}, {a.backend} (vision {a.vision_backend or a.backend})]",
          file=sys.stderr)
    # streamed: a blocking send_message gives up after 10 minutes, and a phone CPU can take longer
    for chunk in conv.send_message_async(message, max_output_tokens=a.max_output_tokens):
      content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", None)
      parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
      for c in parts:
        if isinstance(c, dict) and c.get("type", "text") == "text":
          text += c.get("text", "")
          if not a.json:
            print(c.get("text", ""), end="", flush=True)
  call = first_tool_call(text)
  args = (call or {}).get("arguments") or {}
  point = to_px(args["coordinate"], shot.width, shot.height) if isinstance(args.get("coordinate"), list) \
      and len(args["coordinate"]) == 2 else None
  if a.json:
    print(json.dumps({"reply": text, "tool_call": call, "point_px": point,
                      "screenshot_wh": [shot.width, shot.height]}, ensure_ascii=False))
  else:
    print()
    print(f"[action {args.get('action')!r}, point {point} px on {shot.width}x{shot.height}]"
          if call else "[no tool call in the reply]", file=sys.stderr)


if __name__ == "__main__":
  main()

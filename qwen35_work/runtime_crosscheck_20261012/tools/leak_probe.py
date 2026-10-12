#!/usr/bin/env python3
"""Cross-conversation state probe (litert-lm 0.18.0, CPU, greedy top_k=1). One process = one run.

  sysswitch --mode shared : one Engine; conversation 1 = system A + Q1..Q3 (three turns), then
                            conversation 2 = system B + the same Q1..Q3. Reports conversation 2.
  sysswitch --mode fresh  : one Engine; conversation 2 only (the control).
  qset --mode shared      : one Engine; round 1 = the 20 questions, one new Conversation each
                            (fixed system prompt); round 2 = the same 20 again on the same Engine.
  qset --mode fresh --question i : one Engine, one Conversation, question i only (the control;
                            the caller runs one process per question).
Scoring: regex per question; `degenerate` = the verify_quality.py rule; `echo` = the question text in the answer.
"""
import argparse, json, os, re, sys, time, traceback
from collections import Counter

import litert_lm
from litert_lm import interfaces

SYS_QSET = "You are a helpful assistant. Answer briefly."
SYS_A = "Your name is Alpha. Begin every reply with the word Alpha."
SYS_B = "Your name is Beta. Begin every reply with the word Beta."
SWITCH_QS = ["What is the capital of France?", "What is 17 times 23?", "What is your name?"]

# 1-8 = scripts/verify_quality.py QUESTIONS (gate8q), 9-10 = the 2026-10-09 lane's sanity prompts, 11-20 fixed here.
QUESTIONS = [
    ("17+25=42", "What is 17 + 25?", r"\b42\b"),
    ("capital=Tokyo", "What is the capital of Japan?", r"tokyo"),
    ("opp(hot)=cold", 'What is the opposite of "hot"?', r"\bcold\b"),
    ("days/week=7", "How many days are in a week?", r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?', r"merci"),
    ("8*7=56", "What is 8 times 7?", r"\b56\b"),
    ("0.9>0.11", "Which is larger: 0.9 or 0.11?", r"0\.9"),
    ("rhyme=blue", 'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
    ("17*23=391", "What is 17 times 23?", r"\b391\b"),
    ("capital=Paris", "What is the capital of France? Answer in one word.", r"paris"),
    ("spider=8", "How many legs does a spider have?", r"\beight\b|\b8\b"),
    ("sky=blue", "What color is the sky on a clear day?", r"\bblue\b"),
    ("boil=100", "What is the boiling point of water in degrees Celsius?", r"\b100\b"),
    ("red planet=Mars", "Which planet is known as the Red Planet?", r"\bmars\b"),
    ("100/4=25", "What is 100 divided by 4?", r"\b25\b"),
    ("ocean=Pacific", "What is the largest ocean on Earth?", r"pacific"),
    ("min/hour=60", "How many minutes are in an hour?", r"\bsixty\b|\b60\b"),
    ("water=H2O", "What is the chemical formula for water?", r"h2o|h₂o"),
    ("author=Shakespeare", "Who wrote the play Romeo and Juliet?", r"shakespeare"),
    ("sqrt81=9", "What is the square root of 81?", r"\bnine\b|\b9\b"),
]


def degenerate(text):
  words = text.split()
  if len(words) >= 10:
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    if grams and Counter(grams).most_common(1)[0][1] >= 3:
      return True
    if len(set(words)) / len(words) < 0.30:
      return True
  if len(text) >= 40 and len(set(text)) < 15:
    return True
  if text.count("<|") >= 5 or text.count("<pad>") >= 5:
    return True
  return False


def msg_text(msg):
  out = []
  for p in msg.contents:
    j = p.to_json() if hasattr(p, "to_json") else None
    if isinstance(j, dict):
      out.append(j.get("text", ""))
  return "".join(out)


def channels(msg):
  try:
    return {k: v[:300] for k, v in msg.channels.items()}
  except Exception:  # noqa: BLE001
    return {}


def ask(engine, system, prompt, max_out):
  conv = engine.create_conversation(system_message=system, sampler_config=interfaces.SamplerConfig(top_k=1))
  try:
    t = time.time()
    msg = conv.send_message(prompt, max_output_tokens=max_out)
    return {"text": msg_text(msg), "channels": channels(msg), "s": round(time.time() - t, 2)}, conv
  except Exception as e:  # noqa: BLE001
    return {"text": "", "error": f"{type(e).__name__}: {e}"[:600], "s": round(time.time() - t, 2)}, conv


def score(idx, text):
  key, q, rx = QUESTIONS[idx]
  low = text.lower()
  return {"key": key, "correct": bool(re.search(rx, low)), "degenerate": degenerate(text),
          "echo": q.lower().rstrip("?").rstrip(".")[:30] in low}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("kind", choices=["sysswitch", "qset"])
  ap.add_argument("--model", required=True)
  ap.add_argument("--label", required=True)
  ap.add_argument("--mode", choices=["shared", "fresh"], required=True)
  ap.add_argument("--question", type=int, default=None, help="qset fresh: 0-based index")
  ap.add_argument("--max-output-tokens", type=int, default=64)
  ap.add_argument("--max-num-tokens", type=int, default=1024)
  ap.add_argument("--run", type=int, default=1)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  rec = {"kind": a.kind, "label": a.label, "mode": a.mode, "model": a.model, "model_bytes": os.path.getsize(a.model),
         "runtime": "litert-lm 0.18.0 (python, cpu, fresh process)", "sampler": "top_k=1", "run": a.run,
         "pid": os.getpid(), "max_output_tokens": a.max_output_tokens}
  t0 = time.time()
  try:
    engine = litert_lm.Engine(a.model, backend=interfaces.Backend.CPU(), max_num_tokens=a.max_num_tokens)
    rec["load_s"] = round(time.time() - t0, 1)
    if a.kind == "sysswitch":
      convs = []
      if a.mode == "shared":
        conv1 = engine.create_conversation(system_message=SYS_A, sampler_config=interfaces.SamplerConfig(top_k=1))
        rec["conv1"] = []
        for q in SWITCH_QS:
          msg = conv1.send_message(q, max_output_tokens=a.max_output_tokens)
          rec["conv1"].append({"q": q, "text": msg_text(msg), "channels": channels(msg)})
        convs.append(conv1)
      conv2 = engine.create_conversation(system_message=SYS_B, sampler_config=interfaces.SamplerConfig(top_k=1))
      rec["conv2"] = []
      for q in SWITCH_QS:
        msg = conv2.send_message(q, max_output_tokens=a.max_output_tokens)
        t = msg_text(msg)
        rec["conv2"].append({"q": q, "text": t, "channels": channels(msg),
                             "starts_beta": t.strip().lower().startswith("beta"),
                             "starts_alpha": t.strip().lower().startswith("alpha"),
                             "mentions_alpha": "alpha" in t.lower(), "degenerate": degenerate(t)})
      convs.append(conv2)
      for c in convs:
        try:
          c.close()
        except Exception:  # noqa: BLE001
          pass
    else:
      rec["rounds"] = []
      if a.mode == "fresh":
        plan = [[a.question]]
      else:
        plan = [list(range(len(QUESTIONS))), list(range(len(QUESTIONS)))]
      for r, idxs in enumerate(plan, 1):
        rows = []
        for i in idxs:
          res, conv = ask(engine, SYS_QSET, QUESTIONS[i][1], a.max_output_tokens)
          row = {"i": i, "q": QUESTIONS[i][1], **res, **score(i, res["text"])}
          rows.append(row)
          print(f"R{r} {a.mode:6s} {i:2d} {'RIGHT' if row['correct'] else 'wrong'}"
                f"{' DEGEN' if row['degenerate'] else ''}{' ECHO' if row['echo'] else ''} | {res['text'][:80]!r}", flush=True)
          try:
            conv.close()
          except Exception:  # noqa: BLE001
            pass
        rec["rounds"].append({"round": r, "rows": rows, "right": sum(x["correct"] for x in rows), "n": len(rows),
                              "degenerate": sum(x["degenerate"] for x in rows), "echo": sum(x["echo"] for x in rows)})
    engine.close()
  except Exception as e:  # noqa: BLE001
    rec["error"] = f"{type(e).__name__}: {e}"[:1500]
    rec["traceback"] = traceback.format_exc()[-2000:]
  rec["elapsed_s"] = round(time.time() - t0, 1)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.out, "w", encoding="utf-8") as f:
    json.dump(rec, f, ensure_ascii=False, indent=1)
  summary = {k: rec.get(k) for k in ("kind", "label", "mode", "run", "error")}
  if a.kind == "qset":
    summary["rounds"] = [(x["round"], f"{x['right']}/{x['n']}", f"degen {x['degenerate']}", f"echo {x['echo']}") for x in rec.get("rounds", [])]
  else:
    summary["conv2"] = [(x["starts_beta"], x["starts_alpha"], x["text"][:60]) for x in rec.get("conv2", [])]
  print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())

"""One Engine and one Conversation, three synchronous send_message calls."""
import argparse
import importlib.metadata
import re
import traceback

from runtime_helpers import ROOT, save_json, add_runtime_args

p = argparse.ArgumentParser()
add_runtime_args(p, "multiturn.json")
p.add_argument("--mode", choices=["off", "on"], required=True)
a = p.parse_args()
import litert_lm
out = a.out
model = a.model
data = {"runtime": a.runtime, "mode": a.mode, "model": model, "backend": "cpu", "cache_dir": ":nocache",
        "token_count_definition": "Native Conversation.token_count: number of tokens in the KV Cache (prefill + decode); occupancy can decrease between turns.",
        "thinking_config": {"enable_thinking": False} if a.mode == "off" else "omitted (runtime default)",
        "sampler": {"temperature": 0, "seed": 0}, "turns": [], "verdict": "RUNNING"}
save_json(out, data)
questions = ["My name is Ken and I live in Osaka. Reply with exactly: OK.",
             "Which city do I live in? Answer with the city name only.",
             "What is my name? Answer with the name only."]
try:
    with litert_lm.Engine(model_path=model, backend=litert_lm.Backend.CPU(),
                          cache_dir=":nocache", max_num_tokens=4096, enable_benchmark=False) as engine:
        options = {"sampler_config": litert_lm.SamplerConfig(temperature=0, seed=0)}
        if a.mode == "off":
            options["thinking_config"] = litert_lm.ThinkingConfig(enable_thinking=False)
        with engine.create_conversation(**options) as conv:
            for index, question in enumerate(questions):
                row = {"turn": index + 1, "question": question,
                       "conversation_token_count_before": conv.token_count}
                data["turns"].append(row)
                save_json(out, data)
                response = conv.send_message(question)
                after = conv.token_count
                content = response.get("content", [])
                text = content if isinstance(content, str) else "".join(
                    x.get("text", "") for x in content if x.get("type") == "text")
                row.update(text=text, channels=response.get("channels", {}), response=response,
                           conversation_token_count_after=after,
                           conversation_token_count_delta=after - row["conversation_token_count_before"])
                save_json(out, data)
                print(f"turn {index + 1}: {response!r}; conversation token count {after}", flush=True)
    data["checks"] = {"three_completed_turns": len(data["turns"]) == 3 and all("response" in x for x in data["turns"]),
                      "turn2_osaka": bool(re.search(r"\bosaka\b", data["turns"][1]["text"], re.I)),
                      "turn3_ken": bool(re.search(r"\bken\b", data["turns"][2]["text"], re.I)),
                      "no_exception": True}
    data["manual_no_garbage"] = "PENDING"
    data["verdict"] = "PASS" if all(data["checks"].values()) else "FAIL"
except Exception as exc:
    data.update(verdict="FAIL", exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    print(data["traceback"], flush=True)
save_json(out, data)
raise SystemExit(0 if data["verdict"] == "PASS" else 1)

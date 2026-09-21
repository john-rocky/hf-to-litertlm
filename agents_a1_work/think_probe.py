"""Capture native CLI thought channels with default and 64-token budgets."""
import argparse
from pathlib import Path
import re
from runtime_helpers import ROOT, run_cli, save_json, add_runtime_args, configure_output, cli_success

p = argparse.ArgumentParser()
add_runtime_args(p, "think_probe.json")
a = p.parse_args()
model = a.model
out = Path(a.out)
configure_output(out)
data = {"runtime": a.runtime, "model": model, "backend": "cpu", "runs": [],
        "budget_comparisons": [], "verdict": "RUNNING"}
channel_pattern = re.compile(r"\[thought\]\s?(.*?)\s*\[/thought\]", re.S)
questions = [("arithmetic", "What is 17 + 25?", r"\b42\b"),
             ("capital", "What is the capital of Japan?", r"\btokyo\b")]
for key, question, answer_pattern in questions:
    pair = []
    for budget in (None, 64):
        mode = "default" if budget is None else "budget64"
        command = [a.cli, "run", model, "--prompt", question, "--backend", "cpu",
                   "--cache", "no", "--temperature", "0", "--seed", "0", "--thinking", "true"]
        if budget is not None:
            command += ["--thinking-budget", str(budget)]
        execution = run_cli(command, f"think_{a.runtime}_{key}_{mode}")
        stdout = execution["stdout"]
        thought = "\n".join(channel_pattern.findall(stdout))
        answer = channel_pattern.sub("", stdout).strip()
        checks = {"exit_zero": cli_success(execution),
                  "thought_channel_markers": bool(channel_pattern.search(stdout)),
                  "correct_answer": bool(re.search(answer_pattern, answer, re.I)),
                  "answer_has_no_literal_think_tags": "<think>" not in answer and "</think>" not in answer}
        row = {"question": question, "mode": mode, "thinking_budget": budget,
               "stdout": stdout, "reasoning": thought, "reasoning_characters": len(thought),
               "answer": answer, "checks": checks, "execution": execution}
        pair.append(row)
        data["runs"].append(row)
        save_json(out, data)
        print(question, mode, checks, "reasoning_characters", len(thought), "answer", repr(answer), flush=True)
    comparison = {"question": question, "default_reasoning_characters": pair[0]["reasoning_characters"],
                  "budget64_reasoning_characters": pair[1]["reasoning_characters"],
                  "budget64_strictly_shorter": 0 < pair[1]["reasoning_characters"] < pair[0]["reasoning_characters"],
                  "budget64_still_answers": pair[1]["checks"]["correct_answer"]}
    data["budget_comparisons"].append(comparison)
data["verdict"] = "PASS" if all(all(r["checks"].values()) for r in data["runs"]) and all(
    x["budget64_strictly_shorter"] and x["budget64_still_answers"] for x in data["budget_comparisons"]) else "FAIL"
save_json(out, data)
raise SystemExit(0 if data["verdict"] == "PASS" else 1)

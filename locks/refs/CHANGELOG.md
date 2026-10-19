# Reference locks, one row per model and toolchain

Written by scripts/verify_refs.sh. Build 1 = convert.py, build 2 = verify_lock.py's rebuild; peak RSS from /usr/bin/time -l.
compat = the toolchain-skew shims convert.py applied (recipe.compat of the lock); each one goes away with the release that fixes it.

| date | toolchain | model @ revision | bundle sha256 | vs the previous toolchain | build 2 | check (litert-lm CPU) | 17 + 25 (information) | compat | seconds build 1 / build 2 | peak RSS build 1 / build 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | lt0.9.4-b0.18.0-t2.13.0 | Qwen/Qwen3-1.7B @ 70d244cc | `b8d6b890a81f1370` | first lock | MATCH | PASS 'pineapple' | '42' (expected 42) | none | 240 / 94 | 25.1 / 25.1 GB |
| 2026-10-19 | lt0.9.4-b0.18.0-t2.13.0 | HuggingFaceTB/SmolLM2-1.7B-Instruct @ 31b70e2e | `9c9445ff00459876` | first lock | MATCH | PASS 'pineapple' | '32' (expected 42) | none | 172 / 121 | 25.1 / 25.2 GB |
| 2026-10-19 | lt0.9.4-b0.18.0-t2.13.0 | google/gemma-4-E2B-it @ 3e22461f | `5bb43a5416fee1d8` | first lock | MATCH | PASS 'pineapple' | '42' (expected 42) | gemma4-use_template_for_fc_format (litert-torch#1329, fixed on main by #1241) | 428 / 549 (198 s of the 549 paused by another job on the host; 356 s real) | 63.9 / 54.3 GB |

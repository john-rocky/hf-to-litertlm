"""Card checks for litert-community/LFM3-VL-2B-LiteRT after LFM3-VL-2B_int4_ctx4096.litertlm was added (a checks file
of tools/card_verify.py): every number the card gives for the new file, re-derived from the raw records.

  python tools/card_verify.py --card README.md --checks lfm3vl_work/card_checks_int4_ctx4096.py \
      --sources sources.json --opt phones --out verify.json

Shares no code with the scorer or the card writer: the Mac medians are taken here from the per-run records, the
placement answers are scored here (the first "point_2d" of the answer against the request's truth mask, grey > 127 at
[int(y), int(x)] of the 1000 x 1000 mask; a missing point or one outside 0-1000 = wrong), reply lengths and cut-offs
are read from the run files, and the phone numbers are parsed from each leg's own .err (LiteRT-LM BenchmarkInfo, the
vision executor line, Peak system ram) and .out (the reply). "--opt phones" adds the Galaxy S26 and Pixel 11 Pro
sessions.
"""
import glob
import os
import re

SOURCES = {
    "bundle_ctx4096": "LFM3-VL-2B_int4_ctx4096.litertlm (the uploaded bytes)",
    "mac_ctx4096_gpu": "Mac timing record of the 4,096-token file, GPU (text runs + image runs, JSON)",
    "mac_ctx4096_cpu": "Mac timing record of the 4,096-token file, CPU",
    "mac_int4_gpu": "Mac timing record of LFM3-VL-2B_int4.litertlm, GPU (same window)",
    "mac_int4_cpu": "Mac timing record of LFM3-VL-2B_int4.litertlm, CPU (same window)",
    "fixtures": "fixtures.json of tools/demo_scene_check (where-to-put scene; masks relative to it)",
    "placement_ctx4096": "run_litertlm.py output: the 4,096-token file, GPU, thinking on, 30 requests",
    "placement_int4": "run_litertlm.py output: LFM3-VL-2B_int4.litertlm, GPU, thinking on, 30 requests",
    "placement_source": "run_hf.py output: the source model in float32 (MPS), the plan with S512 thinking on",
    "sections_diff": "section and tensor comparison of the two int4 files (JSON)",
    "s26_session": "Galaxy S26 session directory written by tools/devicebench (legs .err/.out/.meta.json/.state_before.txt/.ceil.txt)",
    "pixel_session": "Pixel 11 Pro session directory written by tools/devicebench",
}
# leg-name prefixes of the device plan: the 4,096-token file and the int4 file
NEW, OLD = "k4", "i4"


def run(v):
  file_checks(v)
  pg, _ = mac_checks(v)
  placement_checks(v)
  if "phones" in v.opts:
    s26_checks(v, pg)
    pixel_checks(v)


def file_checks(v):
  p = v.path("bundle_ctx4096")
  n, h = os.path.getsize(p), v.sha256(p)
  v.check("file row: bytes and sha256", f"| {v.commas(n)} B ({n / 1e9:.2f} GB) | `{h}` |", p)


def mac_checks(v):
  c_ = v.commas
  vals = {}
  for lab in ("mac_ctx4096_gpu", "mac_int4_gpu", "mac_ctx4096_cpu", "mac_int4_cpu"):
    r = v.json(lab)
    text = [x for x in r["text"]["runs"] if not x.get("superseded")]
    vl = r["vl"]["runs"]
    vals[lab] = {
        "pre": v.median([x["prefill_turns"][0][1] for x in text]), "dec": v.median([x["decode_turns"][0][1] for x in text]),
        "dec_all": [x["decode_turns"][0][1] for x in text],
        "ttft": v.median([x["ttft_s_runtime"] for x in text]), "img": v.median([x["first_chunk_s"] for x in vl]),
        "fp_text": max(x["peak_footprint_bytes"] for x in text) / 1e9,
        "fp_vl": max(x["peak_footprint_bytes"] for x in vl) / 1e9,
        "la": [x["loadavg_before"][0] for x in text + vl], "max_num_tokens": r["max_num_tokens"]}
  g, c = vals["mac_ctx4096_gpu"], vals["mac_ctx4096_cpu"]
  gi, ci = vals["mac_int4_gpu"], vals["mac_int4_cpu"]
  src = os.path.dirname(v.path("mac_ctx4096_gpu"))
  v.check("Mac GPU row", f"| int4, 4,096-token context | GPU (vision encoder on the GPU) | {c_(round(g['pre']))} | "
          f"{g['dec']:.1f} | {g['ttft']:.3f} | {g['img']:.3f} | {g['fp_text']:.2f} / {g['fp_vl']:.2f} |", src)
  v.check("Mac CPU row", f"| int4, 4,096-token context | CPU (vision encoder on the CPU) | {c_(round(c['pre']))} | "
          f"{c['dec']:.1f} | {c['ttft']:.2f} | {c['img']:.2f} | {c['fp_text']:.2f} / {c['fp_vl']:.2f} |", src)
  la = [x for x_ in vals.values() for x in x_["la"]]
  v.check("Mac window load average", f"load average {int(min(la))}–{round(max(la))}", src)
  mt = {lab: x["max_num_tokens"] for lab, x in vals.items()}
  if mt["mac_ctx4096_gpu"] == mt["mac_ctx4096_cpu"] and mt["mac_int4_gpu"] == mt["mac_int4_cpu"]:
    v.check("Mac window max_num_tokens", f"`max_num_tokens` {c_(mt['mac_ctx4096_gpu'])} for the 4,096-token file and "
            f"{c_(mt['mac_int4_gpu'])} for the other", src)
  else:
    v.record("Mac window max_num_tokens", False, "one value per file", str(mt))
  v.check("Mac window: the int4 file's numbers",
          f"the 2,048-token file gave {c_(round(gi['pre']))} / {gi['dec']:.1f} tok/s on the GPU and "
          f"{round(ci['pre'])} / {ci['dec']:.1f} tok/s on the CPU", src)
  pg = (g["dec"] - gi["dec"]) / gi["dec"] * 100
  pc = (c["dec"] - ci["dec"]) / ci["dec"] * 100
  v.check("Mac decode differences", f"decoded {round(-pg)} % slower on the GPU and {round(-pc)} % slower on the CPU", src)
  v.check("which file: Mac GPU %", f"and {round(-pg)} % slower on the Mac GPU", src)
  v.check("Mac disturbed runs", f"One of its three GPU decode runs gave {min(g['dec_all']):.1f} tok/s and one CPU run "
          f"{min(c['dec_all']):.1f}", src)
  return pg, pc


def placement_checks(v):
  c_ = v.commas
  fxp = v.path("fixtures")
  fxd = os.path.dirname(fxp)
  fx = {it["id"]: it for it in v.json("fixtures")["items"]}

  def right(item_id, ans):
    p = v.point_2d(ans)
    if p is None or not (0 <= p[0] <= 1000 and 0 <= p[1] <= 1000):
      return False
    return v.mask_hit(os.path.join(fxd, fx[item_id]["truth"]["mask"]), *p)
  k4 = {c["id"]: c for c in v.json("placement_ctx4096")["cases"]}
  i4 = {c["id"]: c for c in v.json("placement_int4")["cases"]}
  L = {r["id"]: r for r in v.jsonl("placement_source")
       if r["arm"] == "S512" and r["think"] == "on" and r["q_type"] == "place"}
  n = len(fx)
  assert len(k4) == len(i4) == len(L) == n, (len(k4), len(i4), len(L), n)
  assert {r["device"] for r in L.values()} == {"mps"} and {r["dtype"] for r in L.values()} == {"fp32"}
  rk = {i: right(i, c["content"]) for i, c in k4.items()}
  ri = {i: right(i, c["content"]) for i, c in i4.items()}
  # the source model's text holds its thought; the answer = the text after the last "</think>" (none = no answer)
  l_ans = {i: (r["text_raw"].rsplit("</think>", 1)[1] if "</think>" in (r.get("text_raw") or "") else None)
           for i, r in L.items()}
  rl = {i: right(i, a) if a is not None else False for i, a in l_ans.items()}
  cut = lambda c: (c["content"] or "").strip() == "" and c["decode_tokens"] >= c["max_new"]  # noqa: E731
  ck = [i for i, c in k4.items() if cut(c)]
  ci = [i for i, c in i4.items() if cut(c)]
  cl = [i for i, r in L.items() if not r["ended_on_eos"] or l_ans[i] is None]
  longest = lambda cs: max(c["decode_tokens"] for c in cs.values())  # noqa: E731
  nl = max(r["n_new"] for r in L.values())
  lm = {r["max_new"] for r in L.values()}
  assert len(lm) == 1, lm
  lm = lm.pop()
  src_l = v.path("placement_source")
  v.check("placement table: source model",
          f"| Source model in float32 (Transformers on the Mac GPU), the same 512 × 512 images | {sum(rl.values())} of {n} | "
          f"{len(cl)} (allowed {c_(lm)} new tokens) | {c_(nl)} |", src_l)
  for nm, cs, ctx in (("4,096-token", k4, 4096), ("2,048-token", i4, 2048)):
    v.record(f"placement budget sums ({nm}: max_new + prompt tokens = {ctx})",
             all(c["max_new"] + c["hf_input_tokens"] == ctx for c in cs.values()), "all", nm)
  mr = lambda cs: f"{c_(min(c['max_new'] for c in cs.values()))}–{c_(max(c['max_new'] for c in cs.values()))}"  # noqa: E731
  v.check("placement budgets", f"({mr(k4)} new tokens for the 4,096-token file, {mr(i4)} for the 2,048-token file); "
          f"the source model was allowed {c_(lm)} new tokens.", "both runs + source")
  hi = sorted({c["hf_input_tokens"] for c in list(k4.values()) + list(i4.values())})
  v.check("usage: prompt tokens", f"our placement prompts were {hi[0]}–{hi[-1]} tokens", "both runs hf_input_tokens")
  ni = {r["n_image_tokens"] for r in L.values()}
  v.check("usage: image tokens", f"an image is {ni.pop()} tokens" if len(ni) == 1 else "(several image token counts)",
          "source n_image_tokens")
  # the new file's weights: every big buffer of the decoder but one is in both files; the one that is not = an int32
  # [1, 1, ctx] position constant, and the other sections are equal
  sw = v.json("sections_diff")
  dec_ = [s for s in sw["sections"] if s["label"] == "tf_lite_prefill_decode"][0]["tflite"]
  bd = {(x["name"].split(";")[0], x["type"], tuple(x["new_shape"]), tuple(x["old_shape"]))
        for g in dec_["subgraphs"] for x in g["tensor_diffs"] if "buffer" in x["kind"]}
  eq = {s["label"]: s["equal"] for s in sw["sections"]}
  wok = (dec_["big_buffers"]["old_not_in_new"] == 1 == dec_["big_buffers"]["new_not_in_old"] and len(bd) <= 2
         and all(t == 2 and ns == (1, 1, 4096) and os_ == (1, 1, 2048) for _, t, ns, os_ in bd)
         and eq["tf_lite_embedder"] and eq["tf_lite_vision_encoder"] and eq["tf_lite_vision_adapter"]
         and eq["HF_Tokenizer_Zlib"])
  v.record("weights byte for byte (only an int32 [1, 1, ctx] position constant differs among the decoder's buffers; "
           "tokenizer / embedder / vision / adapter sections equal)",
           wok and "its weights are the int4 file's byte for byte" in v.card, "true", v.path("sections_diff"))
  v.check("placement table: int4", f"| `LFM3-VL-2B_int4.litertlm` | {sum(ri.values())} of {n} | {len(ci)} | "
          f"{c_(longest(i4))} |", v.path("placement_int4"))
  v.check("placement table: ctx 4096", f"| `LFM3-VL-2B_int4_ctx4096.litertlm` | {sum(rk.values())} of {n} | {len(ck)} | "
          f"{c_(longest(k4))} |", v.path("placement_ctx4096"))
  answered = [i for i in i4 if i not in ci]
  same = sum(1 for i in answered if k4[i]["content"] == i4[i]["content"]
             and (k4[i].get("channels") or {}).get("thought") == (i4[i].get("channels") or {}).get("thought"))
  if same == len(answered):
    v.check("placement: the same thought and point on the answered", f"On the {len(answered)} requests the 2,048-token "
            "file answered, the two int4 files wrote the same thought and gave the same point.", "both runs")
  else:
    v.record("placement: the same thought and point on the answered", False, f"{same} / {len(answered)}", "both runs")
  starts = sum(1 for i in ci if (k4[i]["channels"]["thought"]).startswith((i4[i]["channels"]["thought"]).rstrip()))
  newright = sum(1 for i in ci if rk[i])
  if starts == len(ci):
    v.check("placement: the cut ones", f"On the {len(ci)} where it ran out of room, the 4,096-token file's thought begins "
            f"with the same words and goes on to an answer, {newright} of them right.", "both runs")
  else:
    v.record("placement: the cut ones", False, f"{starts} / {len(ci)}", "both runs")
  top = max(k4.values(), key=lambda c: c["decode_tokens"])
  room = 4096 - (top["decode_tokens"] + top["hf_input_tokens"])
  v.check("placement: longest reply room", f"within {room} tokens of the end of its context "
          f"({c_(top['decode_tokens'] + top['hf_input_tokens'])} of 4,096 tokens with the prompt)", top["id"])
  both = sum(1 for i in rk if rk[i] and rl[i])
  onlyk = sum(1 for i in rk if rk[i] and not rl[i])
  onlyl = sum(1 for i in rk if rl[i] and not rk[i])
  if onlyk == onlyl:
    v.check("placement: against the source model", f"both on {both}, each alone on {onlyk}.", "runs + source")
  else:
    v.record("placement: against the source model", False, f"{both} / {onlyk} / {onlyl}", "runs + source")
  v.check("first paragraph: right and cut", f"it answered {sum(rk.values())} of our {n} requests right and ran out of "
          f"room on none, where the 2,048-token file answered {sum(ri.values())} and ran out before the answer on "
          f"{len(ci)}", "runs")
  v.check("which file: cut requests", f"on {len(ci)} of our {n} requests to place an object in a photo, the 2,048-token "
          "file's thought reached the end of the context and no answer came", "runs")
  if same == len(answered):
    v.check("which file: the answered ones equal", f"on the {len(answered)} placement requests below that fit in 2,048 "
            "tokens its thoughts and points equal the int4 file's", "both runs")
  v.check("limitations: cut thoughts and the room left", f"{len(ci)} of {n} thoughts did not fit in 2,048 tokens, and the "
          f"4,096-token file's longest reply came within {room} tokens of its end", "runs")
  v.check("usage: longest reply", f"the 4,096-token file's longest was {c_(top['decode_tokens'])} tokens", "runs")
  kinds = {}
  for it in fx.values():
    kinds[it["rel_kind"]] = kinds.get(it["rel_kind"], 0) + 1
  v.check("placement kinds", f"{kinds['lr']} to the left or right of something, {kinds['between']} between two things, "
          f"{kinds['fb']} in front of or behind something, {kinds['surface']} on a surface", fxp)
  # the runtime of both runs: one interpreter, and the litert-lm version installed beside it is the card's
  pys = {v.json(s).get("python") for s in ("placement_ctx4096", "placement_int4")}
  ver = None
  if len(pys) == 1:
    venv = os.path.dirname(os.path.dirname(next(iter(pys))))
    for meta in glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "litert_lm-*.dist-info", "METADATA")):
      m = re.search(r"^Version: (\S+)", open(meta).read(), re.M)
      ver = m.group(1) if m else None
  v.record("placement runtime (one interpreter for both runs; the litert-lm version installed with it)",
           ver is not None and f"litert-lm {ver}" in v.card, f"litert-lm {ver}", str(sorted(pys)))


def leg(v, rdir, tag):
  e = open(os.path.join(rdir, tag + ".err"), errors="replace").read()
  op = os.path.join(rdir, tag + ".out")
  o = open(op, errors="replace").read() if os.path.exists(op) else ""
  b = v.bench(e)
  b["thought"], b["answer"] = v.thought_answer(o)
  return b


def s26_checks(v, pg_mac):
  c_ = v.commas
  rdir = v.path("s26_session")
  rows = {}
  for tag in (f"{NEW}-gpu-tA1", f"{NEW}-gpu-tA2", f"{NEW}-gpu-tA3", f"{OLD}-gpu-tB1", f"{OLD}-gpu-tB2", f"{OLD}-gpu-tB3",
              f"{NEW}-gpu-vA1", f"{NEW}-gpu-vA2", f"{NEW}-gpu-vA3", f"{OLD}-gpu-vB1", f"{OLD}-gpu-vB2", f"{OLD}-gpu-vB3"):
    p = os.path.join(rdir, tag + ".err")
    if os.path.exists(p) and "BenchmarkInfo:" in open(p, errors="replace").read():
      rows[tag] = leg(v, rdir, tag)
  T = {f: [rows[t] for t in rows if t.startswith(f + "-gpu-t")] for f in (NEW, OLD)}
  V = {f: [rows[t] for t in rows if t.startswith(f + "-gpu-v")] for f in (NEW, OLD)}
  T = {f: [x for x in xs if x["prefill"] and x["prefill"][0][0] == 1024 and x["decode"] and x["decode"][0][0] == 256
           and x["nfd"] == 0] for f, xs in T.items()}
  V = {f: [x for x in xs if x["prefill"] and x["prefill"][0][0] == 283 and x["nfd"] == 0] for f, xs in V.items()}
  enough = all(len(T[f]) >= 3 and len(V[f]) >= 3 for f in (NEW, OLD))
  v.record("S26 counted legs", enough, "3 per file per block", str({f: (len(T[f]), len(V[f])) for f in T}))
  if not enough:
    return
  dk, di = v.median([x["decode"][0][1] for x in T[NEW]]), v.median([x["decode"][0][1] for x in T[OLD]])
  pk_, pi_ = v.median([x["prefill"][0][1] for x in T[NEW]]), v.median([x["prefill"][0][1] for x in T[OLD]])
  tk, ti = v.median([x["ttft_s"] for x in T[NEW]]), v.median([x["ttft_s"] for x in T[OLD]])
  fk = v.median([x["vision_s"] + x["ttft_s"] for x in V[NEW]])
  fi = v.median([x["vision_s"] + x["ttft_s"] for x in V[OLD]])
  vk = v.median([x["vision_s"] for x in V[NEW]]) * 1000
  tvk = v.median([x["ttft_s"] for x in V[NEW]])
  pct = (dk - di) / di * 100
  v.check("S26 row", f"| int4, 4,096-token context | GPU, vision encoder on the GPU | {c_(round(pk_))} ⁶ | {dk:.2f} ⁶ | "
          f"{tk:.2f} ⁶ | {fk:.3f} ⁷ (the int4 file in the same session: {fi:.3f}) |", rdir)
  v.check("S26 first paragraph", f"it decodes {dk:.2f} tokens/s after a 1,024-token prompt, against {di:.2f} tokens/s", rdir)
  v.check("S26 which file %", f"It decodes {round(-pct)} % slower than `LFM3-VL-2B_int4.litertlm` on the Galaxy S26 GPU",
          rdir)
  v.check("S26 runs", " / ".join(f"{x['decode'][0][1]:.2f}" for x in T[NEW]), rdir)
  v.check("S26 int4 medians", f"prefill {c_(round(pi_))} tok/s, decode {di:.2f} tok/s, time to first token {ti:.2f} s",
          rdir)
  v.check("S26 image", f"vision encoder {round(vk)} ms, then {tvk:.2f} s; the int4 file in that session: {fi:.3f} s", rdir)
  v.record("S26 vs Mac decode differences (recorded, not a card string)", True,
           f"S26 {pct:.1f} %, Mac GPU {pg_mac:.1f} %", "medians")
  rg = lambda xs, f: f(min(xs)) if f(min(xs)) == f(max(xs)) else f"{f(min(xs))}–{f(max(xs))}"  # noqa: E731
  mb = lambda x: c_(round(x))  # noqa: E731
  s1 = lambda x: f"{x:.1f}"  # noqa: E731
  pk = {f: ([x["peak_mb"] for x in T[f]], [x["peak_mb"] for x in V[f]]) for f in T}
  v.check("S26 row memory", f"{fk:.3f} ⁷ (the int4 file in the same session: {fi:.3f}) | "
          f"{rg(pk[NEW][0] + pk[NEW][1], mb)} |", rdir)
  s0 = open(os.path.join(rdir, "s0_state_initial.txt")).read()
  date = re.search(r"^time: (\d{4}-\d\d-\d\d) ", s0, re.M).group(1)
  v.check("S26 session date", f"in the same session ({date}).", os.path.join(rdir, "s0_state_initial.txt"))
  v.check("S26 memory note", f"the 4,096-token int4 file {rg(pk[NEW][0], mb)} MB in its text runs and "
          f"{rg(pk[NEW][1], mb)} MB in its image runs (the int4 file in the same session: {rg(pk[OLD][0], mb)} and "
          f"{rg(pk[OLD][1], mb)} MB)", rdir)
  it = {f: ([x["init_s"] for x in T[f]], [x["init_s"] for x in V[f]]) for f in T}
  v.check("S26 engine start", f"it started in {rg(it[NEW][0], s1)} s in the benchmark runs and {rg(it[NEW][1], s1)} s in "
          f"the image-question runs (the int4 file in the same session: {rg(it[OLD][0], s1)} s in the benchmark runs "
          f"and {rg(it[OLD][1], s1)} s in the image-question runs)", rdir)
  # the text runs' conditions: the session's first reading, each leg's state line and ready note, and the phone-side
  # clock-ceiling samples (scaling_max / cpuinfo_max per policy; a sample is capped when any policy is under 0.999)
  t0 = int(re.search(r"Thermal Status:\s*(\d+)", s0).group(1))
  b0 = int(re.search(r"battery_temp:\s*temperature:(\d+)", s0).group(1))
  tt = {f: [t for t in rows if t.startswith(f + "-gpu-t")] for f in (NEW, OLD)}
  th = sorted({int(re.search(r"thermal=(\d+)", open(os.path.join(rdir, t + ".state_before.txt")).read()).group(1))
               for f in tt for t in tt[f]})
  clean = all(v_meta(rdir, t)["ready_note"].startswith("clean") for f in tt for t in tt[f])

  def caps(tags):
    n = c = 0
    for t in tags:
      for line in open(os.path.join(rdir, t + ".ceil.txt")):
        pr = [(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", line) if int(b) > 0]
        if pr:
          n += 1
          c += min(a / b for a, b in pr) < 0.999
    return c, n
  (ck, nk), (ci, ni) = caps(tt[NEW]), caps(tt[OLD])
  if clean:
    v.check("S26 conditions", f"The session began at thermal status {t0} with the battery at {b0 / 10:.1f} °C; every run "
            f"started at thermal status {' or '.join(map(str, th))} with no limit on the CPU clock, and during the runs "
            f"the phone held its CPU clock below the maximum in {ck} of {nk} samples taken every half second for the "
            f"4,096-token file and in {ci} of {ni} for the int4 file.", rdir)
  else:
    v.record("S26 conditions", False, "every text run's ready note = clean", rdir)


def v_meta(rdir, tag):
  import json
  return json.load(open(os.path.join(rdir, tag + ".meta.json")))


def pixel_checks(v):
  c_ = v.commas
  r = v.path("pixel_session")
  if not os.path.exists(os.path.join(r, f"{NEW}-cpu-c.err")):
    v.record("Pixel 11 Pro legs", False, "4 legs", r)
    return
  g, p6, p1, cp = (leg(v, r, t) for t in (f"{NEW}-gpu-c", f"{NEW}-gpu-p06", f"{NEW}-gpu-p01", f"{NEW}-cpu-c"))
  fxd = os.path.dirname(v.path("fixtures"))
  fx = {it["id"]: it for it in v.json("fixtures")["items"]}
  k4 = {c["id"]: c for c in v.json("placement_ctx4096")["cases"]}
  inside = lambda item, p: v.mask_hit(os.path.join(fxd, fx[item]["truth"]["mask"]), *p)  # noqa: E731
  q6, q1 = v.point_2d(p6["answer"]), v.point_2d(p1["answer"])
  m6, m1 = v.point_2d(k4["p06_right_shaker"]["content"]), v.point_2d(k4["p01_between_lamp_books"]["content"])
  ok = (g["thought"] is not None and g["answer"] == "Paris" and cp["thought"] is not None and cp["answer"] == "Paris"
        and q6 == m6 and inside("p06_right_shaker", q6) and inside("p01_between_lamp_books", q1)
        and inside("p01_between_lamp_books", m1)
        and g["prefill"][0][0] == 22 and cp["prefill"][0][0] == 22 and p6["prefill"][0][0] == 341
        and p1["prefill"][0][0] == 339 and g["nfd"] == p6["nfd"] == p1["nfd"] == 0)
  v.record("Pixel 11 Pro gate re-read (Paris x2 after a thought, p06 = the Mac's point and inside, p01 and the Mac's p01 "
           "inside, prefill 22 / 341 / 339 / 22, GPU fully delegated)", ok, "all true", r)
  dec = lambda x: f"{x['decode'][0][0]} tokens at {x['decode'][0][1]:.2f} tok/s"  # noqa: E731
  i = lambda x: f"{x['init_s']:.1f}"  # noqa: E731
  ip = sorted({i(p6), i(p1)})
  v.check("Pixel GPU text", f"it answered \"Paris\" after a closed thought ({dec(g)}; the engine took {i(g)} s to start", r)
  v.check("Pixel p06", f"with the Mac's point ({dec(p6)}; the first token {p6['vision_s'] + p6['ttft_s']:.2f} s after the "
          f"request: vision encoder {p6['vision_s'] * 1000:.0f} ms, then {p6['ttft_s']:.2f} s)", r)
  v.check("Pixel p01", f"with [{q1[0]:.0f}, {q1[1]:.0f}] ({dec(p1)}), inside the region, as the Mac's [{m1[0]:.0f}, "
          f"{m1[1]:.0f}] is; the engine started in {'–'.join(ip)} s in those two runs", r)
  thr = re.search(r"number_of_threads: (\d+)", open(os.path.join(r, f"{NEW}-cpu-c.err"), errors="replace").read()).group(1)
  v.check("Pixel CPU", f"CPU, {thr} threads: the same text question, \"Paris\" after a closed thought ({dec(cp)}, time to "
          f"first token {cp['ttft_s']:.2f} s; the engine took {i(cp)} s to start", r)
  gm = [g["peak_mb"], p6["peak_mb"], p1["peak_mb"]]
  v.check("Pixel memory", f"Peak memory (the runtime's line): {c_(round(min(gm)))}–{c_(round(max(gm)))} MB on the GPU, "
          f"{c_(round(cp['peak_mb']))} MB on the CPU.", r)
  v.check("Pixel accuracy note", f"both replies, thought and answer, were longer than on the Mac ({p6['decode'][0][0]} and "
          f"{p1['decode'][0][0]} decoded tokens against {k4['p06_right_shaker']['decode_tokens']} and "
          f"{k4['p01_between_lamp_books']['decode_tokens']})", r)
  tags = (f"{NEW}-gpu-c", f"{NEW}-gpu-p06", f"{NEW}-gpu-p01", f"{NEW}-cpu-c")
  st = {t: v_meta(r, t)["host_start"] for t in tags}
  th = {t: int(re.search(r"thermal=(\d+)", open(os.path.join(r, t + ".state_before.txt")).read()).group(1)) for t in tags}
  first = min(st, key=st.get)
  rest = sorted({th[t] for t in tags if t != first})
  v.check("Pixel thermal", f"The first GPU run (the France question) started at thermal status {th[first]}, the other "
          f"three runs at {' or '.join(map(str, rest))}." if first == f"{NEW}-gpu-c" else "(the France run was not first)",
          r)
  s0 = open(os.path.join(r, "s0_setup.txt")).read() + open(os.path.join(r, "s_gpu.txt")).read()
  dv = ("model=Pixel 11 Pro" in s0 and "release=17" in s0 and "soc=Tensor G6" in s0 and "egl=powervr" in s0)
  v.record("Pixel device line", dv and "**Pixel 11 Pro** (Tensor G6, PowerVR GPU, Android 17)" in v.card,
           "Pixel 11 Pro (Tensor G6, PowerVR GPU, Android 17)", "s0_setup.txt + s_gpu.txt")

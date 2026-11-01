#!/usr/bin/env python3
"""Tables and lines from one score file: does the scene work in the shipped form, and could a cheaper route make the
same clip?

  python tools/demo_scene_check/table.py DIR/score/all.json --fixtures DIR/fixtures.json \
      --subject-label "lfm3 S512 on std" [--app-label "bundle int4_gpu_on"] [--work-min 20] [--swap-margin 8] \
      [--group Q=qwen35 --group B=lfm25_16,lfm25_3 --group C=detector ...] [--ctx 4096] [--decode-tps 37.88] \
      --out DIR/table.md [--json DIR/table.json]

Groups (the arms): a row of the detector + rule or of the detector-to-text arm belongs to "detector", a .litertlm run
to "litertlm", any other row to its model; --group NAME=member[,member] renames or merges them. The subject's model
group and the .litertlm runs are the demo's own forms; every other group is a rival.
Sections:
  summary          per label: right / n (favourable reading in brackets where it differs), per relation kind, on the
                   rule-solvable requests, no point, no answer (thinking not closed, or no reply), new tokens
  best per group   the most right in the favourable column, then the most right in the registered column, then the
                   earlier label (first appearance in the score file = the run order given to score.py)
  lines            work: --subject-label and --app-label each right on at least --work-min requests;
                   swap: every rival group's best (favourable) <= the subject label's right - --swap-margin (a group
                   that breaks it is named: the claim narrows against it)
  candidates       requests the subject label answers right while the best condition of every rival group is wrong in
                   the favourable column, ordered by the distance of the subject's point from the region's edge
                   (largest first, the ones that look right at a glance); strict = every condition of every rival
                   wrong; the --app-label verdict beside each
  objects          when object runs exist: per request, whether the same model, picture and form pointed every
                   reference object (think off), next to the placement: object and place / object only / place only /
                   neither (a rival that finds the object and misses the place fails at the relation)
  budget           --ctx: input + output <= ctx and output tokens median / max for the subject and app labels, with
                   --decode-tps the decode seconds that implies (an estimate; prefill not counted); per thinking label
                   how many thoughts closed
  favourable only  the placement answers that are right only in the favourable reading
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np
from PIL import Image

KINDS = ("lr", "between", "fb", "surface")


def cell(r):
  return "-" if r is None else ("o" if r["ok"] else ("(o)" if r["ok_fav"] else "x"))


def pt(r):
  if r is None:
    return "-"
  if r["point"] is None:
    return "no answer" if (r["how"] or "").startswith(("think on", "bundle: no reply")) else "no point"
  return f"[{r['point'][0]:g}, {r['point'][1]:g}]"


def inside_dist(mask, p):
  ys, xs = np.nonzero(~mask)
  if p is None or not len(xs):
    return None
  return round(float(np.sqrt(((xs + 0.5 - p[0]) ** 2 + (ys + 0.5 - p[1]) ** 2).min())), 1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("score")
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--subject-label", required=True)
  ap.add_argument("--app-label", default=None)
  ap.add_argument("--work-min", type=int, default=20)
  ap.add_argument("--swap-margin", type=int, default=8)
  ap.add_argument("--group", action="append", default=[])
  ap.add_argument("--ctx", type=int, default=None)
  ap.add_argument("--decode-tps", type=float, default=None)
  ap.add_argument("--out", required=True)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  d = json.load(open(a.score))
  subject = d["subject"]
  fx = json.load(open(a.fixtures))
  fx_dir = os.path.dirname(os.path.abspath(a.fixtures))
  items = {it["id"]: it for it in fx["items"]}
  targets = [it["id"] for it in fx["items"]]
  place = [r for r in d["rows"] if r["q_type"] == "place"]
  objs = [r for r in d["rows"] if r["q_type"] == "obj"]
  by = {(r["label"], r["id"]): r for r in place}
  labels = []
  for r in place:
    if r["label"] not in labels:
      labels.append(r["label"])
  rename = {}
  for g in a.group:
    name, members = g.split("=", 1)
    for m in members.split(","):
      rename[m.strip()] = name.strip()

  def group_of(r):
    base = "detector" if r["arm"] in ("CRULE", "CLLM") else ("litertlm" if r["model"].startswith("bundle ") else r["model"])
    return rename.get(base, base)
  own = {rename.get(subject, subject), rename.get("litertlm", "litertlm")}
  groups = {}
  for r in place:
    groups.setdefault(group_of(r), [])
    if r["label"] not in groups[group_of(r)]:
      groups[group_of(r)].append(r["label"])
  rivals = [g for g in groups if g not in own]
  res = {"score": os.path.abspath(a.score), "subject": subject, "subject_label": a.subject_label,
         "app_label": a.app_label, "groups": groups, "rivals": rivals}
  md = [f"# Scene table (`{os.path.basename(a.score)}`; subject {subject})", ""]

  def cnt(xs):
    r_, f = sum(r["ok"] for r in xs), sum(r["ok_fav"] for r in xs)
    return f"{r_}/{len(xs)}" + (f" ({f})" if f != r_ else "")
  n_kind = {k: sum(1 for t in targets if items[t]["rel_kind"] == k) for k in KINDS}
  n_solv = sum(1 for t in targets if items[t]["rule_solvable"])
  md += ["## Summary (registered reading; favourable reading in brackets where it differs)", "",
         "| label | right / " + str(len(targets)) + " | " + " | ".join(f"{k} / {n_kind[k]}" for k in KINDS) +
         f" | rule-solvable / {n_solv} | no point | no answer | new tokens median / max |",
         "|---|---|" + "---|" * len(KINDS) + "---|---|---|---|"]
  summ = {}
  for lab in labels:
    rs = [r for r in place if r["label"] == lab]
    nn = [r["n_new"] for r in rs if r.get("n_new") is not None]
    na = sum(1 for r in rs if (r["how"] or "").startswith(("think on", "bundle: no reply")))
    npt = sum(1 for r in rs if r["point"] is None) - na
    summ[lab] = {"n": len(rs), "right": sum(r["ok"] for r in rs), "right_fav": sum(r["ok_fav"] for r in rs),
                 "by_kind": {k: [sum(r["ok"] for r in rs if r["rel_kind"] == k),
                                 sum(r["ok_fav"] for r in rs if r["rel_kind"] == k)] for k in KINDS},
                 "solvable": [sum(r["ok"] for r in rs if r["rule_solvable"]),
                              sum(r["ok_fav"] for r in rs if r["rule_solvable"])],
                 "no_point": npt, "no_answer": na,
                 "new_median": statistics.median(nn) if nn else None, "new_max": max(nn) if nn else None}
    md.append(f"| {lab} | {cnt(rs)} | " + " | ".join(cnt([r for r in rs if r["rel_kind"] == k]) for k in KINDS) +
              f" | {cnt([r for r in rs if r['rule_solvable']])} | {npt} | {na} | " +
              (f"{statistics.median(nn):g} / {max(nn)}" if nn else "-") + " |")
  res["summary"] = summ
  if a.subject_label not in summ:
    raise SystemExit(f"--subject-label {a.subject_label!r} has no rows; labels: {labels}")
  # best per group
  best = {}
  md += ["", "## Best condition per group (favourable, then registered, then run order)", "",
         "| group | best | right fav / registered | every condition (fav / registered, n) |", "|---|---|---|---|"]
  for g, labs in groups.items():
    sc = sorted((-summ[lab]["right_fav"], -summ[lab]["right"], i, lab) for i, lab in enumerate(labs))
    f, r_, _, lab = sc[0]
    best[g] = {"label": lab, "right_fav": -f, "right": -r_, "n": summ[lab]["n"],
               "all": [{"label": x, "right_fav": summ[x]["right_fav"], "right": summ[x]["right"], "n": summ[x]["n"]}
                       for x in labs]}
    md.append(f"| {g}{'' if g in rivals else ' (own)'} | {lab} | {-f} / {-r_} | " +
              "; ".join(f"{x} {summ[x]['right_fav']} / {summ[x]['right']} ({summ[x]['n']})" for x in labs) + " |")
  res["best"] = best
  # lines
  s = summ[a.subject_label]
  thr = s["right"] - a.swap_margin
  md += ["", "## Lines", ""]
  work = {a.subject_label: s["right"]}
  if a.app_label:
    if a.app_label not in summ:
      raise SystemExit(f"--app-label {a.app_label!r} has no rows")
    work[a.app_label] = summ[a.app_label]["right"]
  for lab, v in work.items():
    md.append(f"- work: {lab} {v}/{summ[lab]['n']} (>= {a.work_min}) -> {'holds' if v >= a.work_min else 'BROKEN'}")
  res["work"] = {lab: {"right": v, "holds": v >= a.work_min} for lab, v in work.items()}
  res["swap"] = {"threshold": thr}
  for g in rivals:
    b = best[g]
    ok = b["right_fav"] <= thr
    res["swap"][g] = {"best": b["label"], "right_fav": b["right_fav"], "holds": ok}
    md.append(f"- swap {g}: best {b['label']} {b['right_fav']}/{b['n']} fav (<= {s['right']} - {a.swap_margin} = {thr})"
              f" -> {'holds' if ok else 'BROKEN: the claim narrows against this group'}")
  # candidates
  md += ["", f"## Candidates ({a.subject_label} right; the best condition of every rival group wrong, favourable)", "",
         "| # | target | request | kind | distance from the edge | " + a.subject_label + " | " +
         " | ".join(f"{g} best ({best[g]['label']})" for g in rivals) + " | strict |" +
         (f" {a.app_label} |" if a.app_label else ""),
         "|" + "---|" * (7 + len(rivals) + (1 if a.app_label else 0))]
  cands = []
  for t in targets:
    sr = by.get((a.subject_label, t))
    if sr is None or not sr["ok"]:
      continue
    if any(by[(best[g]["label"], t)]["ok_fav"] for g in rivals if (best[g]["label"], t) in by):
      continue
    strict = not any(by[(lab, t)]["ok_fav"] for g in rivals for lab in groups[g] if (lab, t) in by)
    mask = np.array(Image.open(os.path.join(fx_dir, items[t]["truth"]["mask"])).convert("L")) > 127
    cands.append((inside_dist(mask, sr["point"]), t, strict))
  cands.sort(key=lambda x: -(x[0] or 0))
  for i, (dist, t, strict) in enumerate(cands, 1):
    it = items[t]
    md.append(f"| {i} | {t} | {it['request']} | {it['rel_kind']} | {dist} | {pt(by[(a.subject_label, t)])} | " +
              " | ".join(f"{cell(by.get((best[g]['label'], t)))} {pt(by.get((best[g]['label'], t)))}" for g in rivals) +
              f" | {'yes' if strict else 'no'} |" +
              (f" {cell(by.get((a.app_label, t)))} {pt(by.get((a.app_label, t)))} |" if a.app_label else ""))
  res["candidates"] = {"targets": [t for _, t, _ in cands], "strict": [t for _, t, s_ in cands if s_],
                       "distance": {t: dd for dd, t, _ in cands}}
  md.append("")
  md.append(f"- candidates: {len(cands)}; strict: {len(res['candidates']['strict'])}" +
            (f"; {a.app_label} right on {sum(1 for _, t, _ in cands if by.get((a.app_label, t), {}).get('ok'))}"
             if a.app_label else ""))
  # per target, compact
  cols = [("subject", a.subject_label)] + ([("app", a.app_label)] if a.app_label else []) + \
      [(f"{g} best", best[g]["label"]) for g in rivals]
  md += ["", "## Per target (o right, x wrong, (o) right only in the favourable reading; point in 0-1000)", "",
         "| target | kind | solvable | " + " | ".join(f"{n} ({lab})" for n, lab in cols) + " |",
         "|---|---|---|" + "---|" * len(cols)]
  for t in targets:
    it = items[t]
    md.append(f"| {t} | {it['rel_kind']} | {'yes' if it['rule_solvable'] else 'no'} | " +
              " | ".join(f"{cell(by.get((lab, t)))} {pt(by.get((lab, t)))}" for _, lab in cols) + " |")
  res["per_target"] = {lab: {t: cell(by.get((lab, t))) for t in targets} for lab in labels}
  # objects
  if objs:
    ob = {(r["model"], r["arm"], r.get("form") or "std", r["id"]): r for r in objs}
    olabels = []
    for r in objs:
      if r["label"] not in olabels:
        olabels.append(r["label"])
    md += ["", "## Objects: the reference objects pointed (think off, the same model, picture and form) next to the "
           "placement", ""]
    res["objects"] = {}
    pairs = [("subject", a.subject_label)] + [(g, best[g]["label"]) for g in rivals]
    for g, plab in pairs:
      pr0 = next(r for r in place if r["label"] == plab)
      model, parm, pform = pr0["model"], pr0["arm"], pr0.get("form") or "std"
      if not any(k[0] == model and k[1] == parm for k in ob):
        continue
      key = "ok" if g == "subject" else "ok_fav"
      cats = {"object and place": [], "object only": [], "place only": [], "neither": [], "no object run": []}
      for t in targets:
        it = items[t]
        refs = [f"{it['photo']}_obj_{r['name'].replace(' ', '-')}" for r in it["refs"]]
        orows = [ob.get((model, parm, pform, oid)) for oid in refs]
        prr = by.get((plab, t))
        if prr is None:
          continue
        if any(x is None for x in orows):
          cats["no object run"].append(t)
          continue
        o_ok, p_ok = all(x[key] for x in orows), prr[key]
        cats["object and place" if o_ok and p_ok else "object only" if o_ok else "place only" if p_ok
             else "neither"].append(t)
      res["objects"][g] = {"placement": plab, "objects": f"{model} {parm} off {pform}", **cats}
      md.append(f"- {g} (placement {plab}; objects {model} {parm} off {pform}): " +
                "; ".join(f"{k} {len(v)}" for k, v in cats.items() if v or k != "no object run"))
      for k, v in cats.items():
        if v:
          md.append(f"  - {k}: {', '.join(v)}")
    md += ["", "| object label | right / " + str(len(fx["objects"])) + " |", "|---|---|"]
    res["object_labels"] = {}
    for lab in olabels:
      rs = [r for r in objs if r["label"] == lab]
      md.append(f"| {lab} | {cnt(rs)} |")
      res["object_labels"][lab] = {"n": len(rs), "right": sum(r["ok"] for r in rs),
                                   "right_fav": sum(r["ok_fav"] for r in rs)}
  # budget
  md += ["", "## Budget", ""]
  res["budget"] = {}
  for lab in [a.subject_label] + ([a.app_label] if a.app_label else []):
    rs = [by[(lab, t)] for t in targets if (lab, t) in by]
    nn = [r["n_new"] for r in rs if r.get("n_new") is not None]
    if not nn:
      continue
    med, mx = statistics.median(nn), max(nn)
    e = {"out_median": med, "out_max": mx}
    line = f"- {lab}: output tokens median {med:g} / max {mx}"
    if a.ctx:
      tot = [r["n_input_ids"] + r["n_new"] for r in rs if r.get("n_input_ids") is not None and r.get("n_new") is not None]
      e.update({"fit": sum(1 for x in tot if x <= a.ctx), "n": len(tot), "largest": max(tot) if tot else None})
      line += f"; input + output <= {a.ctx} in {e['fit']}/{e['n']} (largest {e['largest']})"
    if a.decode_tps:
      e.update({"decode_s_median": round(med / a.decode_tps, 1), "decode_s_max": round(mx / a.decode_tps, 1)})
      line += f"; at {a.decode_tps:g} tok/s {med / a.decode_tps:.1f} s / {mx / a.decode_tps:.1f} s of decode (estimate)"
    res["budget"][lab] = e
    md.append(line)
  for lab in labels:
    rs = [r for r in place if r["label"] == lab and r["think"] == "on" and r.get("n_new") is not None
          and not r["model"].startswith("bundle ")]
    if rs:
      closed = sum(1 for r in rs if r.get("think_tokens"))
      md.append(f"- {lab}: thinking closed in {closed}/{len(rs)}")
      res["budget"].setdefault("closed", {})[lab] = [closed, len(rs)]
  # favourable only
  fav = [r for r in place if r["ok_fav"] and not r["ok"]]
  md += ["", "## Placement answers right only in the favourable reading", ""]
  md += [f"- {r['label']} | {r['id']} | registered {pt(r)} | {r['fav_how']}" for r in fav] or ["- none"]
  res["fav_only"] = [{"label": r["label"], "id": r["id"], "point": r["point"], "fav_how": r["fav_how"]} for r in fav]
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  open(a.out, "w").write("\n".join(md) + "\n")
  if a.json:
    json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
  print("\n".join(md))
  return 0


if __name__ == "__main__":
  sys.exit(main())

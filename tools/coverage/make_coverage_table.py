#!/usr/bin/env python3
"""Join the Hub's most-downloaded open models to their LiteRT-LM conversions in litert-community.

Sources, all public and read without a token:
  /api/models?pipeline_tag=...&sort=downloads    the ranking (30-day downloads, likes, tags, card base_model,
                                                 safetensors parameter count)
  /api/models?author=litert-community            every repo of the org: card base_model and file list
  <repo>/resolve/main/litertlm_manifest.json     for each joined repo that ships one (manifest/SCHEMA.md)

The ranking takes the top N models per pipeline tag (text-generation, image-text-to-text) by the Hub's
30-day download count, ties ordered by repo id. Two kinds of repo are left out of it and counted in the output:
  - quantized re-uploads: the card declares a `base_model:quantized:` relation, or the repo carries the
    `gguf` or `mlx` tag (another runtime's packaging of a model that is ranked, or not, on its own)
  - repos whose safetensors parameter count is known and under --min-params (default 10 M: tiny
    random-weight repos used by test suites)
Quantization-method tags (`fp8`, `8-bit`, `mxfp4` ...) are not used: openai/gpt-oss-20b ships MXFP4
weights as its original release.

Joining a ranked model to litert-community repos. Only repos that hold a runtime bundle (a `.litertlm` file,
or a MediaPipe `.task` file) and whose pipeline tag is unset or one of CONVERSION_PIPELINES count as
conversions: a text-to-image or classifier repo can name an LLM as its base_model without being one.
  1. base_model: every repo whose card `base_model` names the model (Hub ids compared case-insensitively).
  2. name: only when step 1 finds nothing, a repo whose card declares no usable base_model (none, or a
     value without an `org/` part) and whose name equals the model's name after lowercasing, dropping
     `-`, `_`, spaces and a `-litert-lm` / `-litertlm` / `-litert` suffix. These rows carry "(by name)".

For each joined model the table reports, over its joined repos: .litertlm files, a manifest, a file whose
manifest `backends` lists gpu (the schema's verified-to-generate list), a file with external weights
(a TFLiteWeights section in the manifest's section table), and the manifest's image / audio input flags.
A joined model misses whatever none of its repos has; without a manifest the last four are unknown and
the gap reads "no manifest". "no image input" applies to image-text-to-text models only.

  python3 tools/coverage/make_coverage_table.py                   # fetch, write COVERAGE.md + coverage.json
  python3 tools/coverage/make_coverage_table.py --top 50          # top 50 per pipeline tag
  python3 tools/coverage/make_coverage_table.py --save-raw raw.json    # also keep every API answer
  python3 tools/coverage/make_coverage_table.py --from-raw raw.json    # rebuild offline from a kept fetch

HTTP 429 answers are waited out (Retry-After, else the RateLimit header's reset time) and the fetch
continues; 5xx and network errors back off and retry, up to --attempts times per request.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINES = ("text-generation", "image-text-to-text")
KIND = {"text-generation": "text", "image-text-to-text": "image+text"}
POP_FIELDS = ("downloads", "likes", "pipeline_tag", "tags", "cardData", "safetensors", "gated")
ORG_FIELDS = ("downloads", "pipeline_tag", "cardData", "siblings")
MANIFEST = "litertlm_manifest.json"
UA = {"User-Agent": "hf-to-litertlm-coverage/1"}
SUFFIXES = ("-litert-lm", "-litertlm", "-litert")
CONVERSION_PIPELINES = (None, "text-generation", "image-text-to-text", "any-to-any", "audio-text-to-text",
                        "video-text-to-text")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def wait_seconds(err, attempt):
    """Seconds to wait before retrying a failed request."""
    if isinstance(err, urllib.error.HTTPError) and err.code == 429:
        ra = err.headers.get("Retry-After", "") if err.headers else ""
        if ra.strip().isdigit():
            return min(int(ra) + 1, 600)
        m = re.search(r"\bt=(\d+)", (err.headers.get("RateLimit", "") if err.headers else ""))
        if m:
            return min(int(m.group(1)) + 1, 600)
    return min(2 ** attempt, 60)


def get(url, attempts, allow_404=False):
    """-> (body bytes or None on an allowed 404, response headers)."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code == 404 and allow_404:
                return None, e.headers
            if e.code != 429 and e.code < 500:
                raise
            reason = "rate limited (429)" if e.code == 429 else f"HTTP {e.code}"
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            reason, err = f"network error ({e})", e
        if attempt == attempts:
            raise SystemExit(f"giving up after {attempts} attempts: {reason}: {url}")
        s = wait_seconds(err, attempt)
        log(f"{reason}: waiting {s} s, then retrying (attempt {attempt + 1}/{attempts}): {url}")
        time.sleep(s)


def pages(url, attempts):
    """Yield the items of a paged /api/models listing, following the Link: rel="next" header."""
    while url:
        body, headers = get(url, attempts)
        yield from json.loads(body)
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
        url = m.group(1) if m else None


def expand(fields):
    return "".join(f"&expand[]={f}" for f in fields)


def relation(tags, kind):
    return [t.split(":", 2)[2] for t in tags or [] if t.startswith(f"base_model:{kind}:")]


def params(item):
    p = (item.get("safetensors") or {}).get("parameters") or {}
    return sum(p.values()) if p else None


def excluded(item, min_params):
    """-> reason the repo is left out of the ranking, or None."""
    tags = item.get("tags") or []
    if relation(tags, "quantized"):
        return "quantized re-upload (base_model:quantized)"
    if "gguf" in tags or "mlx" in tags:
        return "gguf / mlx repo"
    p = params(item)
    if p is not None and p < min_params:
        return f"under {min_params / 1e6:g} M parameters"
    return None


def fetch(args):
    api = args.api_base.rstrip("/")
    raw = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "api_base": api,
           "pipelines": {}, "org": [], "manifests": {}}
    for tag in args.pipelines:
        url = f"{api}/api/models?pipeline_tag={urllib.parse.quote(tag)}&sort=downloads&direction=-1&limit=100{expand(POP_FIELDS)}"
        items, kept = [], []
        for item in pages(url, args.attempts):
            items.append(item)
            if not excluded(item, args.min_params):
                kept.append(item.get("downloads", 0))
            # stop once N are kept and the listing has gone below the N-th kept count (ties at the cut stay)
            if len(kept) >= args.top and item.get("downloads", 0) < sorted(kept, reverse=True)[args.top - 1]:
                break
            if len(items) >= args.max_scan:
                log(f"{tag}: stopped at --max-scan {args.max_scan} with {len(kept)} kept")
                break
        raw["pipelines"][tag] = items
        log(f"{tag}: {len(items)} listed, {len(kept)} kept")
    url = f"{api}/api/models?author={urllib.parse.quote(args.org)}&limit=1000{expand(ORG_FIELDS)}"
    raw["org"] = list(pages(url, args.attempts))
    log(f"{args.org}: {len(raw['org'])} repos")
    ranked = rank(raw, args)
    want = sorted({r["repo"] for row in ranked for r in link(row["_item"], raw["org"])[1]
                   if MANIFEST in r["files"]})
    for repo in want:
        body, _ = get(f"{api}/{repo}/resolve/main/{MANIFEST}", args.attempts, allow_404=True)
        if body is None:
            raw["manifests"][repo] = {"_error": "listed but 404"}
            continue
        try:
            raw["manifests"][repo] = json.loads(body)
        except ValueError as e:
            raw["manifests"][repo] = {"_error": f"not JSON: {e}"}
    log(f"manifests read: {len(want)}")
    return raw


def rank(raw, args):
    rows = []
    for tag in args.pipelines:
        kept = [i for i in raw["pipelines"].get(tag, []) if not excluded(i, args.min_params)]
        kept.sort(key=lambda i: (-i.get("downloads", 0), i["id"].lower(), i["id"]))
        for n, item in enumerate(kept[:args.top], 1):
            rows.append({"pipeline": tag, "rank": n, "_item": item})
    return rows


def bases(card):
    b = (card or {}).get("base_model")
    if isinstance(b, str):
        b = [b]
    return [x.strip() for x in b or [] if isinstance(x, str) and x.strip()]


def norm_name(name):
    n = name.lower()
    for s in SUFFIXES:
        if n.endswith(s):
            n = n[:-len(s)]
            break
    return re.sub(r"[-_ ]", "", n)


def repo_facts(r):
    files = sorted(s["rfilename"] for s in r.get("siblings") or [])
    return {"repo": r["id"], "files": files, "litertlm": [f for f in files if f.endswith(".litertlm")]}


def link(item, org):
    """-> (method, [repo facts]) for one ranked model; method is "base_model", "name" or None."""
    mid = item["id"].lower()
    org = [r for r in org if r.get("pipeline_tag") in CONVERSION_PIPELINES
           and any(s["rfilename"].endswith((".litertlm", ".task")) for s in r.get("siblings") or [])]
    hits = [r for r in org if mid in {b.lower() for b in bases(r.get("cardData"))}]
    if hits:
        return "base_model", [repo_facts(r) for r in sorted(hits, key=lambda r: r["id"].lower())]
    key = norm_name(item["id"].split("/")[-1])
    hits = [r for r in org
            if not any(b.count("/") == 1 for b in bases(r.get("cardData")))
            and norm_name(r["id"].split("/")[-1]) == key]
    if hits:
        return "name", [repo_facts(r) for r in sorted(hits, key=lambda r: r["id"].lower())]
    return None, []


def manifest_facts(m):
    if not isinstance(m, dict) or "_error" in m:
        return {"manifest": "unreadable" if m else "no"}
    variants = m.get("variants") or []
    caps = (m.get("model") or {}).get("capabilities") or {}
    return {
        "manifest": "yes",
        "gpu_files": sorted(v.get("file", "?") for v in variants if "gpu" in (v.get("backends") or [])),
        "external_weights_files": sorted(v.get("file", "?") for v in variants
                                         if any(s.get("type") == "TFLiteWeights" for s in v.get("sections") or [])),
        "vision": bool(caps.get("vision")),
        "audio": bool(caps.get("audio")),
    }


def build(raw, args):
    rows, org = [], raw["org"]
    for row in rank(raw, args):
        item = row["_item"]
        method, repos = link(item, org)
        for r in repos:
            r.update(manifest_facts(raw["manifests"].get(r["repo"])) if MANIFEST in r["files"] else {"manifest": "no"})
            del r["files"]
        out = {"pipeline": row["pipeline"], "rank": row["rank"], "id": item["id"],
               "downloads_30d": item.get("downloads", 0), "likes": item.get("likes", 0),
               "params_b": round(params(item) / 1e9, 3) if params(item) else None,
               "gated": bool(item.get("gated")), "link": method, "repos": repos}
        if method:
            lm = [r for r in repos if r["litertlm"]]
            mf = [r for r in lm if r["manifest"] == "yes"]
            gaps = []
            if not lm:
                gaps.append("no .litertlm file (.task only)")
            elif not mf:
                gaps.append("no manifest")
            else:
                if not any(r["gpu_files"] for r in mf):
                    gaps.append("no GPU row")
                if not any(r["external_weights_files"] for r in mf):
                    gaps.append("no external weights")
                if row["pipeline"] == "image-text-to-text" and not any(r["vision"] for r in mf):
                    gaps.append("no image input")
            out["gaps"] = gaps
            out["status"] = "gaps" if gaps else "covered"
        else:
            out["status"] = "not linked"
        rows.append(out)
    counted = {}
    for tag in args.pipelines:
        c = {}
        for i in raw["pipelines"].get(tag, []):
            why = excluded(i, args.min_params)
            if why:
                c[why] = c.get(why, 0) + 1
        counted[tag] = dict(sorted(c.items()))
    org_lm = [r for r in org if any(s["rfilename"].endswith(".litertlm") for s in r.get("siblings") or [])]
    return {
        "generated_by": "tools/coverage/make_coverage_table.py",
        "fetched_utc": raw["fetched_utc"],
        "org": args.org,
        "top": args.top,
        "pipelines": list(args.pipelines),
        "min_params": args.min_params,
        "listed": {t: len(raw["pipelines"].get(t, [])) for t in args.pipelines},
        "excluded_from_ranking": counted,
        "org_repos": len(org),
        "org_repos_with_litertlm": len(org_lm),
        "org_repos_with_manifest": sum(any(s["rfilename"] == MANIFEST for s in r.get("siblings") or []) for r in org),
        "rows": rows,
    }


def fmt_int(n):
    return f"{n:,}"


def cell_repos(row):
    mark = " (by name)" if row["link"] == "name" else ""
    return "<br>".join(f"[{r['repo'].split('/')[-1]}](https://huggingface.co/{r['repo']}){mark}" for r in row["repos"])


def yes_no(row, key):
    mf = [r for r in row["repos"] if r["litertlm"] and r["manifest"] == "yes"]
    if not mf:
        return "—"
    return "yes" if any(r[key] for r in mf) else "no"


def inputs(row):
    mf = [r for r in row["repos"] if r["litertlm"] and r["manifest"] == "yes"]
    if not mf:
        return "—"
    got = [k for k in ("image", "audio") if any(r["vision" if k == "image" else "audio"] for r in mf)]
    return ", ".join(got) if got else "text only"


def markdown(t):
    rows = t["rows"]
    by = {s: sorted((r for r in rows if r["status"] == s), key=lambda r: (-r["downloads_30d"], r["id"].lower()))
          for s in ("covered", "gaps", "not linked")}
    head = "| Rank | Model | Downloads (30 d) | Likes | Params (B) |"
    rule = "|---|---|---:|---:|---:|"
    model = lambda r: f"{KIND.get(r['pipeline'], r['pipeline'])} #{r['rank']} | [{r['id']}](https://huggingface.co/{r['id']}) | " \
                      f"{fmt_int(r['downloads_30d'])} | {fmt_int(r['likes'])} | {r['params_b'] if r['params_b'] is not None else '?'}"
    out = [f"# Popular open models and their LiteRT-LM conversions in {t['org']}", "",
           f"Fetched from the Hugging Face Hub API at **{t['fetched_utc']}** (UTC) by `{t['generated_by']}`; "
           f"rebuild with `python3 {t['generated_by']}`.", "",
           f"The top {t['top']} models per pipeline tag ({', '.join(f'`{p}`' for p in t['pipelines'])}) by the Hub's "
           f"30-day download count, ties ordered by repo id. Left out of the ranking and counted here:"]
    for p in t["pipelines"]:
        ex = t["excluded_from_ranking"][p]
        out.append(f"- `{p}` ({t['listed'][p]} listed): " + (", ".join(f"{n} {why}" for why, n in ex.items()) or "none"))
    n = {s: {p: sum(r["pipeline"] == p for r in by[s]) for p in t["pipelines"]} for s in by}
    out += ["",
            f"{t['org']} has {t['org_repos']} repos; {t['org_repos_with_litertlm']} hold a `.litertlm` file and "
            f"{t['org_repos_with_manifest']} ship `litertlm_manifest.json`.", "",
            "| | " + " | ".join(f"`{p}`" for p in t["pipelines"]) + " |",
            "|---|" + "---:|" * len(t["pipelines"])]
    for s, label in (("covered", "joined, no gap"), ("gaps", "joined, with gaps"), ("not linked", "not joined")):
        out.append(f"| {label} | " + " | ".join(str(n[s][p]) for p in t["pipelines"]) + " |")
    out += ["",
            "A model is joined to the repos whose card `base_model` names it; when none does, to a repo whose card "
            "declares no `org/name` base_model and whose name matches (marked *(by name)*). GPU = a file whose "
            "manifest `backends` lists gpu (verified to generate). Ext. weights = a file with a TFLiteWeights "
            "section. Image / audio = the manifest's input flags. A dash = no manifest to read.", ""]
    lhead = head + " litert-community repo | .litertlm files | GPU | Ext. weights | Image / audio |"
    lrule = rule + "---|---:|---|---|---|"
    out += ["## Joined, no gap", ""]
    if by["covered"]:
        out += [lhead, lrule] + [f"| {model(r)} | {cell_repos(r)} | {sum(len(x['litertlm']) for x in r['repos'])} | "
                                 f"{yes_no(r, 'gpu_files')} | {yes_no(r, 'external_weights_files')} | {inputs(r)} |"
                                 for r in by["covered"]]
    else:
        out.append("None.")
    out += ["", "## Joined, with gaps", ""]
    if by["gaps"]:
        out += [lhead + " Gaps |", lrule + "---|"] + [
            f"| {model(r)} | {cell_repos(r)} | {sum(len(x['litertlm']) for x in r['repos'])} | "
            f"{yes_no(r, 'gpu_files')} | {yes_no(r, 'external_weights_files')} | {inputs(r)} | {', '.join(r['gaps'])} |"
            for r in by["gaps"]]
    else:
        out.append("None.")
    out += ["", "## Not joined", "", head, rule] + [f"| {model(r)} |" for r in by["not linked"]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=100, help="models per pipeline tag (default 100)")
    ap.add_argument("--pipelines", default=",".join(PIPELINES), help="comma-separated Hub pipeline tags")
    ap.add_argument("--org", default="litert-community")
    ap.add_argument("--min-params", type=float, default=10e6, help="leave out repos under this many parameters")
    ap.add_argument("--max-scan", type=int, default=2000, help="most listing entries read per pipeline tag")
    ap.add_argument("--attempts", type=int, default=8, help="tries per HTTP request (429 / 5xx / network)")
    ap.add_argument("--api-base", default="https://huggingface.co")
    ap.add_argument("--out-dir", default=HERE, help="where COVERAGE.md and coverage.json go")
    ap.add_argument("--save-raw", help="also write every API answer to this file")
    ap.add_argument("--from-raw", help="build from a file written by --save-raw, no network")
    args = ap.parse_args()
    args.pipelines = [p for p in args.pipelines.split(",") if p]
    if args.from_raw:
        raw = json.load(open(args.from_raw))
    else:
        raw = fetch(args)
        if args.save_raw:
            with open(args.save_raw, "w") as f:
                json.dump(raw, f)
    table = build(raw, args)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "coverage.json"), "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(args.out_dir, "COVERAGE.md"), "w") as f:
        f.write(markdown(table))
    s = {k: sum(r["status"] == k for r in table["rows"]) for k in ("covered", "gaps", "not linked")}
    log(f"wrote {args.out_dir}/COVERAGE.md and coverage.json: {len(table['rows'])} rows {s}")


if __name__ == "__main__":
    main()

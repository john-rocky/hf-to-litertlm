#!/usr/bin/env python3
"""Publish a verified .litertlm + model card to Hugging Face — the `publish` stage.

HARD GUARDRAIL: refuses to upload unless the linked quality-gate report
(scripts/verify_quality.py --json) shows "passed": true. A model that didn't pass
the verify stage cannot reach a public repo through this script (MiniCPM int4 was
garbage — the pipeline must make shipping garbage hard, not easy).

Dry-run by default (validates everything, uploads nothing). Pass --confirm to push.

    python3 scripts/publish_hf.py \
        --model deliverables/MiniCPM5-1B_int8_WORKING.litertlm \
        --card  cards/MiniCPM5-1B-LiteRT.md \
        --report reports/minicpm5-1b-int8.json \
        --repo  mlboydaisuke/MiniCPM5-1B-LiteRT \
        [--confirm]

Note: publish to your own namespace (or any org you have write access to) to get
a shareable link.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Publish a verified .litertlm to HF")
    ap.add_argument("--model", required=True, help="path to the .litertlm to upload")
    ap.add_argument("--card", required=True, help="path to README.md model card")
    ap.add_argument("--report", required=True,
                    help="quality-gate JSON (must show passed=true)")
    ap.add_argument("--repo", required=True, help="target repo id (owner/name)")
    ap.add_argument("--filename", default="model.litertlm",
                    help="name to store the model under in the repo (default model.litertlm)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--confirm", action="store_true",
                    help="actually create the repo and upload (otherwise dry-run)")
    args = ap.parse_args()

    model, card, report = Path(args.model), Path(args.card), Path(args.report)
    errors = []
    for label, p in [("model", model), ("card", card), ("report", report)]:
        if not p.exists():
            errors.append(f"{label} not found: {p}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # HARD GATE: the quality report must show passed=true.
    rep = json.loads(report.read_text())
    if not rep.get("passed"):
        print(f"REFUSING TO PUBLISH: quality gate did not pass "
              f"(score {rep.get('score')}/{rep.get('of')}, "
              f"degenerate={rep.get('degenerate')}).\n"
              f"  Re-run scripts/verify_quality.py and fix the conversion first.",
              file=sys.stderr)
        return 1

    size_gb = model.stat().st_size / 1e9
    print("== publish (HF) ==")
    print(f"   repo:     {args.repo}  ({'private' if args.private else 'public'})")
    print(f"   model:    {model}  ({size_gb:.2f} GB)  -> {args.filename}")
    print(f"   card:     {card}  -> README.md")
    print(f"   gate:     ✅ passed {rep.get('score')}/{rep.get('of')}, "
          f"non-degenerate, ~{rep.get('median_decode_tok_s')} tok/s (Mac)")

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    owner = args.repo.split("/")[0]
    member_of = {o.get("name") for o in who.get("orgs", [])}
    if owner != who.get("name") and owner not in member_of:
        print(f"\n   ⚠️  you ({who.get('name')}) are not '{owner}' and not a member of it.")
        print(f"      Push will fail. Use your own namespace (e.g. {who.get('name')}/...) "
              f"or get added to '{owner}'.")
        if args.confirm:
            return 1

    if not args.confirm:
        print("\n   DRY-RUN — nothing uploaded. Re-run with --confirm to push.")
        return 0

    print("\n   creating repo + uploading...")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(model), path_in_repo=args.filename,
                    repo_id=args.repo, repo_type="model")
    print(f"   ✅ published: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

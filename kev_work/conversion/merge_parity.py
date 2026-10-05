"""Merge parity: the full-weight checkpoint written by the author's merge script, loaded by the author's loader
(Checkpoint(dir).full == True, saved_dtype float32), against the adapter checkpoint (LoRA merged at load), on every
request and question of the fixtures: argmax agreement, max / mean |dp|, max |dz| (z_post), max |dh| over the read-out
hidden states (h_sel = h[decide] and h[</opt>]) -> results/merge_parity_0.8b.json. Bar: argmax 100% and max |dp| <= 1e-4.
--model 4b compares oracle_4b (jaredpalmer/kev-4b@591dcb5b) with oracle_merged_4b (merged/kev-4b-v1.0)
-> results/merge_parity_4b.json."""
import argparse
import json
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
BAR_DP = 1e-4


MODELS = {"0.8b": ("0.8b", "merged", "jaredpalmer/kev-0.8b@788ddbdd", "merged/kev-0.8b-v1.0/checkpoint"),
          "4b": ("4b", "merged_4b", "jaredpalmer/kev-4b@591dcb5b", "merged/kev-4b-v1.0/checkpoint")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="0.8b")
    tag_a, tag_b, ref_name, cand_dir = MODELS[ap.parse_args().model]
    a = json.loads((K / f"oracle/oracle_{tag_a}.json").read_text())
    b = json.loads((K / f"oracle/oracle_{tag_b}.json").read_text())
    assert a["fixtures_sha256"] == b["fixtures_sha256"]
    ha, hb = np.load(K / f"oracle/hidden_{tag_a}.npz"), np.load(K / f"oracle/hidden_{tag_b}.npz")
    qa = {(q["id"], q["qid"]): q for q in a["questions"]}
    qb = {(q["id"], q["qid"]): q for q in b["questions"]}
    assert set(qa) == set(qb), len(set(qa) ^ set(qb))
    per, dps, dzs, dhs = [], [], [], []
    for key in qa:
        x, y = qa[key], qb[key]
        assert x["row_ids"] == y["row_ids"] and x["decide_idx"] == y["decide_idx"] and x["opt_idx"] == y["opt_idx"], key
        dp = float(np.max(np.abs(np.array(x["probs"]) - np.array(y["probs"]))))
        dz = float(np.max(np.abs(np.array(x["z_post"]) - np.array(y["z_post"]))))
        dh = float(np.max(np.abs(ha[f"{key[0]}/{key[1]}"] - hb[f"{key[0]}/{key[1]}"])))
        dps.append(dp); dzs.append(dz); dhs.append(dh)
        per.append({"id": key[0], "qid": key[1], "argmax_equal": x["argmax_key"] == y["argmax_key"], "max_abs_dp": dp, "max_abs_dz": dz, "max_abs_dh_sel": dh})
    full_keys = [k for k in ha.files if k.endswith("/full")]
    full = {k: float(np.max(np.abs(ha[k] - hb[k]))) for k in full_keys}
    real = [p for p in per if not p["id"].startswith("red_arm")]
    agree = sum(p["argmax_equal"] for p in per)
    out = {
        "reference": f"oracle/oracle_{tag_a}.json ({ref_name}, adapter merged at load by kev.checkpoint)",
        "candidate": f"oracle/oracle_{tag_b}.json ({cand_dir} via the author's loader, full weights fp32)",
        "questions": len(per), "argmax_equal": agree, "argmax_equal_excluding_red_arm": [sum(p["argmax_equal"] for p in real), len(real)],
        "max_abs_dp": max(dps), "mean_abs_dp": float(np.mean(dps)), "max_abs_dz": max(dzs), "max_abs_dh_sel": max(dhs),
        "full_row_hidden_max_abs": full,
        "bar": {"argmax": "100%", "max_abs_dp": BAR_DP}, "pass": agree == len(per) and max(dps) <= BAR_DP,
        "worst": sorted(per, key=lambda p: -p["max_abs_dp"])[:5],
        "candidate_checkpoint": {"full": b.get("checkpoint"), "load": "Checkpoint(dir).load('cpu', LoadOptions(dtype=torch.float32))"},
        "per_question": per,
    }
    path = K / f"results/merge_parity_{tag_a}.json"
    assert not path.exists(), f"refusing to overwrite {path}"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("questions", "argmax_equal", "max_abs_dp", "mean_abs_dp", "max_abs_dz", "max_abs_dh_sel",
                                          "full_row_hidden_max_abs", "pass", "worst")}, indent=1))


if __name__ == "__main__":
    main()

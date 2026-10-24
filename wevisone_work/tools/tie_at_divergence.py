import json, glob, statistics
out = {}
for lab in ("wi8_e16a16_act32_gpu", "wi8_e16a16_act32_cpu", "wi8_e16a16_actnone_gpu", "wi4_e16a16_act32_gpu"):
    gaps = []
    for p in sorted(glob.glob("ref/B/*.json")):
        r = p.split("/")[-1][:-5]
        b = json.load(open(p))["new_ids"]
        t = json.load(open(f"r2/suite/{lab}/{r}.json"))["new_ids"]
        tf = json.load(open(f"out/r2/hf_tf/B/{r}.json"))
        fd = next((i for i, (x, y) in enumerate(zip(b, t)) if x != y), None)
        if fd is None or fd >= len(tf["logit_gap"]):
            continue
        gaps.append((round(tf["logit_gap"][fd], 3), r, fd, t[fd] == tf["top2"][fd]))
    gaps.sort()
    out[lab] = {"n_div": len(gaps), "max_gap": gaps[-1][0] if gaps else None,
                "median_gap": statistics.median(g[0] for g in gaps) if gaps else None,
                "picked_hf_top2": sum(g[3] for g in gaps), "worst5": gaps[-5:]}
    print(lab, out[lab]["n_div"], "divergent rows; logit gap at the first divergence: median",
          out[lab]["median_gap"], "max", out[lab]["max_gap"], "| test token = HF top-2:", out[lab]["picked_hf_top2"],
          "| largest:", gaps[-3:])
json.dump(out, open("r2/tie_at_divergence.json", "w"), indent=1)

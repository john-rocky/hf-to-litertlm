"""Port (funasr_port.py, no funasr import) vs the funasr oracle dumps + transcripts.

Per clip (25):
  ref    : vendored funasr path on the true-length waveform (kaldi fbank via torch.fft + apply_lfr, encoder /
           adaptor on the unpadded [1, L, 560]) -> checks the vendored modules;
  modin  : forward_ref on the ORACLE LFR dump -> module parity with identical input;
  graph  : FunAsrNanoAudioEncoder on the runtime window (30.24 s, zero-padded) = the exported computation;
  graphn : graph with n_valid overridden to the true sample count (isolates the in-graph length inference).
Metrics vs oracle: LFR rows < L, encoder rows < L, adaptor rows < fake_token_len (the rows the LLM sees).
Transcripts: lm_native (Qwen3ForCausalLM fp32) greedy with KV cache on prefix(18) + features[:ftl] + suffix(5),
for the graph features (primary) and the ref features.
Run with ~/venvs/lt094dev/bin/python. Writes port_parity.json, port_transcripts.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import funasr_port as P  # noqa: E402

DUMPS = os.path.join(C.OUT, "oracle_dumps")
TOL_COS, TOL_ABS = 0.99999, 1e-2


def cmp(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "rel": float(d.max() / (np.abs(b).max() + 1e-12)),
            "cos": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
            "ok": bool(d.max() <= TOL_ABS and (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30) >= TOL_COS)}


class Greedy:
    def __init__(self, path):
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        assert torch.equal(self.lm.lm_head.weight, self.lm.model.embed_tokens.weight)
        emb = self.lm.model.embed_tokens
        with torch.no_grad():
            self.pre = emb(torch.tensor([C.PREFIX_IDS]))
            self.suf = emb(torch.tensor([C.SUFFIX_IDS]))

    @torch.no_grad()
    def __call__(self, feats, max_new=512):
        """feats [n, 1024] fp32 -> (gen ids incl. the stop token, text, text_tn)."""
        e = torch.cat([self.pre, feats[None].float(), self.suf], dim=1)
        lm, emb = self.lm, self.lm.model.embed_tokens
        o = lm.model(inputs_embeds=e, use_cache=True)
        past, lg = o.past_key_values, lm.lm_head(o.last_hidden_state[:, -1])[0]
        ids = []
        for _ in range(max_new):
            nid = int(torch.argmax(lg))
            ids.append(nid)
            if nid in (C.IM_END, C.ENDOFTEXT):
                break
            o = lm.model(inputs_embeds=emb(torch.tensor([[nid]])), past_key_values=past, use_cache=True)
            past, lg = o.past_key_values, lm.lm_head(o.last_hidden_state[:, -1])[0]
        text, text_tn = C.postprocess(self.tok.decode(ids, skip_special_tokens=True))
        return ids, text, text_tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-lm", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    oracle = json.load(open(os.path.join(C.WORK, "oracle_transcripts.json")))
    orow = {r["id"]: r for r in oracle["rows"]}
    enc, n_t = P.load_encoder(os.path.join(C.OUT, "hf_vllm", "model.safetensors"))
    lm = None if args.no_lm else Greedy(os.path.join(C.OUT, "lm_native"))

    par, trs = [], []
    for row in meta:
        fid = row["id"]
        o = orow[fid]
        L, ftl = o["L"], o["fake_token_len"]
        d_lfr = np.load(os.path.join(DUMPS, f"{fid}_lfr.npy"))
        d_enc = np.load(os.path.join(DUMPS, f"{fid}_enc.npy"))
        d_adp = np.load(os.path.join(DUMPS, f"{fid}_adp.npy"))
        wav = P.read_wav_i16(os.path.join(C.WORK, row["file"]))
        assert len(wav) == row["n_samples"]
        rec = {"id": fid, "n_samples": len(wav), "L_oracle": L, "ftl_oracle": ftl}
        with torch.no_grad():
            # ref: vendored funasr path on the true-length clip
            wave = torch.from_numpy(wav.astype(np.float32) / 32768.0)
            lfr_r = P.ref_frontend(wave)
            enc_r, adp_r = enc.forward_ref(lfr_r)
            # modin: vendored modules on the oracle's own LFR
            enc_m, adp_m = enc.forward_ref(torch.from_numpy(d_lfr))
            # graph: the exported computation on the runtime window
            audio = P.frame_window(wav)
            t0 = time.perf_counter()
            g = enc.forward_debug(audio)
            t_graph = time.perf_counter() - t0
            gn = enc.forward_debug(audio, n_valid=torch.tensor([[float(len(wav))]]))
        Lg, ftlg = int(g["L"].item()), int(g["ftl"].item())
        mask_count = int(g["mask"].sum().item())
        rec.update({
            "ref": {"L": lfr_r.shape[1], "lfr": cmp(lfr_r[0], d_lfr[0]), "enc": cmp(enc_r[0], d_enc[0]),
                    "adp_ftl": cmp(adp_r[0, :ftl], d_adp[0, :ftl]), "adp_all": cmp(adp_r[0], d_adp[0])},
            "modin": {"enc": cmp(enc_m[0], d_enc[0]), "adp_ftl": cmp(adp_m[0, :ftl], d_adp[0, :ftl])},
            "graph": {"L": Lg, "ftl": ftlg, "mask_count": mask_count, "L_match": Lg == L, "ftl_match": ftlg == ftl,
                      "lfr": cmp(g["lfr"][0, :L], d_lfr[0]), "enc": cmp(g["enc"][0, :L], d_enc[0]),
                      "adp_ftl": cmp(g["adp"][0, :ftl], d_adp[0, :ftl]), "eager_s": round(t_graph, 3)},
            "graphn": {"L": int(gn["L"].item()), "ftl": int(gn["ftl"].item()),
                       "lfr": cmp(gn["lfr"][0, :L], d_lfr[0]), "enc": cmp(gn["enc"][0, :L], d_enc[0]),
                       "adp_ftl": cmp(gn["adp"][0, :ftl], d_adp[0, :ftl])},
        })
        if not (Lg == L):
            # rows the graph itself exposes (its own L / ftl) against the oracle's first rows
            k = min(Lg, L)
            kf = min(ftlg, ftl)
            rec["graph"]["lfr_min_rows"] = cmp(g["lfr"][0, :k], d_lfr[0, :k])
            rec["graph"]["adp_min_ftl"] = cmp(g["adp"][0, :kf], d_adp[0, :kf])
        par.append(rec)
        line = (f"{fid:12s} L {L}/{Lg} ftl {ftl}/{ftlg} | ref lfr {rec['ref']['lfr']['max_abs']:.2e} enc {rec['ref']['enc']['max_abs']:.2e} "
                f"adp {rec['ref']['adp_ftl']['max_abs']:.2e} | graph lfr {rec['graph']['lfr']['max_abs']:.2e} "
                f"enc {rec['graph']['enc']['max_abs']:.2e} cos {rec['graph']['enc']['cos']:.7f} adp {rec['graph']['adp_ftl']['max_abs']:.2e} "
                f"cos {rec['graph']['adp_ftl']['cos']:.7f} | graphn adp {rec['graphn']['adp_ftl']['max_abs']:.2e}")
        print(line, flush=True)

        if lm is not None:
            t0 = time.perf_counter()
            ids_g, text_g, tn_g = lm(g["features"][0, :mask_count])
            t_lm = time.perf_counter() - t0
            ids_r, text_r, _ = lm(adp_r[0, :ftl])
            tr = {"id": fid, "oracle_text": o["text"], "text": text_g, "text_tn": tn_g, "gen_ids": ids_g,
                  "text_equal": text_g == o["text"], "gen_ids_equal": ids_g == o["gen_ids"],
                  "ref_text": text_r, "ref_text_equal": text_r == o["text"], "ref_gen_ids_equal": ids_r == o["gen_ids"],
                  "valid_tokens": mask_count, "lm_s": round(t_lm, 3)}
            trs.append(tr)
            flag = "==" if tr["text_equal"] and tr["gen_ids_equal"] else "!="
            print(f"   {flag} {text_g}" + ("" if tr["text_equal"] else f"\n   oracle: {o['text']}"), flush=True)

    summ = {
        "tolerance": {"cos_min": TOL_COS, "max_abs_max": TOL_ABS},
        "n_clips": len(par), "encoder_tensors_loaded": n_t,
        "ref_all_ok": all(r["ref"]["lfr"]["ok"] and r["ref"]["enc"]["ok"] and r["ref"]["adp_ftl"]["ok"] for r in par),
        "graph_all_ok": all(r["graph"]["lfr"]["ok"] and r["graph"]["enc"]["ok"] and r["graph"]["adp_ftl"]["ok"] for r in par),
        "graphn_all_ok": all(r["graphn"]["lfr"]["ok"] and r["graphn"]["enc"]["ok"] and r["graphn"]["adp_ftl"]["ok"] for r in par),
        "graph_L_match": sum(r["graph"]["L_match"] for r in par), "graph_ftl_match": sum(r["graph"]["ftl_match"] for r in par),
        "graph_fail_ids": [r["id"] for r in par if not (r["graph"]["lfr"]["ok"] and r["graph"]["enc"]["ok"] and r["graph"]["adp_ftl"]["ok"])],
        "worst": {m: {k: max(r[m][k]["max_abs"] for r in par) for k in ["lfr", "enc", "adp_ftl"]} for m in ["ref", "graph", "graphn"]},
        "worst_cos": {m: {k: min(r[m][k]["cos"] for r in par) for k in ["lfr", "enc", "adp_ftl"]} for m in ["ref", "graph", "graphn"]},
    }
    summ["worst"]["modin"] = {k: max(r["modin"][k]["max_abs"] for r in par) for k in ["enc", "adp_ftl"]}
    summ["worst_cos"]["modin"] = {k: min(r["modin"][k]["cos"] for r in par) for k in ["enc", "adp_ftl"]}
    with open(os.path.join(C.WORK, "port_parity.json"), "w") as f:
        json.dump({"summary": summ, "rows": par}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    if trs:
        by_id = {r["id"]: r for r in meta}
        wer = C.corpus_wer(trs, by_id)
        ts = {"text_match": sum(t["text_equal"] for t in trs), "gen_ids_match": sum(t["gen_ids_equal"] for t in trs),
              "ref_text_match": sum(t["ref_text_equal"] for t in trs), "n": len(trs),
              "wer_en": {k: wer[k] for k in ["errors", "words", "wer"]},
              "mismatches": [{"id": t["id"], "port": t["text"], "oracle": t["oracle_text"]} for t in trs if not t["text_equal"]]}
        with open(os.path.join(C.WORK, "port_transcripts.json"), "w") as f:
            json.dump({"summary": ts, "rows": trs}, f, ensure_ascii=False, indent=1)
        print(json.dumps(ts, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

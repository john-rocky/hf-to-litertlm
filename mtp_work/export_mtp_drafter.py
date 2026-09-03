#!/usr/bin/env python3
"""Export the Qwen3.5 MTP drafter graph (P2) — the 15 `mtp.*` tensors + tied head
as a single-signature tflite matching the LiteRT-LM drafter contract
(llm_litert_mtp_drafter.cc @ 90f42140, read 2026-08-31):

  signature 0 inputs:
    activations      [1, 1, 2*hidden] fp32 — host-side concat, EMBEDDING FIRST
                     (ConcatenateEmbeddingsAndActivations memcpys the embedding
                     at offset 0, previous hidden/projection at offset H)
    input_pos        [1] int32 — the runtime writes position-1 and holds it
                     FIXED across the G chained invocations
    mask             [1, 1, 1, cache_len] fp32 — DECLARED BUT UNUSED. The
                     released 0.16.0 runtime initializes+fills
                     active_drafter_input_buffers_["mask"] UNCONDITIONALLY
                     (llm_litert_mtp_drafter.cc @ c2ab9ab8; found the hard
                     way: without the input, operator[] default-constructs an
                     empty TensorBuffer and PackedSize() dies with
                     INVALID_ARGUMENT at litert_tensor_buffer.h:620). Main
                     (90f42140) guards on input_attn_mask.has_value(), so
                     declaring it is forward-compatible. Its CONTENT is
                     ignored: the runtime's fill allows cache slot
                     input_pos, which is stale-by-construction at drafter
                     time (its true pair needs this round's bonus token).
    kv_cache_k_mtp   [1, kv_heads, cache_len, head_dim] fp32 — bound by name
                     from the BASE bundle's state buffers (Duplicate())
    kv_cache_v_mtp   [1, kv_heads, head_dim, cache_len] fp32
  outputs:
    logits                [1, 1, vocab] fp32 (tied head)
    projected_activations [1, 1, hidden] fp32 (post-mtp.norm; REQUIRED fp32 —
                          fed back as the next chained step's hidden half)

  The attention mask is built in-graph from input_pos instead: attend cache
  slots j < input_pos, plus the own position's k/v appended in-graph. This is
  exactly the P0 oracle `fixed` mode (measured E[G=3] = 2.95 (0.8B) /
  3.06 (2B), the on-device reference). Unused signature inputs survive the
  litert-torch converter (verified), so no dummy consumption is needed.

Build-time equivalence gate (always on): loads the real checkpoint
(transformers bf16), traces a short greedy decode, and asserts this module ==
the P0 oracle's manual_step (which was itself asserted == transformers
layer.forward) at several positions including chained steps — then asserts
the converted tflite == the torch module on the same inputs, with a
garbage-filled stale slot to prove the in-graph mask excludes it.

Usage (lt094dev + the P1 worktree on PYTHONPATH):
  PYTHONPATH=~/code/litert-torch-mtp ~/venvs/lt094dev/bin/python3 \
      mtp_work/export_mtp_drafter.py Qwen/Qwen3.5-0.8B mtp_work/out_08b/drafter.tflite
Then quantize (wi8fc, same recipe as the bundle) with --quantize, or leave
float for interpreter-level gates.
"""

import argparse
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)


def find_shards(model_id):
    if os.path.isdir(model_id):
        shards = sorted(glob.glob(os.path.join(model_id, "*.safetensors")))
    else:
        pat = os.path.expanduser(
            "~/.cache/huggingface/hub/models--"
            + model_id.replace("/", "--") + "/snapshots/*/*.safetensors")
        shards = sorted(glob.glob(pat))
    assert shards, f"no safetensors for {model_id}"
    return shards


def load_tensors(model_id):
    """15 mtp.* tensors + the tied embedding table, fp32."""
    from safetensors import safe_open
    mtp, table = {}, None
    for sh in find_shards(model_id):
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    mtp[k] = f.get_tensor(k).to(torch.float32)
                elif k.endswith("embed_tokens.weight"):
                    table = f.get_tensor(k).to(torch.float32)
    assert len(mtp) == 15, f"expected 15 mtp.* tensors, got {len(mtp)}"
    assert table is not None, "embed_tokens.weight not found"
    return mtp, table


SPECIAL_BLOCK_START = 247000  # P0 convention: keep ids [0,K) + [247000,V)


class Qwen35MtpDrafter(torch.nn.Module):
    """One draft step. The runtime chains G invocations at fixed input_pos.

    topk > 0 switches the head to the P0 top-K variant: a sliced tied-table
    matmul over ids = [0,K) ++ [SPECIAL_BLOCK_START,V), re-expanded in-graph to
    full-vocab logits with a -30000 constant filling the dropped id range, so
    the runtime's greedy argmax lands on real token ids with no runtime-side
    remap. Verify keeps the full head — greedy-exactness is untouched; the only
    cost is acceptance (P0: top-32k E[G=3] 2.95->2.75 (0.8B) / 3.06->2.86 (2B))
    and the win is the per-draft-step head read (254 MB -> ~35 MB int8 on
    0.8B), which is the dominant drafter cost on weight-read-bound phones.
    """

    def __init__(self, text_config, mtp_tensors, tied_table, cache_len,
                 topk=0):
        super().__init__()
        h = text_config.hidden_size
        eps = text_config.rms_norm_eps
        self.n_heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.n_rep = self.n_heads // self.kv_heads
        self.scaling = self.head_dim ** -0.5
        self.hidden = h

        self.ne = Qwen3_5RMSNorm(h, eps=eps)
        self.nh = Qwen3_5RMSNorm(h, eps=eps)
        self.fc = torch.nn.Linear(2 * h, h, bias=False)
        self.input_layernorm = Qwen3_5RMSNorm(h, eps=eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(h, eps=eps)
        self.norm = Qwen3_5RMSNorm(h, eps=eps)  # mtp.norm
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.q_proj = torch.nn.Linear(h, self.n_heads * 2 * self.head_dim,
                                      bias=False)
        self.k_proj = torch.nn.Linear(h, self.kv_heads * self.head_dim,
                                      bias=False)
        self.v_proj = torch.nn.Linear(h, self.kv_heads * self.head_dim,
                                      bias=False)
        self.o_proj = torch.nn.Linear(self.n_heads * self.head_dim, h,
                                      bias=False)
        inter = text_config.intermediate_size
        self.gate_proj = torch.nn.Linear(h, inter, bias=False)
        self.up_proj = torch.nn.Linear(h, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, h, bias=False)
        V = tied_table.shape[0]
        self.topk = topk
        if topk:
            assert 0 < topk < SPECIAL_BLOCK_START <= V
            self.head_ids = torch.cat([
                torch.arange(topk), torch.arange(SPECIAL_BLOCK_START, V)])
            self.head = torch.nn.Linear(h, self.head_ids.numel(), bias=False)
            self.register_buffer(
                "pad_fill",
                torch.full((1, 1, SPECIAL_BLOCK_START - topk), -30000.0),
                persistent=False)
        else:
            self.head_ids = None
            self.head = torch.nn.Linear(h, V, bias=False)
        self.rotary = Qwen3_5TextRotaryEmbedding(config=text_config)

        t = mtp_tensors
        with torch.no_grad():
            self.ne.weight.copy_(t["mtp.pre_fc_norm_embedding.weight"])
            self.nh.weight.copy_(t["mtp.pre_fc_norm_hidden.weight"])
            self.fc.weight.copy_(t["mtp.fc.weight"])
            self.norm.weight.copy_(t["mtp.norm.weight"])
            p = "mtp.layers.0."
            self.input_layernorm.weight.copy_(t[p + "input_layernorm.weight"])
            self.post_attention_layernorm.weight.copy_(
                t[p + "post_attention_layernorm.weight"])
            self.q_norm.weight.copy_(t[p + "self_attn.q_norm.weight"])
            self.k_norm.weight.copy_(t[p + "self_attn.k_norm.weight"])
            self.q_proj.weight.copy_(t[p + "self_attn.q_proj.weight"])
            self.k_proj.weight.copy_(t[p + "self_attn.k_proj.weight"])
            self.v_proj.weight.copy_(t[p + "self_attn.v_proj.weight"])
            self.o_proj.weight.copy_(t[p + "self_attn.o_proj.weight"])
            self.gate_proj.weight.copy_(t[p + "mlp.gate_proj.weight"])
            self.up_proj.weight.copy_(t[p + "mlp.up_proj.weight"])
            self.down_proj.weight.copy_(t[p + "mlp.down_proj.weight"])
            self.head.weight.copy_(tied_table[self.head_ids] if topk
                                   else tied_table)

        self.register_buffer(
            "slot_idx", torch.arange(cache_len, dtype=torch.int32),
            persistent=False)
        self.register_buffer(
            "self_allow", torch.zeros(1, dtype=torch.float32),
            persistent=False)

    def forward(self, activations, input_pos, mask,
              kv_cache_k_mtp, kv_cache_v_mtp):
        del mask  # declared for the 0.16.0 runtime contract; see docstring
        emb = activations[..., : self.hidden]
        hid = activations[..., self.hidden:]
        x = self.fc(torch.cat([self.ne(emb), self.nh(hid)], dim=-1))

        ln = self.input_layernorm(x)
        qg = self.q_proj(ln).reshape(1, 1, self.n_heads, 2 * self.head_dim)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(1, 1, self.n_heads * self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [1, nh, 1, hd]
        k = self.k_norm(
            self.k_proj(ln).reshape(1, 1, self.kv_heads, self.head_dim)
        ).transpose(1, 2)                   # [1, kvh, 1, hd]
        v = self.v_proj(ln).reshape(1, 1, self.kv_heads, self.head_dim)
        v = v.transpose(1, 2)               # [1, kvh, 1, hd]

        pos = input_pos.reshape(1, 1)
        cos, sin = self.rotary(x, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA without repeat_interleave/cat/transpose on cache-size tensors:
        # the first drafter export materialized [1, nh, L+1, hd] keys/values
        # (2x33 MB fp32 per invoke) and ran at 127 ms — 5.4x a full decode
        # step. Group heads as [1, kvh, n_rep, hd] (heads are contiguous per
        # kv group under repeat_interleave: h = kv*n_rep + r) and batch-matmul
        # straight against the cache layouts; adj-style transposes stay
        # virtual inside BATCH_MATMUL.
        q_g = q.reshape(1, self.kv_heads, self.n_rep, self.head_dim)
        # scores vs teacher cache: [1,kvh,r,hd] @ [1,kvh,hd,L] -> [1,kvh,r,L]
        scores_cache = torch.matmul(
            q_g, kv_cache_k_mtp.transpose(2, 3)) * self.scaling
        # own-position score: [1,kvh,r,1]
        scores_own = torch.matmul(q_g, k.transpose(2, 3)) * self.scaling
        # teacher slots j < input_pos allowed; slot input_pos itself is stale
        # garbage -> masked; own position (last column) always allowed.
        stale = (self.slot_idx >= input_pos[0]).to(torch.float32) * -30000.0
        scores = torch.cat([scores_cache + stale, scores_own], dim=-1)
        w = torch.softmax(scores, dim=-1)
        w_cache = w[..., :-1]                      # [1, kvh, r, L]
        w_own = w[..., -1:]                        # [1, kvh, r, 1]
        # attn: values kept cache-layout [1,kvh,hd,L]; w_cache @ values^T
        attn = torch.matmul(w_cache, kv_cache_v_mtp.transpose(2, 3))
        attn = attn + w_own * v                    # [1, kvh, r, hd]
        attn = attn.reshape(1, 1, self.n_heads * self.head_dim)
        attn = attn * torch.sigmoid(gate)

        h = x + self.o_proj(attn)
        ln2 = self.post_attention_layernorm(h)
        h = h + self.down_proj(
            torch.nn.functional.silu(self.gate_proj(ln2)) * self.up_proj(ln2))
        out = self.norm(h)
        if self.topk:
            s = self.head(out)
            logits = torch.cat(
                [s[..., : self.topk], self.pad_fill, s[..., self.topk:]],
                dim=-1)
        else:
            logits = self.head(out)
        return {
            "logits": logits,
            "projected_activations": out,
        }

    def ref_argmax(self, hidden):
        """Full-vocab-id argmax of this head for a [h] hidden — gate reference."""
        s = hidden @ self.head.weight.T
        if self.topk:
            return int(self.head_ids[int(s.argmax())])
        return int(s.argmax())


@torch.no_grad()
def equivalence_gate(module, model_id, device, n_pos=6, chain=2):
    """module == oracle manual_step on a real trace, incl. chained steps."""
    from qwen35_mtp_alpha_oracle import (
        Drafter, find_shards as oracle_shards, manual_step, teacher_and_step1,
        trace_greedy)
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16).to(device).eval()
    tm = model.model.language_model
    oracle = Drafter(model.config.text_config, oracle_shards(model_id), device)

    msgs = [{"role": "user", "content":
             "Explain photosynthesis in one sentence."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True)
    ids = list(enc.ids) if hasattr(enc, "ids") else (
        list(enc[0].ids) if enc and hasattr(enc[0], "ids") else list(enc))
    eos = {tok.eos_token_id}
    seq, H = trace_greedy(model, ids, 48, eos, device)
    X, H1h, _, K, V = teacher_and_step1(oracle, tm, seq, H, device)

    module = module.to(device)
    L = module.slot_idx.shape[0]
    kvh, hd = module.kv_heads, module.head_dim
    torch.manual_seed(0)
    # random content: proves the mask input's values cannot influence outputs
    dummy_mask = torch.randn(1, 1, 1, L, device=device) * 1e4

    positions = [len(ids) + j * max(1, (len(seq) - len(ids) - 3) // n_pos)
                 for j in range(n_pos)]
    positions = [p for p in positions if p < len(seq) - 1]
    worst = 0.0
    for p in positions:
        k_cache = torch.randn(1, kvh, L, hd, device=device)  # garbage incl.
        v_cache = torch.randn(1, kvh, hd, L, device=device)  # slot p-1
        k_cache[:, :, : p - 1] = K[:, :, : p - 1]
        v_cache[:, :, :, : p - 1] = V[:, :, : p - 1].transpose(2, 3)
        pos_t = torch.tensor([p - 1], dtype=torch.int32, device=device)

        tok_mine = tok_ref = seq[p]
        h_mine = h_ref = H[p - 1].float()
        for step in range(chain):
            # mine: runtime-shaped input emb(tok) ++ prev hidden/projection
            e = tm.embed_tokens(
                torch.tensor([tok_mine], device=device)).float()
            act = torch.cat([e, h_mine[None]], dim=-1)[None]
            got = module(act, pos_t, dummy_mask, k_cache, v_cache)
            proj = got["projected_activations"][0, 0]
            am_mine = int(got["logits"][0, 0].argmax())
            # reference: oracle weights end to end (fc/ne/nh + manual_step,
            # itself asserted == transformers layer.forward in P0)
            e_ref = tm.embed_tokens(
                torch.tensor([tok_ref], device=device)).float()
            x_ref = oracle.fc(torch.cat(
                [oracle.ne(e_ref), oracle.nh(h_ref[None])], dim=-1))
            h_ref, _, _ = manual_step(
                oracle, tm, x_ref[None], p - 1,
                K[:, :, : p - 1], V[:, :, : p - 1], device)
            am_ref = module.ref_argmax(h_ref)
            d = (h_ref - proj).abs().max().item()
            worst = max(worst, d)
            assert d < 2e-3, f"module vs manual_step diff {d} at p={p} s={step}"
            assert am_mine == am_ref, f"argmax mismatch at p={p} s={step}"
            tok_mine, tok_ref, h_mine = am_mine, am_ref, proj
    del model, oracle
    print(f"  equivalence gate PASS: {len(positions)} positions x {chain} "
          f"chained steps, worst |proj diff| = {worst:.2e}")
    return module.cpu()


def tflite_gate(tflite_path, module, cache_len, tol=3e-4):
    """Converted tflite == torch module, with garbage in the stale slot."""
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite_path, num_threads=4)
    sigs = it.get_signature_list()
    assert len(sigs) == 1, f"drafter must have exactly 1 signature: {sigs}"
    key = list(sigs)[0]
    runner = it.get_signature_runner(key)
    inames = set(runner.get_input_details())
    assert inames == {"activations", "input_pos", "mask",
                      "kv_cache_k_mtp", "kv_cache_v_mtp"}, \
        f"input names drifted: {inames}"
    onames = set(runner.get_output_details())
    assert {"logits", "projected_activations"} <= onames, onames

    torch.manual_seed(1)
    kvh, hd, h = module.kv_heads, module.head_dim, module.hidden
    worst_p = worst_l = 0.0
    for p in (5, 77, cache_len - 2):
        act = torch.randn(1, 1, 2 * h) * 0.5
        kc = torch.randn(1, kvh, cache_len, hd)
        vc = torch.randn(1, kvh, hd, cache_len)
        pos = torch.tensor([p], dtype=torch.int32)
        mk = torch.randn(1, 1, 1, cache_len)  # content must not matter
        ref = module(act, pos, mk, kc, vc)
        got = runner(activations=act.numpy(), input_pos=pos.numpy(),
                     mask=mk.numpy(),
                     kv_cache_k_mtp=kc.numpy(), kv_cache_v_mtp=vc.numpy())
        dp = float(torch.from_numpy(got["projected_activations"])
                   .sub(ref["projected_activations"]).abs().max())
        dl = float(torch.from_numpy(got["logits"])
                   .sub(ref["logits"]).abs().max())
        am_t = int(got["logits"][0, 0].argmax())
        am_r = int(ref["logits"][0, 0].argmax())
        worst_p, worst_l = max(worst_p, dp), max(worst_l, dl)
        assert dp < tol and am_t == am_r, \
            f"tflite mismatch at pos {p}: proj {dp}, argmax {am_t} vs {am_r}"
    print(f"  tflite gate PASS: worst proj diff {worst_p:.2e}, "
          f"logits diff {worst_l:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--cache-len", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--quantize", action="store_true",
                    help="wi8fc-quantize the exported tflite in place")
    ap.add_argument("--topk", type=int, default=0,
                    help="top-K head: keep ids [0,K) + the >=247000 special "
                         "block, pad the rest with -30000 in-graph; 0 = full")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = cfg.text_config
    print(f"loading mtp tensors + tied table for {args.model} ...", flush=True)
    mtp, table = load_tensors(args.model)
    module = Qwen35MtpDrafter(text_cfg, mtp, table, args.cache_len,
                              topk=args.topk).eval()
    print(f"drafter: hidden={module.hidden} heads={module.n_heads}/"
          f"{module.kv_heads} hd={module.head_dim} vocab={table.shape[0]}"
          + (f" topk={args.topk} (+{table.shape[0] - SPECIAL_BLOCK_START} "
             f"specials, head rows {module.head.weight.shape[0]})"
             if args.topk else ""))

    if not args.skip_gate:
        print("equivalence gate vs oracle manual_step (real trace) ...",
              flush=True)
        module = equivalence_gate(module, args.model, args.device)

    print("converting ...", flush=True)
    from litert_torch._convert import interface as converter_utils
    conv = converter_utils.Converter()
    sample = {
        "activations": torch.zeros(1, 1, 2 * module.hidden),
        "input_pos": torch.tensor([8], dtype=torch.int32),
        "mask": torch.zeros(1, 1, 1, args.cache_len),
        "kv_cache_k_mtp": torch.zeros(
            1, module.kv_heads, args.cache_len, module.head_dim),
        "kv_cache_v_mtp": torch.zeros(
            1, module.kv_heads, module.head_dim, args.cache_len),
    }
    conv.add_signature("draft", module.cpu().eval(), sample_kwargs=sample)
    lrt = conv.convert(strict_export=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lrt.export(args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")

    print("tflite numeric gate ...", flush=True)
    tflite_gate(args.out, module, args.cache_len)

    if args.quantize:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                        "minicpm5_work"))
        from quantize_minicpm5 import build_recipe
        from ai_edge_quantizer import quantizer
        qt = quantizer.Quantizer(args.out, build_recipe("wi8fc"))
        assert not qt.need_calibration
        res = qt.quantize()
        q_out = args.out.replace(".tflite", "_int8.tflite")
        assert q_out != args.out
        res.export_model(q_out)
        print(f"wi8fc: {os.path.getsize(args.out)/1e6:.0f} -> "
              f"{os.path.getsize(q_out)/1e6:.0f} MB -> {q_out}")

    print("DONE")


if __name__ == "__main__":
    main()

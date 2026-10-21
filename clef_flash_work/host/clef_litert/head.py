"""The checkpoint's joint schema head (joint_schema_model.py `JointSchemaHead.forward`, one record) in NumPy float32.

Weights: joint_head.safetensors (bfloat16, 122 tensors, widened to float32 exactly; `load_state_dict(strict=True)` names)
and joint_head_config.json {hidden_size 4096, width 1024, routing_layers 2, layers 4, heads 16, feedforward 4096}.
Inputs: the row graph's hidden states at the record's T real positions (after the final RMSNorm), the record's ids
(processor form) and spans, and the lm_head table (`lm_head.weight` rows, for the lexical vectors).

    x        = hidden_norm(hidden)                          LayerNorm(4096); memory = memory_projection(x) [T, 1024]
    global   = x[T - 1];  q_i = mean of x over question i's span;  ctx_o = mean of x over option o's span
    lexical_o = mean of the lm_head rows of option o's span ids              (not normalized)
    queries_o = option_context_projection(ctx_o) + option_lexical_projection(lexical_o)
                + option_question_projection(q_i)                               i = the question of option o
    2 x evidence routing: queries += MHA(query_norm(queries), memory_norm(memory)); queries += FFN(feedforward_norm(.))
                          (pre-LN, 16 heads of 64, FFN = Linear 1024->4096, GELU, Linear 4096->1024)
    per question: w = softmax over its options of routed . question_projection(q_i) / 32;  summary_i = sum w * routed
    fields   = question_projection(q) + option_summary_norm(summary) + global_projection(global) + type_embedding(type)
    4 x TransformerDecoderLayer (norm_first, GELU): self-attention over the questions, cross-attention to the memory
              (not normalized), FFN;  fields = field_norm(fields)
    prior_o  = e^min(prior_logit_scale, ln 100) * cos(lexical_o, q_i + global)
    joint_o  = e^min(joint_logit_scale, ln 100) * cos(field_i, option_norm(routed_o)) + residual_scorer([f, o, f*o, |f-o|])
    logit_o  = prior_o + sigmoid(residual_gate) * joint_o

Every tensor is float32 as in the author's float32 module; LayerNorm eps 1e-5; attention = softmax(q k^T / 8) v per
head in float32; GELU = the exact erf form (evaluated in float64, rounded to float32); F.normalize eps 1e-12 and
F.cosine_similarity eps 1e-8 (each side divided by max(norm, eps), then the dot product, as torch computes it).
Dropout is the identity (eval). The per-question softmax of the logits is the response's (response.py)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ._erf import gelu
from .tables import load_bf16_tensors

CONFIG = {"hidden_size": 4096, "width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}
LN_EPS = 1e-5
F32 = np.float32


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = LN_EPS) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True, dtype=F32)
    xc = x - mean
    var = (xc * xc).mean(axis=-1, keepdims=True, dtype=F32)
    return (xc * (F32(1.0) / np.sqrt(var + F32(eps)))) * w + b


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    """x / max(||x||_2, eps) over the last axis (F.normalize)."""
    n = np.sqrt((x * x).sum(axis=-1, keepdims=True, dtype=F32))
    return x / np.maximum(n, F32(eps))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True, dtype=F32)


class _Linear:
    def __init__(self, w: np.ndarray, b: np.ndarray | None = None):
        self.wt = np.ascontiguousarray(w.T, dtype=F32)        # [in, out]
        self.b = None if b is None else np.ascontiguousarray(b, dtype=F32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x @ self.wt
        return y if self.b is None else y + self.b


class _LayerNorm:
    def __init__(self, w: np.ndarray, b: np.ndarray):
        self.w, self.b = w.astype(F32), b.astype(F32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return layer_norm(x, self.w, self.b)


class _Attention:
    """nn.MultiheadAttention (batch_first, eval, no masks) for one sequence: q_in [Lq, E], kv_in [Lk, E] -> [Lq, E]."""

    def __init__(self, t: dict, prefix: str, heads: int):
        w, b = t[prefix + "in_proj_weight"], t[prefix + "in_proj_bias"]
        e = w.shape[1]
        self.q = _Linear(w[:e], b[:e])
        self.k = _Linear(w[e:2 * e], b[e:2 * e])
        self.v = _Linear(w[2 * e:], b[2 * e:])
        self.out = _Linear(t[prefix + "out_proj.weight"], t[prefix + "out_proj.bias"])
        self.heads, self.dim = heads, e // heads
        self.scale = F32(1.0 / math.sqrt(self.dim))

    def __call__(self, q_in: np.ndarray, kv_in: np.ndarray) -> np.ndarray:
        h, d = self.heads, self.dim
        q = self.q(q_in).reshape(q_in.shape[0], h, d).transpose(1, 0, 2)          # [H, Lq, D]
        k = self.k(kv_in).reshape(kv_in.shape[0], h, d).transpose(1, 0, 2)        # [H, Lk, D]
        v = self.v(kv_in).reshape(kv_in.shape[0], h, d).transpose(1, 0, 2)
        p = softmax((q @ k.transpose(0, 2, 1)) * self.scale)                     # [H, Lq, Lk]
        o = (p @ v).transpose(1, 0, 2).reshape(q_in.shape[0], h * d)
        return self.out(o)


class _Tracked(dict):
    """The weights by name, recording which names were read (load_state_dict(strict=True): none missing, none unused)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


class JointHead:
    """The author's JointSchemaHead for one record. `scores(hidden, ids, questions, lm_head)` -> one float32 logit
    array per question, in the author's option order."""

    def __init__(self, tensors: dict[str, np.ndarray], config: dict | None = None):
        cfg = dict(CONFIG if config is None else config)
        if cfg != CONFIG:
            raise ValueError(f"head config {cfg} is not the release's {CONFIG}")
        self.config = cfg
        t = _Tracked({k: np.asarray(v, dtype=F32) for k, v in tensors.items()})
        heads = cfg["heads"]
        ln = lambda p: _LayerNorm(t[p + ".weight"], t[p + ".bias"])                # noqa: E731
        lin = lambda p, bias=True: _Linear(t[p + ".weight"], t[p + ".bias"] if bias else None)  # noqa: E731
        self.hidden_norm = ln("hidden_norm")
        self.memory_projection = lin("memory_projection", False)
        self.question_projection = lin("question_projection", False)
        self.option_question_projection = lin("option_question_projection", False)
        self.global_projection = lin("global_projection", False)
        self.option_context_projection = lin("option_context_projection", False)
        self.option_lexical_projection = lin("option_lexical_projection", False)
        self.type_embedding = t["type_embedding.weight"]
        self.evidence = [{"query_norm": ln(f"evidence_layers.{i}.query_norm"),
                          "memory_norm": ln(f"evidence_layers.{i}.memory_norm"),
                          "attention": _Attention(t, f"evidence_layers.{i}.attention.", heads),
                          "feedforward_norm": ln(f"evidence_layers.{i}.feedforward_norm"),
                          "ff0": lin(f"evidence_layers.{i}.feedforward.0"),
                          "ff3": lin(f"evidence_layers.{i}.feedforward.3")}
                         for i in range(cfg["routing_layers"])]
        self.option_summary_norm = ln("option_summary_norm")
        self.layers = [{"norm1": ln(f"layers.{i}.norm1"), "norm2": ln(f"layers.{i}.norm2"),
                        "norm3": ln(f"layers.{i}.norm3"),
                        "self_attn": _Attention(t, f"layers.{i}.self_attn.", heads),
                        "multihead_attn": _Attention(t, f"layers.{i}.multihead_attn.", heads),
                        "linear1": lin(f"layers.{i}.linear1"), "linear2": lin(f"layers.{i}.linear2")}
                       for i in range(cfg["layers"])]
        self.field_norm = ln("field_norm")
        self.option_norm = ln("option_norm")
        self.scorer0 = lin("residual_scorer.0")
        self.scorer3 = lin("residual_scorer.3")
        cap = F32(math.log(100.0))
        self.prior_scale = np.exp(np.minimum(t["prior_logit_scale"].reshape(()), cap)).astype(F32)
        self.joint_scale = np.exp(np.minimum(t["joint_logit_scale"].reshape(()), cap)).astype(F32)
        g = t["residual_gate"].reshape(())
        self.gate = (F32(1.0) / (F32(1.0) + np.exp(-g))).astype(F32)
        unused = set(t) - t.read                     # a missing name already raised KeyError when it was read
        if unused:
            raise ValueError(f"head weights not used: {sorted(unused)}")
        self.n_tensors = len(t)
        self.n_params = int(sum(v.size for v in dict.values(t)))

    @classmethod
    def from_files(cls, weights: str | Path, config: str | Path | None = None) -> "JointHead":
        cfg = json.loads(Path(config).read_text()) if config is not None else None
        return cls(load_bf16_tensors(weights), cfg)

    @staticmethod
    def lexical_rows(lm_head, ids, questions) -> list[np.ndarray]:
        """Per question, [n_options, 4096]: the mean of the lm_head rows of each option span's ids (float32)."""
        ids = np.asarray(ids, dtype=np.int64)
        out = []
        for q in questions:
            rows = [lm_head.rows(ids[a:b]).mean(axis=0, dtype=F32) for a, b in q.option_spans]
            out.append(np.stack(rows).astype(F32))
        return out

    def scores(self, hidden: np.ndarray, ids, questions, lm_head=None, lexical: list | None = None) -> list[np.ndarray]:
        """hidden [T, 4096] (the record's real positions), ids (len T), questions (request.EncodedQuestion or
        anything with question_span / option_spans / type_id) -> [float32 logits per question].
        `lexical` (one [n_options, 4096] array per question) replaces the lm_head gather (controls)."""
        hidden = np.asarray(hidden, dtype=F32)
        if hidden.ndim != 2 or hidden.shape[0] != len(ids):
            raise ValueError(f"hidden {hidden.shape} for {len(ids)} ids")
        if lexical is None:
            lexical = self.lexical_rows(lm_head, ids, questions)
        x = self.hidden_norm(hidden)                                            # [T, 4096]
        memory = self.memory_projection(x)                                      # [T, 1024]
        global_vector = x[-1]
        question_vectors = np.stack([x[a:b].mean(axis=0, dtype=F32) for a, b in (q.question_span for q in questions)])
        type_rows = self.type_embedding[[q.type_id for q in questions]]
        counts = [len(q.option_spans) for q in questions]
        context = [np.stack([x[a:b].mean(axis=0, dtype=F32) for a, b in q.option_spans]) for q in questions]

        queries = np.concatenate([
            self.option_context_projection(ctx) + self.option_lexical_projection(lex)
            + self.option_question_projection(question_vectors[i])[None]
            for i, (ctx, lex) in enumerate(zip(context, lexical))])             # [O, 1024]
        for layer in self.evidence:
            routed = layer["attention"](layer["query_norm"](queries), layer["memory_norm"](memory))
            queries = queries + routed
            queries = queries + layer["ff3"](gelu(layer["ff0"](layer["feedforward_norm"](queries))))
        split = np.split(queries, np.cumsum(counts)[:-1])

        base = self.question_projection(question_vectors)                       # [Q, 1024]
        summaries = []
        for field, options in zip(base, split):
            w = softmax((options @ field) / F32(math.sqrt(options.shape[-1])), axis=0)
            summaries.append((w[:, None] * options).sum(axis=0, dtype=F32))
        fields = (base + self.option_summary_norm(np.stack(summaries)) + self.global_projection(global_vector)[None]
                  + type_rows)
        for layer in self.layers:
            f1 = layer["norm1"](fields)
            fields = fields + layer["self_attn"](f1, f1)
            fields = fields + layer["multihead_attn"](layer["norm2"](fields), memory)
            fields = fields + layer["linear2"](gelu(layer["linear1"](layer["norm3"](fields))))
        fields = self.field_norm(fields)

        out = []
        for i, (field, lex, routed) in enumerate(zip(fields, lexical, split)):
            anchor = normalize(question_vectors[i] + global_vector, 1e-12)
            prior = self.prior_scale * (normalize(lex, 1e-12) @ anchor)
            options = self.option_norm(routed)
            f = np.broadcast_to(field, options.shape)
            cosine = (normalize(f, 1e-8) * normalize(options, 1e-8)).sum(axis=-1, dtype=F32)
            features = np.concatenate([f, options, f * options, np.abs(f - options)], axis=-1)
            residual = self.scorer3(gelu(self.scorer0(features)))[:, 0]
            joint = self.joint_scale * cosine + residual
            out.append((prior + self.gate * joint).astype(F32))
        return out

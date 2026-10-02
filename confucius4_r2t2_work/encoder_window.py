"""Audio-encoder attention windows for the litert-torch Qwen3-ASR export, made equal to transformers 5.14.1.

transformers 5.14.1 (`Qwen3ASREncoder.forward` + `get_audio_cu_seqlens`) runs the encoder self-attention inside windows
of 13 * (n_window_infer // (2 * n_window)) post-CNN tokens, counted from the start of the clip: 13 tokens per 100-frame
mel chunk, n_window_infer 800 -> 104-token (8 s) windows; the last window holds the remainder. litert-torch 0918
(`model_ext/qwen3/qwen3_asr.py` l.26-54) patches `Qwen3ASRAudioAttention.forward` to attend inside 13-token chunks
(`_CHUNK_LEN = 13`, 1 s), which costs en 1.64 / zh 4.98 / ja 9.01 pt (transcripts against the original model) on
60 five-second FLEURS crops.

This module replaces that patched forward (same signature, process-local, the litert-torch tree is not edited) by one
SDPA over the whole sequence with a constant block-diagonal additive mask: 0 inside a window, MASK_VALUE across
windows. Windows = [0, W), [W, 2W), ... with W = 104 tokens, the last one shorter; a sequence of W tokens or fewer is
one window and gets no mask (5 s = 65 tokens). 30 s = 390 tokens -> 104 + 104 + 104 + 78. The mask is a numpy
constant built at trace time, so the exported graph gets a constant ADD inside the SDPA composite and no new op type.
MASK_VALUE is finite and representable in fp16 (no inf arithmetic on an fp16 GPU path); exp(-1e4 + s) underflows to
exactly 0 in fp32 for any attention score s seen here, so on CPU the masked softmax equals the per-window softmax.

  import encoder_window; encoder_window.install()   # before the export builds the model
"""
import numpy as np
import torch

from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr

TOKENS_PER_CHUNK = 13  # post-CNN length of one 100-frame chunk (three k=3, s=2, p=1 convs: 100 -> 50 -> 25 -> 13)
MASK_VALUE = -1.0e4


def window_tokens(config):
    """HF: window_aftercnn = max_len_after_cnn * (n_window_infer // (n_window * 2)) for full chunks."""
    assert config.n_window == 50, config.n_window
    return TOKENS_PER_CHUNK * (config.n_window_infer // (2 * config.n_window))


def windows(seqlen, window):
    """Window lengths from the start, last one shorter (the same split as HF get_audio_cu_seqlens)."""
    out = [window] * (seqlen // window)
    if seqlen % window:
        out.append(seqlen % window)
    return out


def block_diagonal_mask(seqlen, window):
    """[1, 1, seqlen, seqlen] float32: 0 where query and key share a window, MASK_VALUE elsewhere.

    Built in numpy and wrapped with torch.from_numpy so that torch.export records one constant tensor. A torch.arange
    / where version is traced as graph ops instead (the first 30 s export carried FLOOR_MOD, SIGN, NOT_EQUAL,
    LOGICAL_AND, SELECT, SELECT_V2 and EQUAL computing the mask at run time)."""
    wid = np.arange(seqlen) // window
    mask = np.where(wid[:, None] == wid[None, :], 0.0, MASK_VALUE).astype(np.float32)[None, None]
    return torch.from_numpy(np.ascontiguousarray(mask))


def _audio_attention_forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
    """Replacement for qwen3_asr._audio_attention_forward: whole-sequence SDPA with HF's attention windows."""
    kwargs.pop("cu_seqlens", None)
    seqlen, _ = hidden_states.size()
    window = window_tokens(self.config)
    q = self.q_proj(hidden_states).reshape(1, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).reshape(1, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).reshape(1, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
    mask = None if seqlen <= window else block_diagonal_mask(seqlen, window)
    attn_output, _ = asr_model._sdpa(self, q, k, v, attention_mask=mask, scaling=self.scaling, **kwargs)
    attn_output = attn_output.reshape(seqlen, -1)
    return self.out_proj(attn_output)


def install():
    """Make Qwen3Asr(..., override_transformers=True) bind this forward (it reads the module global at __init__)."""
    assert qwen3_asr._CHUNK_LEN == 13, qwen3_asr._CHUNK_LEN
    qwen3_asr._audio_attention_forward = _audio_attention_forward
    print("PATCH qwen3_asr._audio_attention_forward -> encoder_window._audio_attention_forward "
          "(block-diagonal, HF windows; process-local)", flush=True)

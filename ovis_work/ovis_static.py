"""Static single-fixed-resolution rewrite of Ovis2.5-2B's Siglip2Navit vision tower so it
torch.export()s (the original has .item()/.tolist()/grid-loops/argsort = data-dependent).

KEY: config `fullatt_block_indexes=None` => EVERY layer uses full attention (cu_seqlens=[0,N]),
so the window_index reorder (encoder) is a mathematical NO-OP (full attn is permutation-
equivariant + rotary is reordered with the tokens + reversed at the end). => drop the whole
window machinery; a fixed 512x512 image is N=1024 patches, full attention, with a PRECOMPUTED
position-embedding + rotary (both constant for the fixed grid).

`install_static_vision(model)` monkeypatches embeddings/attention/encoder in place.
`verify(model)` checks static _encode ~= original _encode (corr) on random pixels.
"""
import torch
import torch.nn.functional as F


def _find_vision(model):
    vt = model.visual_tokenizer
    vit = vt.vit                       # Siglip2NavitModel
    vm = vit.vision_model              # Siglip2VisionTransformer
    return vt, vit, vm


@torch.no_grad()
def _precompute_consts(vm, grid_h, grid_w):
    """Run the ORIGINAL pos-embed + rotary code eagerly on the fixed grid [1,grid_h,grid_w]
    (in ORIGINAL token order, i.e. WITHOUT the encoder's window reorder) -> constants."""
    emb, enc = vm.embeddings, vm.encoder
    dev = emb.patch_embedding.weight.device
    dt = emb.patch_embedding.weight.dtype
    grid = torch.tensor([[1, grid_h, grid_w]], device=dev, dtype=torch.long)
    hs = emb.hidden_stride
    H, W = grid_h, grid_w

    # --- position embedding (embeddings.forward lines 167-186, single grid, t=1) ---
    pe_const = None
    if getattr(emb, "preserve_original_pe", False):
        pos = emb.position_embedding.weight.reshape(
            emb.position_embedding_size, emb.position_embedding_size, -1).unsqueeze(0).permute(0, 3, 1, 2)
        pe = F.interpolate(pos.float(), size=(H, W), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, H * W, -1)[0]
        pe = pe.reshape(1, H // hs, hs, W // hs, hs, -1).permute(0, 1, 3, 2, 4, 5).reshape(H * W, -1)
        pe_const = pe.to(dt)

    # --- rotary (encoder.rot_pos_emb + cat + cos/sin, UNREORDERED) ---
    rot = enc.rot_pos_emb(grid)                       # (N, D)
    emb2 = torch.cat((rot, rot), dim=-1)
    cos_const, sin_const = emb2.cos(), emb2.sin()
    return pe_const, cos_const, sin_const


def install_static_vision(model, grid_h=32, grid_w=32):
    vt, vit, vm = _find_vision(model)
    pe_const, cos_const, sin_const = _precompute_consts(vm, grid_h, grid_w)
    emb, enc = vm.embeddings, vm.encoder
    N = grid_h * grid_w

    emb.register_buffer("_pe_const", pe_const, persistent=False) if pe_const is not None else None
    enc.register_buffer("_cos_const", cos_const, persistent=False)
    enc.register_buffer("_sin_const", sin_const, persistent=False)

    # import the model's own rotary-apply (non-flash) from the modeling module
    import sys
    modmod = next(m for name, m in sys.modules.items() if "modeling_ovis2_5" in name and hasattr(m, "apply_rotary_pos_emb_vision"))
    apply_rope = modmod.apply_rotary_pos_emb_vision

    # ---- static embeddings.forward: patch_embed (conv) + precomputed pe ----
    def _emb_fwd(self, pixel_values, grid_thws=None):
        pv = pixel_values.view(-1, self.config.num_channels * self.config.temporal_patch_size,
                               self.patch_size, self.patch_size).to(self.patch_embedding.weight.dtype)
        x = self.patch_embedding(pv).reshape(-1, self.embed_dim)
        if getattr(self, "_pe_const", None) is not None:
            x = x + self._pe_const
        return x
    emb.forward = _emb_fwd.__get__(emb, type(emb))

    # ---- static attention.forward: single FULL SDPA over all N tokens ----
    def _attn_fwd(self, hidden_states, cu_seqlens=None, position_embeddings=None):
        L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(L, self.num_heads, self.head_dim)
        if self.use_rope:
            cos, sin = position_embeddings
            q, k = apply_rope(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q, k = q.squeeze(0), k.squeeze(0)
        q = q.transpose(0, 1).unsqueeze(0)   # (1, heads, L, d)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        o = o.squeeze(0).transpose(0, 1).reshape(L, self.embed_dim)
        return self.out_proj(o)
    for layer in enc.layers:
        layer.self_attn.forward = _attn_fwd.__get__(layer.self_attn, type(layer.self_attn))

    # ---- static encoder.forward: no window/reorder, precomputed rotary, full attn ----
    def _enc_fwd(self, inputs_embeds, grid_thws=None, output_hidden_states=False):
        pos = (self._cos_const, self._sin_const)
        h = inputs_embeds
        states = () if output_hidden_states else None
        for block in self.layers:
            h = block(h, None, pos)
            if output_hidden_states:
                states = states + (h,)
        return h, states
    enc.forward = _enc_fwd.__get__(enc, type(enc))
    return N


@torch.no_grad()
def verify(model, grid_h=32, grid_w=32):
    vt, vit, vm = _find_vision(model)
    dev = vm.embeddings.patch_embedding.weight.device
    dt = vm.embeddings.patch_embedding.weight.dtype
    N = grid_h * grid_w
    C = vm.config.num_channels * vm.config.temporal_patch_size * vm.config.patch_size ** 2
    torch.manual_seed(0)
    pv = torch.randn(N, C, device=dev, dtype=dt)
    grid = torch.tensor([[1, grid_h, grid_w]], device=dev, dtype=torch.long)

    orig = vt._encode(pv, grid).float()
    install_static_vision(model, grid_h, grid_w)
    stat = vt._encode(pv, grid).float()

    corr = torch.corrcoef(torch.stack([orig.flatten(), stat.flatten()]))[0, 1].item()
    maxdiff = (orig - stat).abs().max().item()
    print(f"features shape {tuple(orig.shape)} | corr {corr:.8f} | maxdiff {maxdiff:.2e}")
    return corr

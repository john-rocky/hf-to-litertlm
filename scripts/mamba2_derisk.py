"""Minimal standalone Mamba2 decode-step → tflite, to de-risk Metal GPU delegation.

Faithful to granite-4.0-h-350m mamba dims. Random weights — we test OP-STRUCTURE
delegation (does the selective-scan recurrence map to Metal?), not numerics.
Single-token decode form: no sequential chunked scan, fixed shapes, export-friendly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch

# granite-4.0-h-350m mamba dims
D_MODEL = 768
EXPAND = 2
D_INNER = D_MODEL * EXPAND       # 1536
N_HEADS = 48
D_HEAD = 32                      # 48*32 = 1536 = D_INNER
D_STATE = 128
D_CONV = 4
N_GROUPS = 1
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE   # 1792
D_IN_PROJ = 2 * D_INNER + 2 * N_GROUPS * D_STATE + N_HEADS  # 3376


class Mamba2DecodeStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, D_IN_PROJ, bias=False)
        self.conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, kernel_size=D_CONV,
                                groups=CONV_DIM, bias=True)
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)
        self.A_log = nn.Parameter(torch.randn(N_HEADS))
        self.D = nn.Parameter(torch.randn(N_HEADS))
        self.dt_bias = nn.Parameter(torch.randn(N_HEADS))
        self.norm_w = nn.Parameter(torch.ones(D_INNER))

    def forward(self, hidden, conv_state, ssm_state):
        # hidden [B,D_MODEL]; conv_state [B,CONV_DIM,D_CONV-1]; ssm_state [B,N_HEADS,D_HEAD,D_STATE]
        zxbcdt = self.in_proj(hidden)                       # [B, D_IN_PROJ]
        z, xBC, dt = torch.split(zxbcdt, [D_INNER, CONV_DIM, N_HEADS], dim=-1)

        # causal depthwise conv via state ring-buffer (no dynamic shapes)
        conv_in = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)   # [B,CONV_DIM,D_CONV]
        new_conv_state = conv_in[:, :, 1:]                            # [B,CONV_DIM,D_CONV-1]
        xBC = self.conv1d(conv_in).squeeze(-1)                        # [B,CONV_DIM]
        xBC = F.silu(xBC)
        x, B, C = torch.split(xBC, [D_INNER, D_STATE, D_STATE], dim=-1)

        # SSD selective-state recurrence (single step)
        A = -torch.exp(self.A_log)                                   # [N_HEADS]
        dt = F.softplus(dt + self.dt_bias)                           # [B,N_HEADS]
        dA = torch.exp(dt * A)                                       # [B,N_HEADS]
        x = x.view(-1, N_HEADS, D_HEAD)                              # [B,N_HEADS,D_HEAD]
        dBx = (dt[:, :, None, None]
               * B[:, None, None, :]
               * x[:, :, :, None])                                   # [B,N_HEADS,D_HEAD,D_STATE]
        new_ssm = ssm_state * dA[:, :, None, None] + dBx
        y = (new_ssm * C[:, None, None, :]).sum(-1)                  # [B,N_HEADS,D_HEAD]
        y = y + self.D[None, :, None] * x
        y = y.reshape(-1, D_INNER)                                   # [B,D_INNER]
        y = y * self.norm_w * F.silu(z)
        out = self.out_proj(y)                                       # [B,D_MODEL]
        return out, new_conv_state, new_ssm


def main():
    import sys
    out_path = sys.argv[1] if len(sys.argv) > 1 else "mamba2_step.tflite"
    m = Mamba2DecodeStep().eval()
    args = (
        torch.randn(1, D_MODEL),
        torch.randn(1, CONV_DIM, D_CONV - 1),
        torch.randn(1, N_HEADS, D_HEAD, D_STATE),
    )
    with torch.no_grad():
        ref = m(*args)
    print("eager OK; out shapes:", [tuple(t.shape) for t in ref])
    edge = litert_torch.convert(m, args)
    edge.export(out_path)
    print("EXPORTED:", out_path)


if __name__ == "__main__":
    main()

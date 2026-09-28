"""Unit test of scripts/mrope_derived.py against transformers 5.17.0 (the oracle's own position code).

(1) positions: for every g256_mrope image forward of the round-1 oracle and the v2 additions, HF 5.17.0
    Qwen3_5Model.get_rope_index on the stored input_ids vs derived_thw(arange(S)): exact integer equality at every
    position; plus the oracle's recorded rotary-input probes (positions.probes) vs the derivation.
(2) the same over one synthetic row that fills the whole cache (vision_start + 64 image tokens + text to 4096).
(3) cos/sin: HF 5.17.0 Qwen3_5TextRotaryEmbedding on HF's 3-channel positions vs derived_cos_sin on the 1-D
    positions: bit equality (torch.equal) and max |diff| (acceptance: bit-equal or <= 1e-7).
(4) channel masks vs recomposition_frequencies applied to a channel-id tensor.
(5) no-image rows start at p = 65 in the readout: q.k rotary scores at (m + 9, n + 9) vs (m, n) measured with HF's
    apply_rotary_pos_emb on random q, k (not a gate; the graph readout is the test) -- recorded for the report.

    out/venv-oracle/bin/python -B scripts/test_mrope_derived.py [--step-form relu_diff|clamp] [--out results/x.json]

Round 2 ran it with the clamp step (results/mrope_derived_test.json); round 4 runs the relu_diff step
(--out results/mrope_derived_test_r4.json) and adds (6): derived_thw under both step forms over 0..4095 (bit equality)
and each form's step function on integer-valued float32 inputs. The HF reference arrays under out/ are written once and
afterwards only compared (never overwritten).
"""
import argparse
import types

import numpy as np
import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from common import ROOT, SNAPSHOT, read_json, write_json
import mrope_derived as md

IMAGE_ID, VSTART_ID, VEND_ID = 248056, 248053, 248054


def hf_positions(input_ids, grid=(1, 16, 16)):
    """HF 5.17.0 get_rope_index on one row (unbound call with the only config field it reads)."""
    fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    fake.get_vision_position_ids = types.MethodType(modeling.Qwen3_5Model.get_vision_position_ids, fake)
    ids = torch.tensor([input_ids])
    mm = (ids == IMAGE_ID).to(torch.int32)
    pos, _ = modeling.Qwen3_5Model.get_rope_index(fake, ids, mm, image_grid_thw=torch.tensor([list(grid)]))
    return pos                                                            # [3, 1, S] int64


def save_or_compare(path, arr):
    """Write a reference array once; later runs compare against the stored file instead of overwriting it."""
    if path.exists():
        return dict(path=str(path.relative_to(ROOT)), existed=True, bit_equal=bool(np.array_equal(np.load(path), arr)))
    np.save(path, arr)
    return dict(path=str(path.relative_to(ROOT)), existed=False, written=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step-form', default=None)
    ap.add_argument('--out', default='results/mrope_derived_test.json')
    args = ap.parse_args()
    if args.step_form:
        md.set_step_form(args.step_form)
    cfg = AutoConfig.from_pretrained(str(SNAPSHOT))
    tcfg = cfg.text_config
    rot = modeling.Qwen3_5TextRotaryEmbedding(config=tcfg).eval()
    section = rot.mrope_section
    inv = rot.inv_freq
    res = dict(transformers=transformers.__version__, torch=torch.__version__, mrope_section=list(section),
               n_freq=int(inv.shape[0]), constants=dict(N_IMG=md.N_IMG, GH=md.GH, GW=md.GW, OFFSET=md.OFFSET),
               attention_scaling=float(rot.attention_scaling), step_form=md.STEP_FORM)

    # (4) masks vs HF recomposition on a channel-id tensor
    ch = torch.stack([torch.full((1, 1, inv.shape[0]), float(c)) for c in range(3)])   # [3, B, S, F]
    hf_owner = rot.recomposition_frequencies(ch.clone())[0, 0, :inv.shape[0]].long().tolist()
    _, owner = md.channel_masks(inv.shape[0], section)
    res['masks'] = dict(owner=owner, hf_owner=hf_owner, equal=owner == hf_owner,
                        h_idx=[j for j, c in enumerate(owner) if c == 1], w_idx=[j for j, c in enumerate(owner) if c == 2])
    assert owner == hf_owner, (owner, hf_owner)
    assert res['masks']['h_idx'] == list(range(1, 32, 3)) and res['masks']['w_idx'] == list(range(2, 30, 3))

    # (1) fixture rows
    rows = []
    for path in ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json'):
        orc = read_json(path)
        for f in orc['forwards']:
            if f['arm'] != 'g256_mrope' or not f['n_image_tokens']:
                continue
            ids = f['input_ids']
            S = len(ids)
            assert ids[0] == VSTART_ID and ids[1:65] == [IMAGE_ID] * 64 and ids[65] == VEND_ID, f['row_id']
            hf = hf_positions(ids)                                        # [3,1,S]
            p = torch.arange(S, dtype=torch.float32)[None, :]
            t, h, w = md.derived_thw(p)
            der = torch.stack([t, h, w]).round().long()                   # [3,1,S]
            eq_pos = bool(torch.equal(der, hf))
            probes_ok = all([int(v) for v in der[:, 0, pr['index']]] == pr['thw'] for pr in f['positions']['probes'].values())
            cos_h, sin_h = rot(torch.zeros(1, dtype=torch.float32), hf)
            cos_d, sin_d = md.derived_cos_sin(p.long(), inv, section, rot.attention_scaling)
            rows.append(dict(source=path, row_id=f['row_id'], n_tokens=S, positions_equal=eq_pos, probes_equal=probes_ok,
                             n_probes=len(f['positions']['probes']),
                             cos_bit_equal=bool(torch.equal(cos_h, cos_d)), sin_bit_equal=bool(torch.equal(sin_h, sin_d)),
                             cos_max_abs=float((cos_h - cos_d).abs().max()), sin_max_abs=float((sin_h - sin_d).abs().max())))
    res['fixture_rows'] = rows
    res['fixture_summary'] = dict(
        n_rows=len(rows), positions_equal=sum(r['positions_equal'] for r in rows), probes_equal=sum(r['probes_equal'] for r in rows),
        cos_sin_bit_equal=sum(r['cos_bit_equal'] and r['sin_bit_equal'] for r in rows),
        cos_sin_max_abs=max(max(r['cos_max_abs'], r['sin_max_abs']) for r in rows))

    # (2) one synthetic row over the whole cache length
    L = 4096
    ids = [VSTART_ID] + [IMAGE_ID] * 64 + [VEND_ID] + [13] * (L - 66)
    hf = hf_positions(ids)
    p = torch.arange(L, dtype=torch.float32)[None, :]
    t, h, w = md.derived_thw(p)
    der = torch.stack([t, h, w]).round().long()
    cos_h, sin_h = rot(torch.zeros(1, dtype=torch.float32), hf)
    cos_d, sin_d = md.derived_cos_sin(p.long(), inv, section, rot.attention_scaling)
    res['full_cache'] = dict(length=L, positions_equal=bool(torch.equal(der, hf)),
                             derived_is_integral=bool(torch.equal(torch.stack([t, h, w]), torch.stack([t, h, w]).round())),
                             cos_bit_equal=bool(torch.equal(cos_h, cos_d)), sin_bit_equal=bool(torch.equal(sin_h, sin_d)),
                             cos_max_abs=float((cos_h - cos_d).abs().max()), sin_max_abs=float((sin_h - sin_d).abs().max()),
                             hf_last_position=[int(v) for v in hf[:, 0, -1]])
    res['reference_arrays'] = [save_or_compare(ROOT / 'out/mrope_ref_5170_cos4096.npy', cos_h.numpy()),
                               save_or_compare(ROOT / 'out/mrope_ref_5170_sin4096.npy', sin_h.numpy()),
                               save_or_compare(ROOT / 'out/mrope_ref_5170_inv_freq.npy', inv.numpy())]
    assert all(r.get('bit_equal', True) for r in res['reference_arrays']), res['reference_arrays']

    # (6) the two step forms: equal on integer-valued float32 input, and derived_thw bit-equal under both
    xs = torch.arange(-4200, 4200, dtype=torch.float32)
    steps = {name: fn(xs) for name, fn in md.STEP_FORMS.items()}
    want = (xs >= 1).to(torch.float32)
    thw = {}
    keep = md.STEP_FORM
    for name in md.STEP_FORMS:
        md.set_step_form(name)
        thw[name] = torch.stack(md.derived_thw(torch.arange(L, dtype=torch.float32)[None, :]))
    md.set_step_form(keep)
    res['step_forms'] = dict(
        x_range=[-4200, 4199],
        step_equals_indicator={name: bool(torch.equal(v, want)) for name, v in steps.items()},
        forms_bit_equal=bool(torch.equal(steps['clamp'], steps['relu_diff'])),
        derived_thw_bit_equal_over_cache=bool(torch.equal(thw['clamp'], thw['relu_diff'])))

    # (5) relative-position invariance of the +9 shift used by the no-image rows (65 -> 9 on every channel)
    torch.manual_seed(0)
    S = 200
    q = torch.randn(1, 8, S, 256)
    k = torch.randn(1, 8, S, 256)
    pos0 = torch.arange(S)[None, None, :].expand(3, 1, S)
    c0, s0 = rot(q, pos0)
    c9, s9 = rot(q, pos0 + 9)
    q0, k0 = modeling.apply_rotary_pos_emb(q, k, c0, s0)
    q9, k9 = modeling.apply_rotary_pos_emb(q, k, c9, s9)
    a0 = (q0 @ k0.transpose(-1, -2)) / 16.0
    a9 = (q9 @ k9.transpose(-1, -2)) / 16.0
    res['shift9_scores'] = dict(S=S, max_abs_score_diff=float((a0 - a9).abs().max()), score_absmax=float(a0.abs().max()),
                                note='random q,k; scaled by 1/sqrt(256); informational')
    res['pass'] = (all(res['step_forms']['step_equals_indicator'].values()) and res['step_forms']['derived_thw_bit_equal_over_cache']
                   and res['fixture_summary']['positions_equal'] == len(rows) and res['fixture_summary']['probes_equal'] == len(rows)
                   and res['full_cache']['positions_equal']
                   and (res['fixture_summary']['cos_sin_bit_equal'] == len(rows) or res['fixture_summary']['cos_sin_max_abs'] <= 1e-7)
                   and (res['full_cache']['cos_bit_equal'] and res['full_cache']['sin_bit_equal']
                        or max(res['full_cache']['cos_max_abs'], res['full_cache']['sin_max_abs']) <= 1e-7))
    write_json(args.out, res)
    print({k: res[k] for k in ('step_form', 'fixture_summary', 'full_cache', 'step_forms', 'reference_arrays', 'shift9_scores', 'pass')})
    assert res['pass']


if __name__ == '__main__':
    main()

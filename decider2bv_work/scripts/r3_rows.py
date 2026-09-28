"""Round 3 runtime rows, fixed before any runtime result was read (2026-09-29 00:10 JST).

Rule: public, image, single-question (nq = 1) rows of fixtures v1 + v2; at least 4 game frames, at least 3 colour
rows and at least 4 borderline rows. Among the eligible rows the low-margin ones are preferred (they are the ones a
runtime difference could flip): the five synthetic borderline rows whose top-1 is below 0.9 in the oracle's
g256_mrope arm (the arm the graph reproduces) and both v2 colour rows below 0.9, plus four game rows and color_red.
Wording corrected 00:23 JST after the runtime rows had run, selection unchanged: this line first said "author top-1";
the selection was made on the g256_mrope values, and under the author arm a sixth synthetic row (synth_bars_close,
author 0.665, g256_mrope 0.962) would also qualify and is not in the set.
The two no-image rows are the only single-question no-image rows (informational, CPU only).
"""
IMAGE_ROWS = [
    'game_pong_atari_up', 'game_pong_atari_level', 'game_breakout_atari_noball', 'game_breakout_atari_center',
    'color_red', 'v2_color_purple', 'v2_color_split_red_blue',
    'synth_unanswerable', 'synth_teal', 'synth_center_circle', 'synth_count_dots', 'synth_half_half',
]
TEXT_ROWS = ['text_finance', 'text_umbrella']
VSTART_ID, VEND_ID, IMAGE_ID, N_IMG = 248053, 248054, 248056, 64
IMG_RENDER = '<|vision_start|><image_soft_token><|vision_end|>'
ORACLES = ('fixtures/oracle_fp32.json', 'fixtures/oracle_fp32_v2add.json')


def oracle_rows(read_json):
    """row_id -> (fixture row, oracle g256_mrope forward) for IMAGE_ROWS + TEXT_ROWS, with the runtime text."""
    fx = {r['id']: r for r in read_json('fixtures/fixtures_v2.json')['rows']}
    fwd = {}
    for p in ORACLES:
        for f in read_json(p)['forwards']:
            if f['arm'] == 'g256_mrope':
                fwd[f['row_id']] = f
    out = {}
    for rid in IMAGE_ROWS + TEXT_ROWS:
        r, f = fx[rid], fwd[rid]
        ids = f['input_ids']
        image = r['image'] is not None
        assert r['tier'] == 'public' and len(r['questions']) == 1 and f['nq'] == 1 and len(f['slots']) == 1, rid
        assert f['slot_idx'] == [len(ids) - 1], rid                # the answer slot is the last prompt token
        if image:
            assert ids[0] == VSTART_ID and ids[1:1 + N_IMG] == [IMAGE_ID] * N_IMG and ids[1 + N_IMG] == VEND_ID, rid
            tail = ids[2 + N_IMG:]
        else:
            assert VEND_ID not in ids and IMAGE_ID not in ids, rid
            tail = ids
        out[rid] = dict(fixture=r, forward=f, image=image, input_ids=ids, text_ids=tail)
    return out

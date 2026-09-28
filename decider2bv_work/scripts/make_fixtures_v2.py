"""Freeze the round-2 fixture set (v2) before any round-2 inference: v1 rows verbatim + appended borderline rows.

v1 (`fixtures/fixtures.json`, committed in round 1) is read, never written. The v2 file carries every v1 row
byte-for-byte as the same JSON object, then the rows added here (all public tier, all image rows, purpose declared
per row before any inference). The additions target the round-1 adequacy shortfall (image slots whose reference
top-1 < 0.9 were 13/48, target >= 1/3): game frames at the author's two game sizes (256x240 NES, 160x210 Atari),
multi-question rows, and 224x224 colour cases (the author's benchmark size). The set is added once; whether the
target is reached is measured afterwards, not tuned.

The GxG=256 inputs are written here with the same call the oracle uses (PIL BICUBIC on the decoded RGB image), so
the vision calibration can read them before the oracle runs; the oracle re-derives each one and asserts the pixels
are equal.

    out/venv-oracle/bin/python -B scripts/make_fixtures_v2.py        (from decider2bv_work/)
"""
import json

import PIL
from PIL import Image, ImageDraw

from common import ROOT, REVISION, write_json, sha256_file, sha256_rgb
import make_fixtures as v1gen                       # drawing helpers only; v1gen.main() is never called

V1_PATH = ROOT / 'fixtures/fixtures.json'
V2_PATH = ROOT / 'fixtures/fixtures_v2.json'
G = 256

VIS_INTRO = v1gen.VIS_INTRO
PONG_CTX = f'{v1gen.PONG_INTRO} {VIS_INTRO}'
BRK_CTX = f'{v1gen.BREAKOUT_INTRO} {VIS_INTRO}'
Q_ACT = 'What should you do right now?'
MARIO_CTX = v1gen.MARIO_CTX
MARIO_QS = v1gen.MARIO_QS
COLOR_CTX = v1gen.COLOR_CTX
CAULDRON_CTX = v1gen.CAULDRON_CTX

added = []


def add(row_id, family, purpose, image, context, questions, note):
    """Same row schema as v1 (make_fixtures.add), public tier only, plus added_in='v2'."""
    assert 'Answer' not in context, row_id
    qs = []
    for q in questions:
        text, options = q[0], q[1]
        expected = q[2] if len(q) > 2 else None
        assert 2 <= len(options) <= 10, row_id
        assert 'Answer' not in text and not any('Answer' in o for o in options), row_id
        qs.append(dict(text=text, options=list(options), expected=expected))
    added.append(dict(id=row_id, family=family, tier='public', purpose=purpose, image=image,
                      context=context, questions=qs, note=note, added_in='v2'))


def save(name, image):
    info = v1gen.save_public(name, image)
    # the GxG input, written exactly as oracle.py derives it (open -> RGB -> PIL BICUBIC GxG -> PNG)
    src = Image.open(ROOT / info['path']).convert('RGB')
    resized = src.resize((G, G), Image.BICUBIC)
    path = ROOT / f'out/fixtures_resized/g{G}/{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert sha256_rgb(Image.open(path)) == sha256_rgb(resized), f'{path} exists with different pixels'
    else:
        resized.save(path, format='PNG')
    assert sha256_rgb(Image.open(path)) == sha256_rgb(resized)
    return info


def main():
    v1 = json.loads(V1_PATH.read_text())
    v1_ids = [r['id'] for r in v1['rows']]

    # ---------------------------------------------------------------- 256x240 NES game frames
    # Pong ball level with the right paddle's centre line (paddle top 112 -> centre 120; ball rows 118-121),
    # ball near the vertical centre line; three questions whose answers hinge on a few pixels.
    add('v2_game_pong_nes_level_multi', 'game', 'borderline',
        save('v2_game_pong_nes_level_multi', v1gen.pong(256, 240, (82, 118), 112, 60)),
        PONG_CTX, [(Q_ACT, v1gen.PONG_OPTS, 2),
                   ('Is the ball above or below your paddle?', ['above', 'below', 'level with it'], 2),
                   ('Which half of the screen is the ball in?', ['left half', 'right half'], None)],
        note='256x240; ball level with the right paddle centre, 2 px right of the centre line; three slots')
    add('v2_game_pong_nes_center_multi', 'game', 'borderline',
        save('v2_game_pong_nes_center_multi', v1gen.pong(256, 240, (79, 108), 102, 102)),
        PONG_CTX, [('Which paddle is closer to the ball?', ['the left paddle', 'the right paddle'], None),
                   ('Is the ball moving toward your paddle?', ['no', 'yes'], None)],
        note='256x240; ball at the exact centre, both paddles level with it; two unanswerable-by-design slots')
    add('v2_game_breakout_nes_edge_multi', 'game', 'borderline',
        save('v2_game_breakout_nes_edge_multi',
             v1gen.breakout(256, 240, (96, 150), 82, missing={(0, 5), (0, 6), (1, 6), (2, 11), (4, 2)})),
        BRK_CTX, [(Q_ACT, v1gen.BREAKOUT_OPTS, None),
                  ('Is the ball left or right of the paddle?', ['left of it', 'right of it', 'directly above it'], None),
                  ('Is the ball moving up or down?', ['up', 'down'], None)],
        note='256x240; ball above the paddle right edge; three slots')
    add('v2_game_pong_atari_near_multi', 'game', 'borderline',
        save('v2_game_pong_atari_near_multi', v1gen.pong(160, 210, (132, 121), 112, 150)),
        PONG_CTX, [(Q_ACT, v1gen.PONG_OPTS, None),
                   ('Is the ball above or below your paddle?', ['above', 'below', 'level with it'], None)],
        note='160x210; ball next to the right paddle, 3 px below its centre line (teacher threshold 6); two slots')
    add('v2_game_pong_atari_noball', 'game', 'borderline',
        save('v2_game_pong_atari_noball', v1gen.pong(160, 210, None, 90, 120)),
        PONG_CTX, [(Q_ACT, v1gen.PONG_OPTS, 2), ('Is the ball visible?', ['no', 'yes'], 0)],
        note='160x210; no ball on screen (between points); two slots')
    add('v2_game_breakout_atari_fewbricks', 'game', 'borderline',
        save('v2_game_breakout_atari_fewbricks',
             v1gen.breakout(160, 210, (60, 120), 70, missing={(r, b) for r in range(4) for b in range(18)})),
        BRK_CTX, [('How many rows of bricks are left?', ['1', '2', '3', '4', '5', '6'], 1),
                  (Q_ACT, v1gen.BREAKOUT_OPTS, None)],
        note='160x210; only the bottom two brick rows remain; two slots')
    add('v2_game_breakout_atari_high', 'game', 'borderline',
        save('v2_game_breakout_atari_high', v1gen.breakout(160, 210, (100, 96), 40, missing={(5, 11), (5, 12), (4, 12)})),
        BRK_CTX, [(Q_ACT, v1gen.BREAKOUT_OPTS, None), ('Is the ball moving up or down?', ['up', 'down'], None)],
        note='160x210; ball just under the bricks, far right of the paddle; two slots')

    # ---------------------------------------------------------------- 224x224 colour cases (decider/bench/mps.py shape)
    for name, rgb in (('orange', (255, 140, 0)), ('purple', (128, 0, 128)), ('cyan', (0, 200, 200))):
        add(f'v2_color_{name}', 'color', 'borderline', save(f'v2_color_{name}', Image.new('RGB', (224, 224), rgb)),
            COLOR_CTX, [('What color is shown?', ['red', 'green', 'blue'], None)],
            note=f'224x224 solid {rgb}, none of the three options exact')
    im = Image.new('RGB', (224, 224), (255, 0, 0))
    ImageDraw.Draw(im).rectangle([112, 0, 223, 223], fill=(0, 0, 255))
    add('v2_color_split_red_blue', 'color', 'borderline', save('v2_color_split_red_blue', im),
        COLOR_CTX, [('What color is shown?', ['red', 'green', 'blue'], None)],
        note='224x224, left half red, right half blue')

    ids = v1_ids + [r['id'] for r in added]
    assert len(ids) == len(set(ids)), 'duplicate row id'
    rows = list(v1['rows']) + added
    # the v1 rows are carried as the same JSON objects (checked on the serialized form, per row)
    for a, b in zip(rows[:len(v1_ids)], v1['rows']):
        assert json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)
    summary = {}
    for r in added:
        key = f"{r['family']}/{r['tier']}/{r['purpose']}"
        summary[key] = summary.get(key, 0) + 1
    out = dict(schema='decider2bv-fixtures-r2', snapshot_revision=REVISION, pillow=PIL.__version__,
               generator='scripts/make_fixtures_v2.py',
               v1=dict(path='fixtures/fixtures.json', sha256=sha256_file(V1_PATH), n_rows=len(v1_ids)),
               n_rows=len(rows), n_rows_added=len(added),
               n_image_rows=sum(r['image'] is not None for r in rows),
               n_slots=sum(len(r['questions']) for r in rows),
               n_image_slots=sum(len(r['questions']) for r in rows if r['image'] is not None),
               n_slots_added=sum(len(r['questions']) for r in added),
               summary_added=summary, added_ids=[r['id'] for r in added], rows=rows)
    write_json('fixtures/fixtures_v2.json', out)
    print(out['n_rows'], 'rows', out['n_rows_added'], 'added', out['n_slots_added'], 'added slots,',
          out['n_image_slots'], 'image slots in v1+v2')
    print(summary)


if __name__ == '__main__':
    main()

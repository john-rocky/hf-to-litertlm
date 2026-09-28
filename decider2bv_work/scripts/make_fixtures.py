"""Freeze the round-1 fixture set before any inference.

Public copy: the six internal photo/document rows (with their image source) and the two platformer-style frames are
removed; every other row is unchanged.

Deterministic: every image is drawn here with PIL (no randomness, no external data). PNGs are git-ignored, so
fixtures.json records each image's file sha256 and decoded-RGB sha256; a rerun must reproduce
both (the script refuses to overwrite a PNG whose pixels differ).

    out/venv-oracle/bin/python -B scripts/make_fixtures.py        (from decider2bv_work/)
"""
import math

import PIL
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, REVISION, write_json, sha256_file, sha256_rgb

IMG_DIR = ROOT / 'fixtures/images'

# Author's own prompt shapes (see rows below for the source of each).
COLOR_CTX = 'Identify the dominant color in the image.'                     # decider/bench/mps.py
VIS_INTRO = 'The image shows the current game screen.'                      # decider/games/frames_data.py
PONG_INTRO = ('You play Pong (Atari) and control the right paddle. Move the paddle so the ball hits it; '
              'the ball bounces off paddles and walls. Missing the ball loses a point.')   # decider/games/envs.py
PONG_OPTS = ['move paddle up', 'move paddle down', 'stay']
BREAKOUT_INTRO = ('You play Breakout (Atari): move the paddle at the bottom so the ball bounces up into the bricks. '
                  'Missing the ball loses a life.')                         # decider/games/envs.py
BREAKOUT_OPTS = ['move paddle left', 'move paddle right', 'stay', 'launch ball']
MARIO_CTX = 'The image shows the current Super Mario Bros screen. Mario runs right; jump over enemies and gaps.'  # decider/vision.py __main__
MARIO_QS = [('What should Mario do right now?', ['run right', 'jump right', 'step left', 'wait']),
            ('Is an enemy visible?', ['no', 'yes'])]
CAULDRON_CTX = 'This is a visual question about the image.'                 # decider/vision/data.py

rows = []


def add(row_id, family, tier, purpose, image, context, questions, note=''):
    assert 'Answer' not in context, row_id
    qs = []
    for q in questions:
        text, options = q[0], q[1]
        expected = q[2] if len(q) > 2 else None
        assert 2 <= len(options) <= 10, row_id
        assert 'Answer' not in text and not any('Answer' in o for o in options), row_id
        qs.append(dict(text=text, options=list(options), expected=expected))
    rows.append(dict(id=row_id, family=family, tier=tier, purpose=purpose, image=image,
                     context=context, questions=qs, note=note))


def save_public(name, image):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / f'{name}_{image.width}x{image.height}.png'
    rgb = sha256_rgb(image)
    if path.exists():
        assert sha256_rgb(Image.open(path)) == rgb, f'{path} exists with different pixels (non-deterministic drawing?)'
    else:
        image.save(path, format='PNG')
    assert sha256_rgb(Image.open(path)) == rgb
    return dict(path=str(path.relative_to(ROOT)), width=image.width, height=image.height,
                sha256_file=sha256_file(path), sha256_rgb=rgb, source='drawn by scripts/make_fixtures.py')


def font(size):
    return ImageFont.load_default(size=size)


# ------------------------------------------------------------------ game frames
def pong(W, H, ball, right_y, left_y):
    """Atari-Pong look drawn in a 160x210 pixel frame, scaled to WxH. ball=(x,y) or None; *_y = paddle top."""
    sx, sy = W / 160, H / 210
    im = Image.new('RGB', (W, H), (144, 72, 17))
    d = ImageDraw.Draw(im)
    R = lambda x0, y0, x1, y1, c: d.rectangle([round(x0 * sx), round(y0 * sy), round(x1 * sx) - 1, round(y1 * sy) - 1], fill=c)
    R(0, 24, 160, 34, (236, 236, 236))
    R(0, 194, 160, 210, (236, 236, 236))
    R(16, left_y, 20, left_y + 16, (213, 130, 74))
    R(140, right_y, 144, right_y + 16, (92, 186, 92))
    if ball is not None:
        R(ball[0], ball[1], ball[0] + 2, ball[1] + 4, (236, 236, 236))
    return im


BRICK_COLORS = [(200, 72, 72), (198, 108, 58), (180, 122, 48), (162, 162, 42), (72, 160, 72), (66, 72, 200)]


def breakout(W, H, ball, paddle_x, missing=()):
    """Atari-Breakout look in a 160x210 frame scaled to WxH. ball=(x,y) or None; paddle_x = paddle left edge."""
    sx, sy = W / 160, H / 210
    im = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    R = lambda x0, y0, x1, y1, c: d.rectangle([round(x0 * sx), round(y0 * sy), round(x1 * sx) - 1, round(y1 * sy) - 1], fill=c)
    R(0, 17, 160, 32, (142, 142, 142))
    R(0, 17, 8, 196, (142, 142, 142))
    R(152, 17, 160, 196, (142, 142, 142))
    for r, c in enumerate(BRICK_COLORS):
        for b in range(18):
            if (r, b) not in missing:
                R(8 + b * 8, 57 + r * 6, 8 + b * 8 + 8, 57 + r * 6 + 6, c)
    R(paddle_x, 189, paddle_x + 16, 193, (200, 72, 72))
    if ball is not None:
        R(ball[0], ball[1], ball[0] + 2, ball[1] + 4, (200, 72, 72))
    return im


# ------------------------------------------------------------------ synthetic scenes
def canvas(W, H, color=(255, 255, 255)):
    im = Image.new('RGB', (W, H), color)
    return im, ImageDraw.Draw(im)


def arrow(W, H, angle_deg, length, width=24, color=(0, 0, 0)):
    """Arrow through the image centre pointing at angle_deg (0 = right, 90 = up)."""
    im, d = canvas(W, H)
    cx, cy = W / 2, H / 2
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), -math.sin(a)
    px, py = -uy, ux
    tail = (cx - ux * length / 2, cy - uy * length / 2)
    neck = (cx + ux * (length / 2 - 2.2 * width), cy + uy * (length / 2 - 2.2 * width))
    tip = (cx + ux * length / 2, cy + uy * length / 2)
    w = width / 2
    d.polygon([(tail[0] + px * w, tail[1] + py * w), (neck[0] + px * w, neck[1] + py * w),
               (neck[0] + px * 2.5 * w, neck[1] + py * 2.5 * w), tip,
               (neck[0] - px * 2.5 * w, neck[1] - py * 2.5 * w), (neck[0] - px * w, neck[1] - py * w),
               (tail[0] - px * w, tail[1] - py * w)], fill=color)
    return im


def bars(W, H, heights, colors, gap=40):
    im, d = canvas(W, H)
    n = len(heights)
    bw = (W - gap * (n + 1)) / n
    base = H - 40
    d.line([gap / 2, base, W - gap / 2, base], fill=(0, 0, 0), width=3)
    for i, (h, c) in enumerate(zip(heights, colors)):
        x0 = gap + i * (bw + gap)
        d.rectangle([round(x0), round(base - h), round(x0 + bw), base - 2], fill=c)
    return im


def main():
    # 1) author's own MPS benchmark cases (decider/bench/mps.py): 224x224 solid colours
    for name, rgb, gold in (('red', (255, 0, 0), 0), ('green', (0, 128, 0), 1), ('blue', (0, 0, 255), 2)):
        # PIL.Image.new('RGB', (224, 224), 'green') is (0,128,0); red/blue are the pure primaries.
        assert Image.new('RGB', (1, 1), name).getpixel((0, 0)) == rgb
        add(f'color_{name}', 'color', 'public', 'control', save_public(f'color_{name}', Image.new('RGB', (224, 224), name)),
            COLOR_CTX, [('What color is shown?', ['red', 'green', 'blue'], gold)],
            note='decider/bench/mps.py case, verbatim')

    # 2) game frames, author's training prompt (intro + VIS_INTRO, frames_data.py) and the vision.py __main__ Mario example
    pong_ctx = f'{PONG_INTRO} {VIS_INTRO}'
    brk_ctx = f'{BREAKOUT_INTRO} {VIS_INTRO}'
    q_act = 'What should you do right now?'
    add('game_pong_atari_up', 'game', 'public', 'functional', save_public('game_pong_atari_up', pong(160, 210, (78, 64), 150, 110)),
        pong_ctx, [(q_act, PONG_OPTS, 0)], note='160x210 Atari size; ball far above the right paddle')
    add('game_pong_atari_level', 'game', 'public', 'borderline', save_public('game_pong_atari_level', pong(160, 210, (96, 118), 112, 90)),
        pong_ctx, [(q_act, PONG_OPTS, None)], note='ball 4 px from the paddle centre line (teacher threshold 6)')
    add('game_breakout_atari_left', 'game', 'public', 'functional',
        save_public('game_breakout_atari_left', breakout(160, 210, (30, 150), 110, missing={(0, 3), (0, 4), (1, 4), (2, 9)})),
        brk_ctx, [(q_act, BREAKOUT_OPTS, 0)], note='160x210; ball left of the paddle, falling side unknown in a still')
    add('game_breakout_atari_noball', 'game', 'public', 'functional',
        save_public('game_breakout_atari_noball', breakout(160, 210, None, 72)),
        brk_ctx, [(q_act, BREAKOUT_OPTS, 3)], note='160x210; no ball in play')
    add('game_breakout_atari_center', 'game', 'public', 'borderline',
        save_public('game_breakout_atari_center', breakout(160, 210, (80, 160), 74, missing={(0, 8), (1, 8)})),
        brk_ctx, [(q_act, BREAKOUT_OPTS, None)], note='ball just above the paddle centre')
    add('game_pong_nes_multi', 'game', 'public', 'functional', save_public('game_pong_nes_multi', pong(256, 240, (40, 170), 60, 150)),
        pong_ctx, [(q_act, PONG_OPTS, 1), ('Is the ball above or below your paddle?', ['above', 'below', 'level with it'], 1),
                   ('Which half of the screen is the ball in?', ['left half', 'right half'], 0)],
        note='256x240 NES size, three questions (three slots)')
    add('game_breakout_nes_multi', 'game', 'public', 'functional',
        save_public('game_breakout_nes_multi', breakout(256, 240, (120, 130), 40, missing={(0, 1), (0, 2), (1, 2), (3, 14)})),
        brk_ctx, [(q_act, BREAKOUT_OPTS, 1), ('Is a ball in play?', ['no', 'yes'], 1)], note='256x240, two slots')
    # 3) public synthetic scenes, non-square and larger than the grid (dynamic grid != G)
    im, d = canvas(640, 480)
    for (x, y, r) in ((120, 140, 50), (400, 110, 60), (300, 340, 55)):
        d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 30, 30))
    add('synth_count_circles', 'synth', 'public', 'functional', save_public('synth_count_circles', im),
        'Count the objects in the image.', [('How many circles are in the image?', ['1', '2', '3', '4', '5'], 2)])

    im, d = canvas(800, 600)
    d.rectangle([120, 120, 440, 440], fill=(40, 170, 40))
    d.polygon([(600, 380), (700, 380), (650, 290)], fill=(220, 30, 30))
    add('synth_square_color', 'synth', 'public', 'functional', save_public('synth_square_color', im),
        CAULDRON_CTX, [('What color is the large square?', ['red', 'green', 'blue', 'yellow'], 1)])

    add('synth_arrow_down', 'synth', 'public', 'functional', save_public('synth_arrow_down', arrow(300, 500, -90, 360)),
        CAULDRON_CTX, [('Which direction does the arrow point?', ['up', 'down', 'left', 'right'], 1)])

    im, d = canvas(640, 480)
    d.text((320, 240), 'STOP', font=font(150), fill=(0, 0, 0), anchor='mm')
    add('synth_word', 'synth', 'public', 'functional', save_public('synth_word', im),
        CAULDRON_CTX, [('What word is written in the image?', ['STOP', 'SHOP', 'SPOT', 'STEP'], 0)])

    add('synth_bars_max', 'synth', 'public', 'functional',
        save_public('synth_bars_max', bars(800, 600, [180, 260, 470, 330], [(220, 30, 30), (40, 170, 40), (30, 60, 220), (230, 200, 20)])),
        CAULDRON_CTX, [('Which bar is the tallest?', ['the red bar', 'the green bar', 'the blue bar', 'the yellow bar'], 2)])

    im, d = canvas(640, 480)
    d.ellipse([90, 150, 270, 330], fill=(220, 30, 30))
    d.rectangle([400, 160, 560, 320], fill=(30, 60, 220))
    add('synth_multi_shapes', 'synth', 'public', 'functional', save_public('synth_multi_shapes', im),
        CAULDRON_CTX, [('What color is the circle?', ['red', 'green', 'blue'], 0),
                       ('Which shape is on the left side?', ['circle', 'square', 'triangle'], 0),
                       ('How many shapes are in the image?', ['1', '2', '3', '4'], 1)], note='three slots')

    # borderline scenes (reference top-1 is expected to be unsaturated; purpose is recorded before any inference)
    im, d = canvas(640, 480)
    d.rectangle([0, 0, 319, 479], fill=(220, 30, 30))
    d.rectangle([320, 0, 639, 479], fill=(30, 60, 220))
    add('synth_half_half', 'synth', 'public', 'borderline', save_public('synth_half_half', im),
        COLOR_CTX, [('What is the dominant color in the image?', ['red', 'blue'], None)], note='exact 50/50 split')

    add('synth_bars_close', 'synth', 'public', 'borderline',
        save_public('synth_bars_close', bars(800, 600, [400, 406], [(120, 120, 120), (120, 120, 120)], gap=120)),
        CAULDRON_CTX, [('Which bar is the tallest?', ['the left bar', 'the right bar'], 1)], note='6 px difference out of 400')

    im, d = canvas(300, 500)
    d.ellipse([60, 160, 240, 340], fill=(128, 128, 128))
    add('synth_unanswerable', 'synth', 'public', 'borderline', save_public('synth_unanswerable', im),
        CAULDRON_CTX, [('What is the name of the artist who drew this image?', ['Anna', 'Ben', 'Chloe', 'David'], None)],
        note='not answerable from the image')

    im, d = canvas(800, 600)
    for (x, y) in ((110, 90), (300, 160), (520, 80), (690, 210), (180, 420), (430, 380), (640, 500)):
        d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=(0, 0, 0))
    add('synth_count_dots', 'synth', 'public', 'borderline', save_public('synth_count_dots', im),
        'Count the objects in the image.', [('How many dots are in the image?', ['5', '6', '7', '8', '9'], 2)], note='7 dots')

    add('synth_arrow_diagonal', 'synth', 'public', 'borderline', save_public('synth_arrow_diagonal', arrow(640, 480, 45, 380)),
        CAULDRON_CTX, [('Which direction does the arrow point?', ['up', 'right'], None)], note='arrow at 45 degrees up-right')

    im, d = canvas(300, 500)
    d.ellipse([110, 210, 190, 290], fill=(30, 60, 220))
    add('synth_center_circle', 'synth', 'public', 'borderline', save_public('synth_center_circle', im),
        CAULDRON_CTX, [('Is the circle on the left or the right side of the image?', ['left', 'right'], None)], note='circle centred')

    im, d = canvas(640, 480)
    d.rectangle([170, 90, 470, 390], fill=(0, 128, 128))
    add('synth_teal', 'synth', 'public', 'borderline', save_public('synth_teal', im),
        CAULDRON_CTX, [('What color is the square?', ['green', 'blue'], None)], note='teal (0,128,128)')

    im, d = canvas(800, 600, (128, 128, 128))
    add('synth_gray_multi', 'synth', 'public', 'borderline', save_public('synth_gray_multi', im),
        CAULDRON_CTX, [('Is the image bright?', ['no', 'yes'], None), ('Is the image dark?', ['no', 'yes'], None)],
        note='uniform mid-gray 128; two slots')

    # already-square originals (C2: author arm == g{G}_mrope arm when the original is exactly GxG)
    im, d = canvas(256, 256)
    d.polygon([(128, 40), (220, 210), (36, 210)], fill=(30, 60, 220))
    add('synth_sq256_triangle', 'synth', 'public', 'control', save_public('synth_sq256_triangle', im),
        CAULDRON_CTX, [('What shape is shown?', ['circle', 'square', 'triangle'], 2)], note='256x256 original (C2 at G=256)')
    im, d = canvas(512, 512)
    d.ellipse([96, 96, 416, 416], fill=(40, 170, 40))
    d.rectangle([60, 430, 200, 480], fill=(220, 30, 30))
    add('synth_sq512_circle', 'synth', 'public', 'control', save_public('synth_sq512_circle', im),
        CAULDRON_CTX, [('What shape is the large green object?', ['circle', 'square', 'triangle'], 0),
                       ('What color is the small rectangle?', ['red', 'green', 'blue'], 0)], note='512x512 original (C2 at G=512); two slots')

    # 5) no image (C1 controls). Contexts/questions from decider/infer.py __main__ and decider/vision.py __main__ (None, ex)
    add('text_billing_multi', 'text', 'public', 'control', None,
        'My card was charged twice for the same purchase and I want the extra charge refunded.',
        [('Which department should handle this?', ['billing', 'technical support', 'sales'], 0),
         ("What is the customer's sentiment?", ['angry', 'neutral', 'happy'], None),
         ('Does this need a refund action?', ['no', 'yes'], 1)], note='decider/infer.py __main__ demo 1, three slots')
    add('text_lights', 'text', 'public', 'control', None, 'hey can u turn the lights off in the kitchen',
        [('What is the intent?', ['smart home control', 'set alarm', 'play music', 'none of the above'], 0),
         ('Is this request toxic?', ['no', 'yes'], 0)], note='decider/infer.py __main__ demo 2, verbatim options (no neutralisation in vision prepare)')
    add('text_finance', 'text', 'public', 'control', None, 'The quarterly report shows revenue fell 12% while costs rose sharply.',
        [('What is the financial sentiment?', ['bearish', 'neutral', 'bullish'], 0)], note='decider/infer.py __main__ demo 3')
    add('text_mario_noimage', 'text', 'public', 'control', None, MARIO_CTX,
        [(MARIO_QS[0][0], MARIO_QS[0][1], None), (MARIO_QS[1][0], MARIO_QS[1][1], None)],
        note='decider/vision.py __main__ second request (None, ex)')
    add('text_umbrella', 'text', 'public', 'borderline', None, 'I am going for a walk tomorrow afternoon.',
        [('Should I bring an umbrella?', ['no', 'yes'], None)], note='not decidable from the context')

    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids))
    summary = {}
    for r in rows:
        key = f"{r['family']}/{r['tier']}/{r['purpose']}"
        summary[key] = summary.get(key, 0) + 1
    out = dict(schema='decider2bv-fixtures-r1', snapshot_revision=REVISION, pillow=PIL.__version__,
               generator='scripts/make_fixtures.py',
               n_rows=len(rows), n_image_rows=sum(r['image'] is not None for r in rows),
               n_slots=sum(len(r['questions']) for r in rows),
               n_image_slots=sum(len(r['questions']) for r in rows if r['image'] is not None),
               summary=summary, rows=rows)
    write_json('fixtures/fixtures.json', out)
    print(out['n_rows'], 'rows', out['n_image_rows'], 'image rows', out['n_slots'], 'slots', out['n_image_slots'], 'image slots')
    print(summary)


if __name__ == '__main__':
    main()

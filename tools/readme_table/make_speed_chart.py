#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the model table in README.md.

    python3 tools/readme_table/make_speed_chart.py

Reads the phone column of the rows between the models-table markers, which make_readme_table.py writes from the
published manifests. Writes docs/images/speed-light.svg and docs/images/speed-dark.svg. A range counts at its low value.
make_readme_table.py --write runs this too, so the figure follows the table.
"""
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
README = os.path.join(ROOT, "README.md")
OUT = os.path.join(ROOT, "docs", "images")
START, END = "<!-- models-table:start -->", "<!-- models-table:end -->"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "GPU": "#2a78d6", "CPU": "#eb6834", "NPU": "#1baf7a"},
    "dark": {"surface": "#0d1117", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#34342f",
             "GPU": "#3987e5", "CPU": "#d95926", "NPU": "#199e70"},
}
BACKENDS = ["GPU", "CPU", "NPU"]
WIDTH, HEIGHT = 860, 440
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 84, 54
X_TICKS = [0.25, 0.5, 1, 2, 4, 8]
Y_TICKS = [3, 10, 30, 100, 300]
# A few points named on the figure: (name in the table, label offset x, label offset y, text-anchor)
LABELS = [("gemma-3-270m-it-NPU", 10, 4, "start"), ("LFM2.5-1.2B-Instruct", 10, -8, "start"),
          ("Qwen3.5-4B", 10, 16, "start"), ("granite-4.1-8b", -10, 18, "end")]


def rows():
    text = open(README, encoding="utf-8").read()
    if START not in text or END not in text:
        raise SystemExit(f"README.md has no {START} / {END} markers")
    table = text[text.index(START) + len(START):text.index(END)]
    out = []
    for line in table.strip().splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[3] == "—":
            continue
        m = re.match(r"(.+?) (CPU|GPU|NPU) ([\d.]+)", cells[3])
        if not m:
            raise SystemExit(f"cannot read the phone cell: {cells[3]!r}")
        name = re.match(r"\[([^\]]+)\]", cells[0]).group(1)
        out.append({"name": name, "params": float(cells[1].rstrip("B")), "device": m.group(1),
                    "backend": m.group(2), "speed": float(m.group(3))})
    return out


def draw(data, theme):
    c = THEMES[theme]
    x_lo, x_hi = math.log10(0.2), math.log10(11)
    y_lo, y_hi = math.log10(3), math.log10(320)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(v):
        return LEFT + (math.log10(v) - x_lo) / (x_hi - x_lo) * plot_w

    def y(v):
        return TOP + plot_h - (math.log10(v) - y_lo) / (y_hi - y_lo) * plot_h

    def text(px, py, body, size, fill, anchor="start", weight="400"):
        return (f'<text x="{px:.1f}" y="{py:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                f'text-anchor="{anchor}">{body}</text>')

    devices = sorted({d["device"] for d in data})
    title = f"{len(data)} published conversions measured on a phone"
    parts = [text(0, 24, title, 20, c["text"], weight="600"),
             text(0, 46, "Decode speed by model size, fastest verified backend on " + ", ".join(devices), 13, c["muted"])]
    lx = 7
    for b in BACKENDS:
        parts.append(f'<circle cx="{lx}" cy="65" r="5" fill="{c[b]}"/>')
        parts.append(text(lx + 11, 70, b, 14, c["muted"]))
        lx += 70
    for t in Y_TICKS:
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{c["grid"]}"/>')
        parts.append(text(LEFT - 8, y(t) + 4, f"{t:g}", 13, c["muted"], "end"))
    for t in X_TICKS:
        parts.append(text(x(t), HEIGHT - BOTTOM + 20, f"{t:g}B", 13, c["muted"], "middle"))
    parts.append(text(LEFT + plot_w / 2, HEIGHT - 8, "Parameters (log scale)", 13, c["muted"], "middle"))
    parts.append(f'<text transform="translate(14 {TOP + plot_h / 2:.1f}) rotate(-90)" font-size="13" fill="{c["muted"]}" '
                 f'text-anchor="middle">Tokens per second (log scale)</text>')
    for d in data:
        parts.append(f'<circle cx="{x(d["params"]):.1f}" cy="{y(d["speed"]):.1f}" r="5.5" fill="{c[d["backend"]]}" '
                     f'stroke="{c["surface"]}" stroke-width="1.5"/>')
    by_name = {d["name"]: d for d in data}
    for name, dx, dy, anchor in LABELS:
        d = by_name.get(name)
        if d:
            parts.append(text(x(d["params"]) + dx, y(d["speed"]) + dy, name, 12, c["text"], anchor))
    desc = (f"Scatter plot of decode speed against model size for {len(data)} conversions measured on a phone, "
            "colored by CPU, GPU or NPU.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n<title id="t">{title}</title>\n'
            f'<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n")


def main():
    data = rows()
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        path = os.path.join(OUT, f"speed-{theme}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(draw(data, theme))
        print("wrote", os.path.relpath(path, ROOT))
    print(len(data), "rows,", {b: sum(d["backend"] == b for d in data) for b in BACKENDS})


if __name__ == "__main__":
    main()

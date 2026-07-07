"""Falcon3 cross-framework decode-throughput bar chart (team-share format).
Mac M4 Max GPU, 4-bit, 512-token steady-state decode tok/s."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

groups = ["Falcon3-3B", "Falcon3-7B"]
series = {  # framework -> [3B, 7B] decode tok/s
    "Core AI / CoreML": [225, 104],
    "MLX":              [219, 104],
    "LiteRT-LM":        [154, 85],
}
colors = {"Core AI / CoreML": "#0a84ff", "MLX": "#ff9f0a", "LiteRT-LM": "#30d158"}

x = np.arange(len(groups))
w = 0.26
fig, ax = plt.subplots(figsize=(8, 5))
for i, (name, vals) in enumerate(series.items()):
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel("Decode throughput (tok/s)", fontsize=12)
ax.set_title("Falcon3 on-device decode — Mac M4 Max GPU, 4-bit (512-tok steady-state)\n"
             "higher is better", fontsize=12)
ax.legend(frameon=False, fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 260)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = "reports/falcon3-decode-chart.png"
fig.savefig(out, dpi=150)
print("wrote", out)

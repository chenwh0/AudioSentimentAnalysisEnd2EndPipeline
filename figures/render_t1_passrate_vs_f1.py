import json
import pandas as pd
from matplotlib.lines import Line2D
from _t1_style import *

spec, data = load_spec("t1_passrate_vs_f1")
apply_rc()
df = pd.read_csv(data)
corr = json.loads((R / "Milestone2/T1_final/results/correlations.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharey=True)
fig.subplots_adjust(top=0.82, bottom=0.27, left=0.08, right=0.98, wspace=0.08)
panels = [("pass_A123", "A123", "Assertions A1, A2, A3"), ("pass_A123_A4a", "A123_A4a", "A1, A2, A3 + A4a (classifier agreement)")]
for ax, (col, key, title) in zip(axes, panels):
    ax.grid(True, axis="both")
    for _, r in df.iterrows():
        ax.scatter(r[col], r.macro_f1, s=60, marker=ASR_MARKER[r.asr], color=SENT_COLOR[r.sentiment],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
    c = corr[key]
    ax.text(0.03, 0.97, f"Pearson r = {c['pearson']:+.2f}\nSpearman ρ = {c['spearman']:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK2)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel(spec["axes"]["x"])
axes[0].set_ylabel(spec["axes"]["y"])
axes[0].set_ylim(0.25, 0.68)
h_sent = [Line2D([], [], ls="", marker="o", ms=8, color=SENT_COLOR[s], label=SENT_LABEL[s]) for s in SENT_ORDER]
h_asr = [Line2D([], [], ls="", marker=ASR_MARKER[a], ms=8, color=INK2, label=a) for a in ASR_ORDER]
fig.legend(handles=h_sent, title="color: sentiment model", loc="lower left", bbox_to_anchor=(0.08, 0.085),
           ncol=2, fontsize=8.5, title_fontsize=8.5)
fig.legend(handles=h_asr, title="shape: speech-to-text model", loc="lower left", bbox_to_anchor=(0.62, 0.085),
           ncol=3, fontsize=8.5, title_fontsize=8.5)
finish(fig, spec, 130)

import json
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from _t1_style import *

spec, data = load_spec("xai_focus_ratio_by_class")
apply_rc()
df = pd.read_csv(data)
S = json.loads((data.parent / "summary.json").read_text())
n_excl = int(df.focus_ratio_m1.isna().sum())
d = df[df.focus_ratio_m1.notna()].copy()
n_allflag = int((d.frac_flagged >= 1.0).sum())
cnt = d.degradation_class.value_counts()
order = sorted(cnt.index, key=lambda c: (-cnt[c], XAI_CLASS_ORDER.index(c)))
fr = S["focus_ratio_m1"]
mean, (lo, hi), n = fr["mean"], fr["ci"], fr["n"]
assert n == len(d)

fig, ax = plt.subplots(figsize=(9, 6.2))
fig.subplots_adjust(top=0.86, bottom=0.25, left=0.11, right=0.97)
ax.axhspan(lo, hi, color="#0072B2", alpha=0.10, lw=0, zorder=1)
ax.axhline(1.0, color=INK2, ls="--", lw=1, zorder=2)
rng = np.random.default_rng(8001)
for i, c in enumerate(order):
    g = d[d.degradation_class == c]
    xs = i + rng.uniform(-0.26, 0.26, len(g)) if len(g) > 1 else np.array([i])
    for ok in (1, 0):
        m = (g.correct == ok).to_numpy()
        ax.scatter(xs[m], g.focus_ratio_m1[m], s=34, color=CORRECT_COLOR[ok], marker="o" if ok else "X",
                   edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.plot([i - 0.3, i + 0.3], [g.focus_ratio_m1.mean()] * 2, color=INK, lw=2, zorder=3, solid_capstyle="round")
ax.set_xticks(range(len(order)), [f"{XAI_CLASS_LABEL[c]}\n(n = {cnt[c]})" for c in order])
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylim(0, 1.75)
ax.text(len(order) - 0.45, 1.0 + 0.02, "1.0 = no concentration", ha="right", va="bottom", fontsize=8, color=INK2)
ax.text(len(order) - 0.45, lo - 0.02,
        f"overall mean {mean:.3f}, 95% CI {lo:.3f} to {hi:.3f} (n = {n} clips)", ha="right", va="top",
        fontsize=8, color="#0072B2")
ax.text(0.01, 0.98, f"Not shown: {n_excl} clips with no flagged 20 ms frame (ratio undefined).\n"
        f"{n_allflag} of the {n} shown clips are flagged in every frame, so their ratio is 1 by construction.",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
ax.set_xlabel(spec["axes"]["x"])
ax.set_ylabel("focus ratio\n(attribution mass in flagged audio ÷ share of audio flagged)")
handles = [Line2D([], [], ls="", marker="o", ms=6, color=CORRECT_COLOR[1], label="sentiment correct (Cardiff)"),
           Line2D([], [], ls="", marker="X", ms=6, color=CORRECT_COLOR[0], label="sentiment wrong (Cardiff)"),
           Line2D([], [], color=INK, lw=2, label="class mean"),
           Patch(color="#0072B2", alpha=0.18, label="95% CI of the overall mean (bootstrap)")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.075), ncol=4, fontsize=8)
note = ("Focus ratio = share of the attribution that falls in degradation-flagged audio divided by the share of audio that is "
        "flagged (a 20 ms frame is flagged when the degradation detector AADD gives any class probability >= 0.5). "
        "Class = AADD's clip-level dominant class; 'clean' clips here have a few flagged frames.")
finish(fig, spec, 115, note=note)

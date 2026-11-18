import json
import numpy as np, pandas as pd
from _t1_style import *

spec, data = load_spec("t1_macro_f1_grid")
apply_rc()
df = pd.read_csv(data).query("source == 'full'")
base = json.loads((R / "Milestone2/T1_final/results/baseline.json").read_text())["macro_f1"]

fig, ax = plt.subplots(figsize=(8, 5.6))
fig.subplots_adjust(top=0.82, bottom=0.2, left=0.1, right=0.98)
w, gap = 0.17, 0.03
x = np.arange(len(ASR_ORDER))
for i, s in enumerate(SENT_ORDER):
    d = df[df.sentiment == s].set_index("asr").loc[ASR_ORDER]
    xs = x + (i - 1.5) * (w + gap)
    yerr = [d.macro_f1 - d.macro_f1_ci_lo, d.macro_f1_ci_hi - d.macro_f1]
    ax.bar(xs, d.macro_f1, width=w, color=SENT_COLOR[s], label=SENT_LABEL[s], zorder=2)
    ax.errorbar(xs, d.macro_f1, yerr=yerr, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
ax.axhline(base, ls="--", lw=1.2, color=INK2, zorder=4)
ax.text(3.02, base + 0.008, f"always-neutral\nbaseline ({base:.4f})", ha="right", va="bottom",
        fontsize=8.5, color=INK2)
ax.set_xlim(-0.5, 3.05)
ax.set_xticks(x, ASR_ORDER)
ax.set_ylim(0, 0.85)
ax.set_xlabel(spec["axes"]["x"])
ax.set_ylabel(wrap(spec["axes"]["y"], 45))
ax.legend(title="sentiment model", ncol=2, loc="upper right", fontsize=8.5, title_fontsize=8.5)
finish(fig, spec, 105)

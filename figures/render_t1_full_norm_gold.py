import numpy as np, pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from _t1_style import *

spec, data = load_spec("t1_full_norm_gold")
apply_rc()
df = pd.read_csv(data).query("source in ['full', 'norm', 'gold']")
SRC = ["full", "norm", "gold"]
SRC_LABEL = {"full": "full: speech-to-text transcript", "norm": "norm: same text, lowercased, no punctuation",
             "gold": "gold: human transcript (same for every speech model)"}


def tint(c, f=0.5):
    return tuple(1 - (1 - v) * f for v in to_rgb(c))


def style(src, hue):
    if src == "full":
        return dict(color=hue)
    if src == "norm":
        return dict(facecolor=tint(hue, 0.45), hatch="////", edgecolor=hue, linewidth=0)
    return dict(color="#898781")


fig, axes = plt.subplots(1, 4, figsize=(13, 5.0), sharey=True)
fig.subplots_adjust(top=0.8, bottom=0.27, left=0.06, right=0.99, wspace=0.08)
plt.rcParams["hatch.linewidth"] = 0.8
w, gap = 0.24, 0.03
x = np.arange(len(ASR_ORDER))
for ax, s in zip(axes, SENT_ORDER):
    hue = SENT_COLOR[s]
    for i, src in enumerate(SRC):
        d = df[(df.sentiment == s) & (df.source == src)].set_index("asr").loc[ASR_ORDER]
        xs = x + (i - 1) * (w + gap)
        ax.bar(xs, d.macro_f1, width=w, zorder=2, **style(src, hue))
        ax.errorbar(xs, d.macro_f1, yerr=[d.macro_f1 - d.macro_f1_ci_lo, d.macro_f1_ci_hi - d.macro_f1],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.set_title(SENT_LABEL[s], fontsize=9.5, loc="left", color=INK)
    ax.set_xticks(x, ASR_ORDER, fontsize=8.5)
    ax.set_ylim(0, 0.75)
fig.supxlabel(spec["axes"]["x"], y=0.175, fontsize=9.5, color=INK2)
axes[0].set_ylabel(spec["axes"]["y"])
neutral = "#52514e"
handles = [Patch(label=SRC_LABEL[src], **style(src, neutral)) for src in SRC]
fig.legend(handles=handles, title="text fed to the sentiment model (bar color within a panel = that panel's sentiment model; gold bars gray)",
           loc="lower left", bbox_to_anchor=(0.06, 0.06), ncol=3, fontsize=8.5, title_fontsize=8.5)
finish(fig, spec, 160)

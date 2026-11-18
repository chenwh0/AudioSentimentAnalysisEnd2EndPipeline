import json
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from _t1_style import *

spec, data = load_spec("xai_wer_flagged_vs_clean")
apply_rc()
df = pd.read_csv(data)
H2 = json.loads((data.parent / "summary.json").read_text())["H2"]
d = df[df.wer_flagged.notna() & df.wer_clean.notna()].copy()
cnt = df.degradation_class.value_counts()            # fixed class order = count over all 50 clips
classes = [c for c in XAI_CLASS_ORDER if c in set(d.degradation_class)]


def size(nw):
    return 6 * nw


fig, ax = plt.subplots(figsize=(7.6, 6.9))
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.11, right=0.72)
ax.grid(axis="x", visible=True)
ax.plot([0, 1.08], [0, 1.08], color=INK2, ls="--", lw=1, zorder=2)
ax.text(0.80, 0.86, "y = x", rotation=45, ha="center", va="center", fontsize=8, color=INK2)
ax.text(0.03, 1.04, "more errors in degraded audio", fontsize=8, color=MUTED, va="top")
ax.text(1.04, 0.03, "more errors in clean audio", fontsize=8, color=MUTED, ha="right")
for z, (_, r) in enumerate(d.sort_values("n_words", ascending=False).iterrows()):   # big points drawn first
    c = r.degradation_class; col = XAI_CLASS_COLOR[c]
    kw = dict(facecolor="none", edgecolor=col, linewidth=1.4) if c == "clean" else dict(color=col, edgecolor=SURFACE, linewidth=1.2)
    ax.scatter(r.wer_clean, r.wer_flagged, s=size(r.n_words), marker=XAI_CLASS_MARKER[c], zorder=4 + z * 0.01, **kw)
ax.set_xlim(-0.04, 1.08); ax.set_ylim(-0.04, 1.08); ax.set_aspect("equal")
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel(spec["axes"]["x"]); ax.set_ylabel(spec["axes"]["y"])
pd_ = H2["paired_wer_diff_flagged_minus_clean"]
box = (f"Pooled over all {len(df)} clips\n"
       f"error rate in flagged audio: {H2['pooled_err_rate_flagged']:.3f} ({H2['n_ref_words_flagged']} words)\n"
       f"error rate in clean audio:   {H2['pooled_err_rate_clean']:.3f} ({H2['n_ref_words_clean']} words)\n"
       f"per-clip difference (flagged - clean),\n{pd_['n']} clips: {pd_['mean']:+.3f}, 95% CI {pd_['ci'][0]:+.3f} to {pd_['ci'][1]:+.3f}")
ax.text(0.98, 0.35, box, transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK2,
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
ch = [Line2D([], [], ls="", marker=XAI_CLASS_MARKER[c], ms=7, color=XAI_CLASS_COLOR[c],
             markerfacecolor="none" if c == "clean" else XAI_CLASS_COLOR[c],
             label=f"{XAI_CLASS_LABEL[c]} ({(d.degradation_class == c).sum()})") for c in classes]
l1 = fig.legend(handles=ch, title="dominant degradation class\n(clips plotted)", loc="upper left",
                bbox_to_anchor=(0.74, 0.84), fontsize=8, title_fontsize=8, alignment="left")
sh = [plt.scatter([], [], s=size(n), color=MUTED, edgecolor=SURFACE, label=f"{n} words") for n in (12, 20, 36)]
fig.legend(handles=sh, title="transcript length", loc="upper left", bbox_to_anchor=(0.74, 0.5), fontsize=8,
           title_fontsize=8, labelspacing=1.2, borderpad=0.8, alignment="left")
finish(fig, spec, 92, note=WER_NOTE + " Only clips with both flagged and clean audio are plotted.")

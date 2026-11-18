import pandas as pd
from _t1_style import *

spec, data = load_spec("t2_accuracy_vs_size")
apply_rc()
df = pd.read_csv(data).set_index("asr").loc[T2_ORDER]

# label offsets in points (dx, dy, ha) per panel, chosen so close points (AventIQ 147 MB / whisper-tiny 155 MB) separate
OFF_L = {"AventIQ": (0, 8, "center"), "s2t-small": (-8, 0, "right"), "whisper-tiny": (8, 6, "left"),
         "whisper-base": (0, 8, "center"), "moonshine-base": (8, -10, "left")}
OFF_R = {"AventIQ": (-8, -8, "right"), "s2t-small": (8, 0, "left"), "whisper-tiny": (8, 4, "left"),
         "whisper-base": (8, 4, "left"), "moonshine-base": (0, -14, "center")}

fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True)
fig.subplots_adjust(top=0.82, bottom=0.27, left=0.07, right=0.98, wspace=0.22)
for ax in (axl, axr):
    ax.grid(axis="x", visible=True)
for a, r in df.iterrows():
    c, m = T2_COLOR[a], T2_MARKER[a]
    axl.errorbar(r.size_loaded_MB, r.cardiff_macro_f1, yerr=[[r.cardiff_macro_f1 - r.ci_lo], [r.ci_hi - r.cardiff_macro_f1]],
                 fmt=m, color=c, ecolor=c, elinewidth=1.2, capsize=3, ms=7, mec=SURFACE, mew=0.8, zorder=3, label=a)
    axr.plot(r.size_loaded_MB, r.wer_gold, m, color=c, ms=7, mec=SURFACE, mew=0.8, zorder=3)
    # AventIQ and whisper-base labels sit above their CI caps so they clear the neighbouring error bars
    lval = r.ci_hi if a in ("AventIQ", "whisper-base") else r.cardiff_macro_f1
    for ax, val, off in ((axl, lval, OFF_L[a]), (axr, r.wer_gold, OFF_R[a])):
        ax.annotate(a, (r.size_loaded_MB, val), xytext=off[:2], textcoords="offset points", ha=off[2],
                    va="center", fontsize=8.5, color=INK)
axl.set_xlim(60, 340)
axl.set_ylim(0.55, 0.67)
axr.set_ylim(0, 0.7)
axl.set_ylabel(spec["axes"]["y_left"]); axr.set_ylabel(spec["axes"]["y_right"])
axl.set_title("Sentiment macro-F1 (bars = 95% confidence interval; higher is better)", loc="left", fontsize=9.5)
axr.set_title("Word error rate vs human transcripts (lower is better)", loc="left", fontsize=9.5)
fig.supxlabel(spec["axes"]["x"], y=0.175, fontsize=9.5, color=INK2)
fig.legend(*axl.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=5, fontsize=8.5)
note = (MACRO_F1_NOTE + " " + WER_NOTE)
finish(fig, spec, 140, note=note)

import pandas as pd
from _t1_style import *

spec, data = load_spec("wsp_results_table")
apply_rc()
N_PARAMS = 37_760_640  # unique parameters of whisper-tiny (edits_fp17.json n_params_unique)
ORDER = ["base", "fp10", "fp17"]
COLOR = {"base": "#6b6a66", "fp10": "#E69F00", "fp17": "#0072B2"}
LABEL = {"base": "base (unedited)", "fp10": "fp10 (failed)", "fp17": "fp17 (registered)"}
df = pd.read_csv(data).set_index("checkpoint").loc[ORDER]

panels = [("test_wer_gold", "wer_ci_lo", "wer_ci_hi", spec["axes"]["x_left"], "Word error rate (lower is better)"),
          ("delta_wer_vs_base", "delta_ci_lo", "delta_ci_hi", spec["axes"]["x_middle"], "Change in word error rate vs base"),
          ("cardiff_macro_f1", "f1_ci_lo", "f1_ci_hi", spec["axes"]["x_right"], "Sentiment macro-F1 (higher is better)")]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
fig.subplots_adjust(top=0.76, bottom=0.32, left=0.115, right=0.80, wspace=0.12)
y = {c: i for i, c in enumerate(reversed(ORDER))}  # base on top
for ax, (col, lo, hi, xlabel, title) in zip(axes, panels):
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    for c in ORDER:
        r = df.loc[c]
        ax.errorbar(r[col], y[c], xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt="o", color=COLOR[c],
                    ecolor=COLOR[c], elinewidth=1.6, capsize=3, ms=6.5, mec=SURFACE, mew=0.8, zorder=3,
                    label=LABEL[c])
    ax.set_xlabel(wrap(xlabel, 42), fontsize=8.8)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_ylim(-0.6, 2.6)
axes[1].axvline(0, color=INK2, lw=0.9, ls="--", zorder=1)
axes[0].set_xlim(0.15, 0.47)
axes[1].set_xlim(-0.03, 0.23)
axes[2].set_xlim(0.57, 0.65)
axes[0].set_yticks([y[c] for c in ORDER]); axes[0].set_yticklabels([LABEL[c] for c in ORDER])
for t, c in zip(axes[0].get_yticklabels(), ORDER):
    t.set_color(COLOR[c] if c != "base" else INK); t.set_fontweight("bold" if c == "fp17" else "normal")

# row-end annotations: changed weights (% of all parameters) and long/short transcript counts
axes[2].text(1.04, 2.45, "weights changed;\nclips with too-long / too-short\ntranscripts", transform=axes[2].get_yaxis_transform(),
             fontsize=7.5, color=MUTED, va="bottom", ha="left")
for c in ORDER:
    r = df.loc[c]
    pct = 100 * r.n_changed_weights / N_PARAMS
    s = f"{int(r.n_changed_weights):,} ({pct:.2f}%)\nlong {int(r.long)} / short {int(r.short)}"
    axes[2].text(1.04, y[c], s, transform=axes[2].get_yaxis_transform(), fontsize=7.8, color=INK2, va="center", ha="left")
r17 = df.loc["fp17"]  # fp17's interval is too narrow to read at this scale, so label it directly
axes[1].annotate(f"{r17.delta_wer_vs_base:+.4f}\n[{r17.delta_ci_lo:+.4f}, {r17.delta_ci_hi:+.4f}]", (r17.delta_ci_hi, y["fp17"]),
                 xytext=(8, 0), textcoords="offset points", fontsize=7.8, color=COLOR["fp17"], va="center", ha="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.45, 0.135), ncol=3, fontsize=8.5)
note = (MACRO_F1_NOTE + " " + WER_NOTE + f" Percent of weights is out of {N_PARAMS:,} parameters. Long / short = test clips whose "
        "transcript has more than 1.5x / fewer than 0.5x the words of the human transcript. Bars are 95% bootstrap confidence intervals.")
finish(fig, spec, 150, note=note)

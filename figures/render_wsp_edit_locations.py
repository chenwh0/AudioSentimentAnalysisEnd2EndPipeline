import json, re
import pandas as pd
from matplotlib.patches import Patch
from _t1_style import *

spec, data = load_spec("wsp_edit_locations")
apply_rc()
WSP = R / "Milestone2" / "WSP_Whisper"
loc = pd.read_csv(data)
audit = pd.read_csv(WSP / "results" / "layer_audit.csv")
edits = json.loads((WSP / "edits" / "edits_fp17.json").read_text())


def group(layer):
    if layer.startswith("model.encoder.conv"): return "enc_conv"
    if layer == "model.encoder.embed_positions": return "enc_pos"
    m = re.match(r"model\.(encoder|decoder)\.layers\.(\d)\.", layer)
    if m: return f"{m.group(1)[:3]}_b{m.group(2)}"
    if layer in ("model.decoder.embed_tokens", "proj_out"): return "tied"
    if layer == "model.decoder.embed_positions": return "dec_pos"
    if layer in ("model.encoder.layer_norm", "model.decoder.layer_norm"): return "final_ln"
    raise ValueError(layer)


ORDER = ["enc_conv", "enc_pos"] + [f"enc_b{i}" for i in range(4)] + ["tied", "dec_pos"] + [f"dec_b{i}" for i in range(4)] + ["final_ln"]
LABEL = {"enc_conv": "encoder\nconv stem", "enc_pos": "encoder\npositions", "tied": "decoder\nembed_tokens /\nproj_out (tied)",
         "dec_pos": "decoder\npositions", "final_ln": "final layer\nnorms (enc+dec)"}
LABEL.update({f"enc_b{i}": f"encoder\nblock {i}" for i in range(4)}); LABEL.update({f"dec_b{i}": f"decoder\nblock {i}" for i in range(4)})
audit = audit[audit.layer != "proj_out"]  # tied: same storage as embed_tokens, count once
size = audit.groupby(audit.layer.map(group)).numel.sum().reindex(ORDER)
assert size.sum() == edits["n_params_unique"], size.sum()
changed = {ck: loc[(loc.checkpoint == ck) & (loc.layer == "model.decoder.embed_tokens")].iloc[0] for ck in ("fp17", "fp10")}

BLUE, ORANGE = "#0072B2", "#E69F00"
fig, ax = plt.subplots(figsize=(12.5, 7.2))
fig.subplots_adjust(top=0.88, bottom=0.19, left=0.07, right=0.985)
ax.set_yscale("log"); ax.set_ylim(1, 1e11)  # headroom above the bars holds the legend and the text box
x = range(len(ORDER)); w = 0.62
# hollow outline = all weights in the group; filled = weights changed
ax.bar(x, size.values, width=w, fill=False, edgecolor=AXIS, linewidth=1.0, zorder=2)
for i, g in enumerate(ORDER):
    if g == "tied":
        r17, r10 = changed["fp17"], changed["fp10"]
        ax.bar(i - w / 4, r17.n_changed, width=w / 2, color=BLUE, zorder=3)
        ax.bar(i + w / 4, r10.n_changed, width=w / 2, color=ORANGE, alpha=0.45, zorder=3)
        ax.annotate(f"fp17: {int(r17.n_changed):,} weights\nin {int(r17.n_rows_changed)} token rows",
                    (i - w / 4, r17.n_changed), xytext=(-14, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=BLUE, fontweight="bold")
    else:
        ax.text(i, 1.5, "0", ha="center", va="bottom", fontsize=9, color=INK2, fontweight="bold")
    ax.text(i, size[g] * 1.25, f"{size[g]/1e6:.2f}M" if size[g] >= 1e5 else f"{size[g]:,}", ha="center",
            va="bottom", fontsize=7.2, color=MUTED)
ax.set_yticks([10 ** k for k in range(0, 9, 2)]); ax.minorticks_off()
ax.set_xticks(list(x)); ax.set_xticklabels([LABEL[g] for g in ORDER], fontsize=7.8)
ax.get_xticklabels()[ORDER.index("tied")].set_color(BLUE)
ax.plot([ORDER.index("tied") - 0.5] * 2, [1, 2e8], color=GRID, lw=1, zorder=1)
ax.text(2.5, 6e7, "ENCODER (audio in)", ha="center", fontsize=8.5, color=INK2)
ax.text(10, 6e7, "DECODER (text out)", ha="center", fontsize=8.5, color=INK2)
ax.set_xlabel(spec["axes"]["x"]); ax.set_ylabel(spec["axes"]["y"] + ", log scale")

e5 = next(e for e in edits["edits"] if e["edit_id"] == "E5"); e2 = next(e for e in edits["edits"] if e["edit_id"] == "E2net")
box = ("Why one bar: whisper-tiny's decoder uses the same matrix to read a token in (embed_tokens)\n"
       "and to score it as the next output (proj_out). Scaling one row therefore changes both\n"
       "that token's input embedding and its output logit.\n"
       f"fp17 edit groups (rows x 384 weights):\n"
       f"  - {len(e5['rows'])} digit-token rows x {e5['factor']}  (fewer digits, more spelled-out numbers)\n"
       f"  - {len(e2['rows'])} rows for 'cause', 'gonna', 'gotta' (both case variants) x {e2['factor']:.2f}\n"
       f"  = {len(e5['rows']) + len(e2['rows'])} rows, {edits['n_changed']:,} weights, "
       f"{100 * edits['fraction_changed']:.2f}% of {edits['n_params_unique']:,}")
ax.text(0.985, 0.985, box, transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=7.9, color=INK, family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=AXIS, lw=0.8), zorder=5)
handles = [Patch(fc="none", ec=AXIS, label="all weights in the group (outline)"),
           Patch(fc=BLUE, label="changed in fp17 (registered)"),
           Patch(fc=ORANGE, alpha=0.45, label=f"changed in fp10 (failed, {int(changed['fp10'].n_changed):,} weights)")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.99), fontsize=8.2)
note = ("Bars labelled 0 have no changed weights: encoder, attention, feed-forward, positional and layer-norm weights are "
        "bit-identical to the base model. The tied matrix is counted once (51,865 tokens x 384).")
finish(fig, spec, 150, note=note)

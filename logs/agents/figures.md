
## 2026-11-18 figure agent: T1 figures (t1_macro_f1_grid, t1_passrate_vs_f1, t1_full_norm_gold)
- sha256 of both data files matched the specs.
- Palette: Okabe-Ito in fixed order cardiff #0072B2, distilbert #D55E00, siebert #009E73, tabularisai #E69F00, shared via figures/_t1_style.py. dataviz validator (--pairs all, light) passes: worst CVD dE 11.0, normal-vision 15.6. The skill's default 4-slot palette failed all-pairs (yellow vs orange 13.7), so it was not used. #E69F00 is below 3:1 contrast; legends name every series.
- Rendered with /home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python (Agg), PNG 200 dpi + SVG. Scripts: figures/render_<name>.py.
- Visual check: grid v1 had the baseline label on top of the whisper-tiny bars and the legend over a Cardiff error bar. Fixed by putting the label in a right-hand margin and raising ylim to 0.85; re-checked and clean. Other two figures had no collisions.
- Known/accepted: in the scatter, AventIQ and whisper-tiny DistilBERT points nearly coincide (A123 0.985 vs 0.983, macro_f1 0.458 vs 0.457), so they overlap; a surface ring separates them. Points are identified by color (sentiment) + shape (speech model) instead of 24 text labels (spec "label" column), to avoid collisions.
- Spec caption for t1_full_norm_gold says "ASR" and was reproduced verbatim; the spec owner may want "speech-to-text".

## 2026-11-18 figure agent: T2 figures (t2_wer_per_variant, t2_accuracy_vs_size)
- sha256 of wer_test.csv and size_vs_accuracy.csv matched the specs.
- figures/_t1_style.py extended (backward compatible): finish() takes an optional `note` (default still the macro-F1 note); added T2_ORDER/T2_COLOR/T2_MARKER (Okabe-Ito #CC79A7, #56B4E9, #009E73, #D55E00, #000000 in order AventIQ, s2t-small, whisper-tiny, whisper-base, moonshine-base, double-encoded with marker shape) and WER_NOTE. T1 grid script re-run after the edit to confirm it still renders.
- t2_wer_per_variant: single blue hue, solid / light tint / hatched for wer_gold / no-fillers / pseudo-ref; whisper-large-v3 drawn as dashed line at 0.179 (its pseudo-ref 0.0 is not plotted). Deviation: the spec asks for the s2t-small-OLD(batch8) row as "a hatched bar", but hatching already encodes the pseudo-ref series, so that group is drawn in gray with an italic gray tick label "s2t-small, before the batch fix" and its own legend entry.
- Footnote: t2_wer_per_variant defines word error rate; t2_accuracy_vs_size defines macro-F1 and word error rate.
- Visual check v1: value labels near 0.15-0.20 were crossed by the dashed reference line (fixed with background boxes on labels); in the size figure the legend overlapped the footnote, and the s2t-small, AventIQ and whisper-base labels touched neighbouring error bars (fixed by larger bottom margin, moving labels left / above CI caps, xlim 60-340). Re-checked v3: no collisions or overflow.
- Caption note: "Moonshine-base has ... the lowest sentiment macro-F1" holds only by 0.0006 (0.6027 vs s2t-small 0.6032), well inside CIs.

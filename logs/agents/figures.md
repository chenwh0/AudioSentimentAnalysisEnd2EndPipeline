
## 2026-11-18 figure agent: T1 figures (t1_macro_f1_grid, t1_passrate_vs_f1, t1_full_norm_gold)
- sha256 of both data files matched the specs.
- Palette: Okabe-Ito in fixed order cardiff #0072B2, distilbert #D55E00, siebert #009E73, tabularisai #E69F00, shared via figures/_t1_style.py. dataviz validator (--pairs all, light) passes: worst CVD dE 11.0, normal-vision 15.6. The skill's default 4-slot palette failed all-pairs (yellow vs orange 13.7), so it was not used. #E69F00 is below 3:1 contrast; legends name every series.
- Rendered with /home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python (Agg), PNG 200 dpi + SVG. Scripts: figures/render_<name>.py.
- Visual check: grid v1 had the baseline label on top of the whisper-tiny bars and the legend over a Cardiff error bar. Fixed by putting the label in a right-hand margin and raising ylim to 0.85; re-checked and clean. Other two figures had no collisions.
- Known/accepted: in the scatter, AventIQ and whisper-tiny DistilBERT points nearly coincide (A123 0.985 vs 0.983, macro_f1 0.458 vs 0.457), so they overlap; a surface ring separates them. Points are identified by color (sentiment) + shape (speech model) instead of 24 text labels (spec "label" column), to avoid collisions.
- Spec caption for t1_full_norm_gold says "ASR" and was reproduced verbatim; the spec owner may want "speech-to-text".

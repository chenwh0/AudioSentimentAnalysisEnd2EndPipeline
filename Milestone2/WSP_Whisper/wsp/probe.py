"""Pass 4: teacher-forced probe on 64 practice clips (selected by gold-text content only) to measure
logit margins of filler / colloquial tokens. Analysis of the captured logits is free."""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_counter as FP, apply_edits as AE, eval_harness as H
D = f"{FP.W}/wsp/diag"
GROUPS = {"uh": [" uh", " Uh", "Uh"], "um": [" um", " Um", "Um"], "gonna": [" gonna", " Gonna"],
          "kinda": [" kinda", " Kinda"], "gotta": [" gotta", " Gotta"], "cause": [" cause", " Cause"]}
prac = FP.practice_table()
words = set(GROUPS)
prac["n_target"] = prac["gold"].map(lambda g: sum(w in words for w in H.normalize_text(g).split()))
sel = prac.sort_values(["n_target", "original_index"], ascending=[False, True]).head(64).reset_index(drop=True)
sel[["original_index", "file", "n_target"]].to_csv(f"{D}/probe_clips.csv", index=False)
model = AE.build("base").cuda(); proc = AE.load_processor(); tok = proc.tokenizer
gid = {g: [tok.encode(v, add_special_tokens=False) for v in vs] for g, vs in GROUPS.items()}
for g, l in gid.items():
    assert all(len(x) == 1 for x in l), (g, l)
gid = {g: [x[0] for x in l] for g, l in gid.items()}
res, n = FP.teacher_forced_pass(model, proc, sel, sel["gold"].tolist(), purpose="teacher-forced probe: logit margins of filler/colloquial tokens at gold positions",
                                hypothesis_id="H1_fillers+H2_colloquial", edit_id="none")
E = model.proj_out.weight.detach().float().cpu()
out = {"pass_id": n}
# consistency: logits == hidden @ E^T
ids0, lg0, h0 = res[0]
out["logit_recon_maxabs"] = float((h0 @ E.T - lg0.float()).abs().max())
for g, toks in gid.items():
    main = toks[0]
    pos_ratio, neg_ratio = [], []
    for ids, lg, h in res:
        lg = lg.float()
        for p in range(3, len(ids) - 1):  # logits at p predict ids[p+1]
            nxt = ids[p + 1]
            l = lg[p]
            lt = l[toks].max().item()
            other = l.clone(); other[toks] = -1e9
            lm = other.max().item()
            r = lm / lt if lt > 0 else np.inf  # scale factor needed for the group to win
            (pos_ratio if nxt in toks else neg_ratio).append(r)
    pos, neg = np.array(pos_ratio), np.array(neg_ratio)
    grid = {}
    for s in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]:
        grid[str(s)] = {"wins_at_gold": int((pos < s).sum()), "false_at_other": int((neg < s).sum())}
    out[g] = {"tokens": toks, "n_gold_pos": int(len(pos)), "n_other_pos": int(len(neg)),
              "pos_ratio_quantiles": np.quantile(pos[np.isfinite(pos)], [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if np.isfinite(pos).any() else [],
              "pos_frac_negative_logit": float((~np.isfinite(pos)).mean()) if len(pos) else None,
              "grid_scale": grid}
json.dump(out, open(f"{D}/probe_margins.json", "w"), indent=1)
print(json.dumps(out, indent=1))

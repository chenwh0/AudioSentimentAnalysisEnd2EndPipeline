"""Step 5: diagnostic on practice (3 passes x 64 clips) + layer audit (weights free, activations from those passes)."""
import collections, json, os, re, sys
import numpy as np, pandas as pd, torch, jiwer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_counter as FP, apply_edits as AE, eval_harness as H

D = f"{FP.W}/wsp/diag"
prac = FP.practice_table()
order = np.random.default_rng(8001).permutation(len(prac))
prac = prac.iloc[order].reset_index(drop=True)
prac[["original_index", "file"]].to_csv(f"{D}/practice_order.csv", index=False)
model = AE.build("base").cuda(); proc = AE.load_processor()

# ---- tied weights
tied = model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr()

# ---- activation hooks
stats = {}
def mk(name):
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(t): return
        t = t.detach().float()
        if name.endswith("conv1") or name.endswith("conv2"):
            t = t.transpose(1, 2)
        flat = t.reshape(-1, t.shape[-1])
        s = stats.setdefault(name, {"n": 0, "sum": 0.0, "sq": 0.0, "amax": 0.0, "ch": torch.zeros(t.shape[-1], device=t.device)})
        s["n"] += flat.numel(); s["sum"] += flat.sum().item(); s["sq"] += (flat ** 2).sum().item()
        s["amax"] = max(s["amax"], flat.abs().max().item())
        if t.shape[-1] <= 4096: s["ch"] += flat.abs().sum(0)
        s["rows"] = s.get("rows", 0) + flat.shape[0]
    return fn
hooks = []
for name, mod in model.named_modules():
    if re.search(r"(conv[12]|layers\.\d+$|layers\.\d+\.(self_attn|encoder_attn|fc1|fc2|self_attn_layer_norm|encoder_attn_layer_norm|final_layer_norm)$|encoder\.layer_norm$|decoder\.layer_norm$|embed_tokens$|embed_positions$)", name):
        hooks.append((mod, mk(name)))

outs = []
for b in range(3):
    o, wer, n = FP.run_pass(model, proc, prac.iloc[b * 64:(b + 1) * 64], purpose="base diagnostic + activation capture for layer audit", hypothesis_id="DIAG", edit_id="none", hooks=hooks)
    outs.append(o); print("pass", n, "wer", round(wer, 4), flush=True)
dg = pd.concat(outs).reset_index(drop=True)

# ---- error analysis
def ngram_rep(t, n=3):
    w = H.normalize_text(t).split()
    c = collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values()) if c else 0
dg["max_trigram_rep"] = dg["transcript"].map(ngram_rep)
dg["hyp_words"] = dg["transcript"].map(lambda t: len(H.normalize_text(t).split()))
dg["ratio"] = dg["hyp_words"] / dg["nref_gold"].clip(lower=1)
sub, dele, ins = collections.Counter(), collections.Counter(), collections.Counter()
S = Dl = I = 0
for r, h in zip(dg["gold"], dg["transcript"]):
    r, h = H.normalize_text(r), H.normalize_text(h)
    if not r: continue
    o = jiwer.process_words(r, h if h else "")
    rw, hw = r.split(), h.split()
    S += o.substitutions; Dl += o.deletions; I += o.insertions
    for a in o.alignments[0]:
        if a.type == "substitute":
            for k in range(a.ref_end_idx - a.ref_start_idx):
                if a.hyp_start_idx + k < a.hyp_end_idx: sub[(rw[a.ref_start_idx + k], hw[a.hyp_start_idx + k])] += 1
        elif a.type == "delete":
            for w in rw[a.ref_start_idx:a.ref_end_idx]: dele[w] += 1
        elif a.type == "insert":
            for w in hw[a.hyp_start_idx:a.hyp_end_idx]: ins[w] += 1
nref = int(dg["nref_gold"].sum())
digits = int(dg["transcript"].str.contains(r"\d").sum())
gold_fill = collections.Counter(w for g in dg["gold"] for w in H.normalize_text(g).split() if w in {"uh", "um", "uhm", "er", "ah", "like", "mm", "hmm"})
hyp_fill = collections.Counter(w for g in dg["transcript"] for w in H.normalize_text(g).split() if w in {"uh", "um", "uhm", "er", "ah", "like", "mm", "hmm"})
summary = {"n_clips": len(dg), "n_ref_words": nref, "corpus_wer_gold": (S + Dl + I) / nref, "S": S, "D": Dl, "I": I,
           "clips_trigram_rep>=3": int((dg["max_trigram_rep"] >= 3).sum()), "long>1.5": int((dg["ratio"] > 1.5).sum()),
           "short<0.5": int((dg["ratio"] < 0.5).sum()), "clips_with_digits": digits, "tied_proj_out_embed_tokens": tied,
           "top_sub": [f"{a}->{b}:{c}" for (a, b), c in sub.most_common(40)], "top_del": dele.most_common(40),
           "top_ins": ins.most_common(40), "gold_fillers": dict(gold_fill), "hyp_fillers": dict(hyp_fill)}
json.dump(summary, open(f"{D}/diag_summary.json", "w"), indent=1, default=str)
dg.drop(columns=["wav"]).to_csv(f"{D}/diag_practice_transcripts.csv", index=False)

# ---- layer audit
rows = []
mods = dict(model.named_modules())
for name, p in model.named_parameters(remove_duplicate=False):
    w = p.detach().float()
    mod_name = name.rsplit(".", 1)[0]
    # activation stats of the owning module (or parent layer for proj sub-modules)
    act_key = mod_name if mod_name in stats else re.sub(r"\.(q_proj|k_proj|v_proj|out_proj)$", "", mod_name)
    a = stats.get(act_key)
    note = ""
    if a:
        mean = a["sum"] / a["n"]; std = (a["sq"] / a["n"] - mean ** 2) ** 0.5
        ch = a["ch"] / max(a["rows"], 1)
        med = ch.median().item(); top = torch.topk(ch, min(3, len(ch)))
        act = f"mod={act_key};mean={mean:.4g};std={std:.4g};absmax={a['amax']:.4g};ch_absmean_median={med:.4g};top_ch={top.indices.tolist()}@{[round(v, 3) for v in top.values.tolist()]}"
        if med > 0 and top.values[0].item() / med > 8: note += f"outlier channel {top.indices[0].item()} ({top.values[0].item() / med:.1f}x median); "
    else:
        act = ""
    if w.dim() == 2 and min(w.shape) > 1:
        sv = torch.linalg.svdvals(w.cuda()).cpu()
        extra = f"sv_max={sv[0]:.4g};sv_min={sv[-1]:.4g};eff_rank={(sv.sum() ** 2 / (sv ** 2).sum()).item():.1f}"
    else:
        extra = ""
    if name in ("proj_out.weight", "model.decoder.embed_tokens.weight"):
        note += f"TIED proj_out<->embed_tokens={tied}: editing a vocab row changes both input embedding and output logit of that token; "
    rows.append({"layer": mod_name, "tensor": name, "shape": "x".join(map(str, w.shape)), "numel": w.numel(),
                 "norm": w.norm().item(), "mean": w.mean().item(), "std": w.std().item() if w.numel() > 1 else 0.0,
                 "absmax": w.abs().max().item(), "svd": extra, "activation_stats": act, "note": note})
pd.DataFrame(rows).to_csv(f"{FP.W}/results/layer_audit.csv", index=False)
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("top")}, default=str))
print("SUB", summary["top_sub"][:25]); print("DEL", summary["top_del"][:25]); print("INS", summary["top_ins"][:25])

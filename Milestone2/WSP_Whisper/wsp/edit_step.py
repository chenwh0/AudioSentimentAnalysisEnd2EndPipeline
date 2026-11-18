"""Propose -> measure (one practice pass on fixed set M) -> keep or revert.
Usage: python edit_step.py '<json edit spec>'
spec: {"edit_id","hypothesis_id","hypothesis","op":"scale_rows","tensor","rows":[...],"factor":f}
State: edits/state.json (accepted edits, in order), edits/edit_log.jsonl (every proposal + outcome).
M = first 64 clips of wsp/diag/practice_order.csv (their base transcripts come from diagnostic pass 1)."""
import json, os, sys, datetime
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_counter as FP, apply_edits as AE, eval_harness as H

E = f"{FP.W}/edits"; D = f"{FP.W}/wsp/diag"
STATE = f"{E}/state.json"


def apply_spec(model, spec):
    params = dict(model.named_parameters(remove_duplicate=False))
    p = params[spec["tensor"]]
    with torch.no_grad():
        if spec["op"] == "scale_rows":
            p.data[spec["rows"]] *= spec["factor"]
        elif spec["op"] == "add_beta_component":
            # rows += k * (-(row . beta) / |beta|^2) * beta : cancels fraction k of the constant logit offset
            # that the final decoder LayerNorm bias (beta) gives each token; context-dependent part untouched
            beta = dict(model.named_parameters())["model.decoder.layer_norm.bias"].detach()
            for r in spec["rows"]:
                c = -(p.data[r] @ beta) / (beta @ beta)
                p.data[r] += spec["k"] * c * beta
        else:
            raise ValueError(spec["op"])


def build_state(edits):
    m = AE.load_base()
    for s in edits:
        apply_spec(m, s)
    return m


def make_patch(model):
    base = dict(AE.load_base().named_parameters())
    cur = dict(model.named_parameters())
    patch = {}
    for n, p in cur.items():
        d = (p.detach().cpu().view(-1) != base[n].detach().view(-1)).nonzero().view(-1)
        if len(d):
            patch[n] = (d, p.detach().cpu().view(-1)[d].clone(), base[n].detach().view(-1)[d].clone())
    return patch


def save_checkpoint(name, edits):
    m = build_state(edits)
    patch = make_patch(m)
    AE.write_patch({n: (i, v) for n, (i, v, o) in patch.items()}, AE.patch_path(name))
    total = sum(p.numel() for p in m.parameters())
    nchg = sum(len(i) for i, _, _ in patch.values())
    meta = {"checkpoint": name, "base": AE.MODEL_ID, "revision": AE.REVISION, "passes_used": FP.passes_used(),
            "edits": edits, "n_changed": int(nchg), "n_params_unique": int(total), "fraction_changed": nchg / total,
            "per_tensor": {n: {"n_changed": int(len(i)), "old_values_sha_note": "old values stored below (first 8)",
                               "old_first8": o[:8].tolist(), "new_first8": v[:8].tolist()} for n, (i, v, o) in patch.items()},
            "tied_note": "proj_out.weight shares storage with model.decoder.embed_tokens.weight; row edits change input embedding and output logit together; patch is stored under the embed_tokens name only (named_parameters dedups)."}
    json.dump(meta, open(f"{E}/edits_{name}.json", "w"), indent=1)
    # verify rebuild
    r = AE.build(name)
    for (a, pa), (b, pb) in zip(r.named_parameters(), m.named_parameters()):
        assert torch.equal(pa, pb), a
    return meta


def main_isolated(spec):
    """Measure base + this edit alone on spec['measure_ids'] (practice clips from the diagnostic, whose base
    transcripts are known). Valid when greedy decoding can only change where edited tokens become/stop being argmax."""
    ids = spec["measure_ids"]
    dg = pd.read_csv(f"{D}/diag_practice_transcripts.csv").set_index("original_index").loc[ids]
    base_wer = float(dg["err_gold"].sum() / dg["nref_gold"].sum())
    prac = FP.practice_table().set_index("original_index").loc[ids].reset_index()
    model = build_state([spec]).cuda(); proc = AE.load_processor()
    out, wer, n = FP.run_pass(model, proc, prac, purpose=f"isolated measure of edit {spec['edit_id']} on its target clips", hypothesis_id=spec["hypothesis_id"], edit_id=spec["edit_id"])
    ref = spec.get("compare_to_wer", base_wer)  # when spec replaces an accepted edit: must beat that edit's set WER
    keep = wer < ref - 1e-12
    rec = dict({k: v for k, v in spec.items() if k != "rows"}, n_rows=len(spec.get("rows", [])), ts=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               pass_id=n, isolated=True, set_wer_base=base_wer, set_wer_after=wer, decision="keep" if keep else "revert")
    with open(f"{E}/edit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    out.drop(columns=["wav"]).to_csv(f"{D}/M_{spec['edit_id']}.csv", index=False)
    if keep:
        st = json.load(open(STATE))
        if spec.get("replaces"):
            st["accepted"] = [a for a in st["accepted"] if a["edit_id"] != spec["replaces"]]
        st["accepted"].append(spec); st["M_stale"] = True
        json.dump(st, open(STATE, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("edit_id", "pass_id", "set_wer_base", "set_wer_after", "decision")}))


def main(spec):
    if spec.get("isolated"):
        return main_isolated(spec)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"accepted": [], "M_wer": None, "M_transcripts": None}
    order = pd.read_csv(f"{D}/practice_order.csv")
    Mids = order["original_index"].tolist()[:64]
    prac = FP.practice_table().set_index("original_index").loc[Mids].reset_index()
    if st["M_wer"] is None:
        dg = pd.read_csv(f"{D}/diag_practice_transcripts.csv").set_index("original_index").loc[Mids]
        st["M_wer"] = float(dg["err_gold"].sum() / dg["nref_gold"].sum()); st["M_base_wer"] = st["M_wer"]
        st["M_errs"] = dg["err_gold"].astype(int).tolist()
    model = build_state(st["accepted"] + [spec]).cuda()
    proc = AE.load_processor()
    out, wer, n = FP.run_pass(model, proc, prac, purpose=f"measure edit {spec['edit_id']} on M", hypothesis_id=spec["hypothesis_id"], edit_id=spec["edit_id"])
    keep = wer < st["M_wer"] - 1e-12
    hyps = out["transcript"].tolist()
    fill = sum(w in ("uh", "um") for t in hyps for w in H.normalize_text(t).split())
    rec = dict(spec, ts=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), pass_id=n,
               M_wer_before=st["M_wer"], M_wer_after=wer, M_base_wer=st["M_base_wer"], decision="keep" if keep else "revert",
               hyp_uh_um_count=fill, clips_changed=int(sum(np.array(out["err_gold"]) != np.array(st["M_errs"]))))
    with open(f"{E}/edit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    out.drop(columns=["wav"]).to_csv(f"{D}/M_{spec['edit_id']}.csv", index=False)
    if keep:
        st["accepted"].append(spec); st["M_wer"] = wer; st["M_errs"] = out["err_gold"].astype(int).tolist()
    json.dump(st, open(STATE, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("edit_id", "pass_id", "M_wer_before", "M_wer_after", "decision", "hyp_uh_um_count", "clips_changed")}))


if __name__ == "__main__":
    if sys.argv[1] == "--rebase":
        st = json.load(open(STATE))
        Mids = pd.read_csv(f"{D}/practice_order.csv")["original_index"].tolist()[:64]
        prac = FP.practice_table().set_index("original_index").loc[Mids].reset_index()
        model = build_state(st["accepted"]).cuda()
        out, wer, n = FP.run_pass(model, proc := AE.load_processor(), prac, purpose="re-baseline cumulative accepted edits on M", hypothesis_id="REBASE", edit_id="+".join(a["edit_id"] for a in st["accepted"]))
        rec = {"edit_id": "REBASE", "pass_id": n, "accepted": [a["edit_id"] for a in st["accepted"]], "M_wer_before": st["M_wer"], "M_wer_after": wer, "M_base_wer": st["M_base_wer"], "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        with open(f"{E}/edit_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        out.drop(columns=["wav"]).to_csv(f"{D}/M_rebase_{n}.csv", index=False)
        st["M_wer"] = wer; st["M_errs"] = out["err_gold"].astype(int).tolist(); st["M_stale"] = False
        json.dump(st, open(STATE, "w"), indent=1); print(json.dumps(rec))
    elif sys.argv[1] == "--save-subset":  # candidate checkpoint from a subset of accepted edits (for a holdout eval)
        st = json.load(open(STATE)); keep_ids = sys.argv[3].split(",")
        sub = [a for a in st["accepted"] if a["edit_id"] in keep_ids]
        print(json.dumps({k: v for k, v in save_checkpoint(sys.argv[2], sub).items() if k not in ("per_tensor", "edits")}))
    elif sys.argv[1] == "--save":
        st = json.load(open(STATE))
        print(json.dumps({k: v for k, v in save_checkpoint(sys.argv[2], st["accepted"]).items() if k != "per_tensor"}))
    else:
        main(json.loads(sys.argv[1]))

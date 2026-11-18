"""Passes 16-17: generalization check on the 58 practice clips never used so far (practice_order[192:]):
base vs current accepted state. Looks for repetition loops / regressions outside M."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_counter as FP, apply_edits as AE, edit_step as ES
D = f"{FP.W}/wsp/diag"
ids = pd.read_csv(f"{D}/practice_order.csv")["original_index"].tolist()[192:]
prac = FP.practice_table().set_index("original_index").loc[ids].reset_index()
proc = AE.load_processor(); st = json.load(open(ES.STATE))
ob, wb, nb = FP.run_pass(AE.build("base").cuda(), proc, prac, purpose="generalization check: base on unused practice clips", hypothesis_id="GENCHECK", edit_id="none")
oe, we, ne = FP.run_pass(ES.build_state(st["accepted"]).cuda(), proc, prac, purpose="generalization check: current state on unused practice clips", hypothesis_id="GENCHECK", edit_id="+".join(a["edit_id"] for a in st["accepted"]))
ob.drop(columns=["wav"]).to_csv(f"{D}/gen_base.csv", index=False); oe.drop(columns=["wav"]).to_csv(f"{D}/gen_state.csv", index=False)
rec = {"edit_id": "GENCHECK", "passes": [nb, ne], "n": len(ids), "wer_base": wb, "wer_state": we, "state": [a["edit_id"] for a in st["accepted"]],
       "max_hyp_words_ratio_state": float((oe["transcript"].str.split().str.len() / oe["nref_gold"].clip(lower=1)).max())}
open(f"{ES.E}/edit_log.jsonl", "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec))

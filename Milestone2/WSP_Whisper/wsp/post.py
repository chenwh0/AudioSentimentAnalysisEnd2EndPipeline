"""edit_locations.csv for every saved checkpoint + transcripts_whisper-tiny-wsp.csv (registered checkpoint, test + dev)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_edits as AE
RES = f"{AE.W}/results"
reg = [l.split(":", 1)[1].strip() for l in open(f"{RES}/REGISTERED.txt") if l.startswith("registered_checkpoint")][0]
m = AE.load_base(); shapes = {n: p.shape for n, p in m.named_parameters(remove_duplicate=False)}
rows = []
for ck in ["fp10", "fp17"]:
    patch = AE.read_patch(AE.patch_path(ck))
    for n, (idx, val) in patch.items():
        numel = 1
        for s in shapes[n]: numel *= s
        nrows = len(set((idx // shapes[n][1]).tolist())) if len(shapes[n]) == 2 else None
        for layer_name in ([n, "proj_out.weight (tied, same storage)"] if n == "model.decoder.embed_tokens.weight" else [n]):
            rows.append({"checkpoint": ck, "layer": layer_name.rsplit(".", 1)[0], "tensor": layer_name, "n_changed": int(len(idx)),
                         "n_rows_changed": nrows, "fraction_changed": len(idx) / numel, "registered": ck == reg})
pd.DataFrame(rows).to_csv(f"{RES}/edit_locations.csv", index=False)
t = pd.concat([pd.read_csv(f"{RES}/eval_{reg}_test.csv").assign(split="test"), pd.read_csv(f"{RES}/eval_{reg}_dev.csv").assign(split="dev")])
t["transcript"] = t["transcript"].fillna("")
t[["file", "transcript"]].to_csv(f"{RES}/transcripts_whisper-tiny-wsp.csv", index=False)
print(pd.DataFrame(rows).to_string()); print(len(t), "transcripts, registered", reg)

"""Rebuild a full whisper-tiny model from the pinned base + a sparse patch.

Patch format (edits/patch_<name>.safetensors): for each edited parameter P two tensors
  "P::idx"  int64, flat indices into P (row-major)
  "P::val"  float32, new values at those indices
No other change is made (generation_config, tokenizer, processor untouched).
Note: proj_out.weight is tied to model.decoder.embed_tokens.weight on this checkpoint,
so an edit to either name changes both (recorded in the edit json).
"""
import os
import torch
from safetensors.torch import load_file, save_file

MODEL_ID = "openai/whisper-tiny"
REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
W = "/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/WSP_Whisper"
EDITS = f"{W}/edits"


def load_base():
    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32)
    return m.eval()


def load_processor():
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained(MODEL_ID, revision=REVISION)


def patch_path(name):
    return f"{EDITS}/patch_{name}.safetensors"


def apply_patch(model, patch):
    """patch: dict name -> (idx LongTensor, val FloatTensor). In place, no grad."""
    params = dict(model.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name, (idx, val) in patch.items():
            p = params[name]
            flat = p.data.view(-1)
            flat[idx.to(flat.device)] = val.to(flat.device, flat.dtype)
    return model


def read_patch(path):
    t = load_file(path)
    names = sorted({k.rsplit("::", 1)[0] for k in t})
    return {n: (t[f"{n}::idx"], t[f"{n}::val"]) for n in names}


def write_patch(patch, path):
    out = {}
    for n, (idx, val) in patch.items():
        out[f"{n}::idx"] = idx.to(torch.int64).contiguous().cpu()
        out[f"{n}::val"] = val.to(torch.float32).contiguous().cpu()
    save_file(out, path)


def build(name):
    """name == 'base' -> pinned base; otherwise base + edits/patch_<name>.safetensors."""
    m = load_base()
    if name != "base":
        apply_patch(m, read_patch(patch_path(name)))
    return m


def count_changed(name):
    if name == "base":
        return 0
    return int(sum(len(i) for i, _ in read_patch(patch_path(name)).values()))

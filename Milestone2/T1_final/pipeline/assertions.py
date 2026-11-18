"""Per-clip assertions for the audio-sentiment pipeline (pure functions; imported by Phase 2 and the T3 services).

A1  transcript has at least 2 words
A2  words / duration is within [0.5, 5.0]
A3  classifier score >= t            (t tuned on the development slice by the guide's balanced-accuracy rule)
A4a classifier agreement             (the configuration's mapped label equals a second classifier's mapped label)
A4b audio quality                    (AADD clean fraction = 1 - degraded_fraction >= t_b, t_b tuned like A3)
Edge cases: empty / None / NaN transcript -> A1, A2 False; duration <= 0 -> A2 False; NaN score -> A3 False;
missing second label -> A4a False; NaN clean fraction -> A4b False.
"""
import math
import numpy as np

def _words(transcript):
    if transcript is None: return 0
    if isinstance(transcript, float) and math.isnan(transcript): return 0
    return len(str(transcript).split())

def a1_min_words(transcript, min_words=2):
    return _words(transcript) >= min_words

def a2_speech_rate(transcript, duration_s, lo=0.5, hi=5.0):
    try: d = float(duration_s)
    except (TypeError, ValueError): return False
    if not (d > 0) or math.isnan(d): return False
    rate = _words(transcript) / d
    return lo <= rate <= hi

def a3_confidence(score, t):
    try: s = float(score)
    except (TypeError, ValueError): return False
    if math.isnan(s): return False
    return s >= t

def a4a_agreement(label, second_label):
    if label is None or second_label is None: return False
    if isinstance(label, float) or isinstance(second_label, float): return False
    return str(label) == str(second_label)

def a4b_audio_quality(clean_fraction, t):
    try: c = float(clean_fraction)
    except (TypeError, ValueError): return False
    if math.isnan(c): return False
    return c >= t

def tune_threshold(scores, correct, grid):
    """Guide rule: pick t in grid with the highest balanced accuracy of (score >= t) at predicting correctness.
    First maximum wins (smallest t among ties). Returns (t, balanced_accuracy, {t: ba})."""
    from sklearn.metrics import balanced_accuracy_score
    scores = np.asarray(scores, dtype=float); correct = np.asarray(correct).astype(int)
    scores = np.where(np.isnan(scores), -np.inf, scores)
    table = {}
    for g in grid:
        table[round(float(g), 2)] = float(balanced_accuracy_score(correct, (scores >= g).astype(int)))
    t = max(table, key=table.get)
    return t, table[t], table

def assert_all(transcripts, durations, scores, t, labels=None, second_labels=None, clean_fractions=None, t_b=None):
    """Vectorized convenience wrapper. Returns dict of boolean numpy arrays."""
    n = len(transcripts)
    out = {
        "A1": np.array([a1_min_words(x) for x in transcripts]),
        "A2": np.array([a2_speech_rate(x, d) for x, d in zip(transcripts, durations)]),
        "A3": np.array([a3_confidence(s, t) for s in scores]),
    }
    if labels is not None and second_labels is not None:
        out["A4a"] = np.array([a4a_agreement(a, b) for a, b in zip(labels, second_labels)])
    if clean_fractions is not None and t_b is not None:
        out["A4b"] = np.array([a4b_audio_quality(c, t_b) for c in clean_fractions])
    out["A123"] = out["A1"] & out["A2"] & out["A3"]
    return out

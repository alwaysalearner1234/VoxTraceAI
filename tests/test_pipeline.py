"""Smoke tests: determinism + gradient sanity (fast, offline, no TTS)."""
import numpy as np
from voxtrace.features import extract
from voxtrace.pipeline import analyze_arrays
from voxtrace import dataset as ds


def _tone(dur=3.0, sr=16000, f0drift=True):
    t = np.arange(int(dur * sr)) / sr
    f0 = 130 + (25 * np.sin(2 * np.pi * t / 1.5) if f0drift else 0)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    y = (np.sin(ph) + 0.3 * np.sin(2 * ph)).astype(np.float32) * 0.4
    gate = (np.sin(2 * np.pi * t * 2.2) > -0.2).astype(np.float32) * 0.8 + 0.2
    return (y * gate).astype(np.float32)


def test_features_deterministic():
    y = _tone()
    a, b = extract(y), extract(y.copy())
    assert np.allclose(a["f0_var"], b["f0_var"], equal_nan=True)
    assert np.allclose(a["mfcc"], b["mfcc"])


def test_self_comparison_scores_100():
    tx = " ".join(["voice"] * 40)
    good = _tone(4.0, f0drift=True)
    rep = analyze_arrays(good, good, tx)
    assert rep["scores"]["total"] == 100.0
    assert rep["flaws"] == []


def test_gradient_mumble_worse_than_clean():
    tx = " ".join(["voice"] * 40)
    good = _tone(4.0, f0drift=True)
    bad = ds.mumble(good, 16000)
    rep_good = analyze_arrays(good, good, tx)
    rep_bad = analyze_arrays(good, bad, tx)
    assert rep_bad["scores"]["total"] < rep_good["scores"]["total"]


def test_inserted_pause_is_grounded():
    tx = " ".join(["voice"] * 40)
    good = _tone(6.0, f0drift=True)
    bad, lab = ds.insert_pause(good, 16000, at_frac=0.5, pause_s=1.0)
    rep = analyze_arrays(good, bad, tx)
    assert len(rep["flaws"]) >= 1
    s, e = lab["start"], lab["end"]
    overlaps = [f for f in rep["flaws"]
                if f["start"] < e and f["end"] > s]
    assert overlaps, f"no flaw overlaps injected pause {[s, e]}"


def test_tempo_direction_labels():
    # respeed semantics: rate>1 faster/shorter, rate<1 slower/longer
    y = _tone(2.0)
    fast = ds.respeed(y, 16000, 1.28)
    slow = ds.respeed(y, 16000, 0.78)
    assert len(fast) < len(y) < len(slow)

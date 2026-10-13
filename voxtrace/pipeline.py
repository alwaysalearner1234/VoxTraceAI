"""End-to-end analysis pipeline: (baseline_wav, test_wav, transcript) -> report."""
import numpy as np
from .utils import load_mono
from .features import extract, summarize
from .alignment import get_word_timestamps, warp_baseline_to_test
from .grounding import ground_flaws
from .scoring import score_utterance


def analyze_arrays(y_base: np.ndarray, y_test: np.ndarray, transcript: str,
                   sr: int = 16000, base_path_hint: str = "",
                   test_path_hint: str = "") -> dict:
    fb = extract(np.asarray(y_base, dtype=np.float32), sr)
    ft = extract(np.asarray(y_test, dtype=np.float32), sr)
    warp = warp_baseline_to_test(fb, ft)
    base_words = get_word_timestamps(base_path_hint, transcript, fb["duration"]) \
        if base_path_hint else _uniform(transcript, fb["duration"])
    flaws, debug = ground_flaws(fb, ft, warp, base_words, ft["pauses"], fb["pauses"])
    sb, st = summarize(fb), summarize(ft)
    warp["local_rate"] = debug["local_rate"]  # length-matched grid used by dashboard
    coverage = sum(f["end"] - f["start"] for f in flaws) / max(ft["duration"], 1e-6)
    scores = score_utterance(fb, ft, warp, sb, st, flaw_coverage=coverage)
    wpm_base = len(transcript.split()) / max(fb["duration"] / 60, 1e-6)
    wpm_test = len(transcript.split()) / max(ft["duration"] / 60, 1e-6)
    return {
        "transcript": transcript,
        "baseline": {"duration": fb["duration"], "summary": sb, "words": base_words,
                     "pauses": fb["pauses"], "f0_median": fb["f0_median"]},
        "test": {"duration": ft["duration"], "summary": st, "pauses": ft["pauses"],
                 "f0_median": ft["f0_median"]},
        "features": {"times": ft["times"][:len(debug["composite"])], "test": _slim_n(ft, len(debug["composite"])),
                     "warped_baseline": _slim_w_n(warp["warped"], len(debug["composite"])),
                     "local_rate": debug["local_rate"], "composite": debug["composite"],
                     "parts": {k: v for k, v in debug["parts"].items()}},
        "flaws": flaws,
        "scores": scores,
        "speech_rate": {"baseline_wpm": round(wpm_base, 1), "test_wpm": round(wpm_test, 1),
                        "tempo_ratio": round(ft["duration"] / max(fb["duration"], 1e-6), 3)},
        "dtw_cost": round(float(warp["cost"]), 4),
    }


def analyze_files(base_wav: str, test_wav: str, transcript: str) -> dict:
    yb, sr = load_mono(base_wav)
    yt, sr2 = load_mono(test_wav)
    assert sr == sr2
    return analyze_arrays(yb, yt, transcript, sr, base_wav, test_wav)


def _uniform(transcript, duration):
    from .alignment import uniform_word_timestamps
    return uniform_word_timestamps(transcript, duration)


def _slim(f: dict) -> dict:
    return {"f0_var": f["f0_var"], "f0_st_filled": f["f0_st_filled"],
            "loge_z": f["loge_z"], "flatness": f["flatness"]}


def _slim_n(f: dict, n: int) -> dict:
    return {"f0_var": np.asarray(f["f0_var"])[:n], "f0_st_filled": np.asarray(f["f0_st_filled"])[:n],
            "loge_z": np.asarray(f["loge_z"])[:n], "flatness": np.asarray(f["flatness"])[:n]}


def _slim_w_n(w: dict, n: int) -> dict:
    return {"f0_var": np.asarray(w["f0_var"])[:n], "f0_st_filled": np.asarray(w["f0_st_filled"])[:n],
            "loge_z": np.asarray(w["loge_z"])[:n], "flatness": np.asarray(w["flatness"])[:n]}


def _slim_w(w: dict) -> dict:
    return {"f0_var": w["f0_var"], "f0_st_filled": w["f0_st_filled"],
            "loge_z": w["loge_z"], "flatness": w["flatness"]}

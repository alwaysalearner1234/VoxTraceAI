"""Forced alignment: transcript words -> audio timestamps + cross-utterance DTW.

Two layers:
1. *Within* one recording: word timestamps. Uses faster-whisper/openai-whisper
   word timestamps when installed; otherwise deterministic uniform allocation
   weighted by word length (reproducible fallback, documented as such).
2. *Across* recordings (baseline vs participant, same text, different rate):
   DTW on MFCC footprints gives a warping path + local tempo (slope) used by
   temporal grounding. Implemented from scratch (no extra deps) with
   downsampling so it stays fast.
"""
import numpy as np
from .utils import words_of


def uniform_word_timestamps(transcript: str, duration: float,
                            nudge_pauses: list | None = None) -> list:
    """Deterministic fallback: split duration proportional to word length."""
    words = words_of(transcript)
    if not words or duration <= 0:
        return []
    weights = np.array([max(2, len(w.strip(".,!?;:'\""))) for w in words], dtype=float)
    weights /= weights.sum()
    bounds, t = [], 0.0
    for w, wt in zip(words, weights):
        d = float(wt * duration)
        bounds.append({"word": w, "start": round(t, 3), "end": round(min(t + d, duration), 3)})
        t += d
    bounds[-1]["end"] = round(duration, 3)
    return bounds


def whisper_word_timestamps(wav_path: str, transcript: str | None = None) -> list | None:
    """Try whisper word timestamps; return None if backends unavailable."""
    try:
        import whisper  # openai-whisper
        model = whisper.load_model("tiny")
        res = model.transcribe(wav_path, word_timestamps=True, verbose=False)
        out = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                out.append({"word": w.get("word", "").strip(),
                            "start": round(float(w.get("start", 0)), 3),
                            "end": round(float(w.get("end", 0)), 3)})
        return out or None
    except Exception:
        pass
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(wav_path, word_timestamps=True)
        out = []
        for seg in segments:
            for w in (seg.words or []):
                out.append({"word": w.word.strip(), "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3)})
        return out or None
    except Exception:
        return None


def get_word_timestamps(wav_path: str, transcript: str, duration: float) -> list:
    """Best-effort word timestamps (whisper if present else uniform)."""
    wt = whisper_word_timestamps(wav_path, transcript)
    if wt:
        return wt
    return uniform_word_timestamps(transcript, duration)


# ---------------- DTW across baseline/test ----------------

def _downsample_frames(X: np.ndarray, max_frames: int = 400) -> tuple[np.ndarray, int]:
    T = X.shape[1]
    if T <= max_frames:
        return X, 1
    step = int(np.ceil(T / max_frames))
    return X[:, ::step], step


def dtw_path(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """DTW (Euclidean, mean-variance normalized) on (D,Ta),(D,Tb) MFCCs.

    Returns (path [[i,j]...], normalized_cost). Path maps baseline frame i
    to test frame j.
    """
    A, step_a = _downsample_frames(np.asarray(A, float))
    B, step_b = _downsample_frames(np.asarray(B, float))
    mu, sd = A.mean(axis=1, keepdims=True), A.std(axis=1, keepdims=True) + 1e-9
    An, Bn = (A - mu) / sd, (B - mu) / sd  # normalize in baseline space (speaker-agnostic)
    na, nb = An.shape[1], Bn.shape[1]
    # frame distance matrix (chunked to bound memory)
    D = np.empty((na, nb), dtype=np.float64)
    CH = 128
    for i in range(0, na, CH):
        ie = min(na, i + CH)
        d = ((An[:, i:ie, None] - Bn[:, None, :]) ** 2).sum(axis=0)
        D[i:ie] = np.sqrt(d)
    C = np.full((na + 1, nb + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    # backtrack
    i, j, path = na, nb, []
    while i > 1 or j > 1:
        path.append((i - 1, j - 1))
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            prev = [(C[i - 1, j - 1], -1, -1), (C[i - 1, j], -1, 0), (C[i, j - 1], 0, -1)]
            _, di, dj = min(prev, key=lambda t: t[0])
            i, j = i + di, j + dj
    path.append((0, 0))
    path = np.array(path[::-1])
    path[:, 0] *= step_a
    path[:, 1] *= step_b
    cost = float(C[na, nb] / (na + nb))
    return path, cost


def warp_baseline_to_test(baseline: dict, test: dict):
    """Interpolate baseline frame features onto the TEST time grid via DTW.

    The warp is computed on full MFCCs (+deltas): empirically the cleanest
    alignment for near-identical pairs (delta-only warps were tried and
    produced noisier residuals on clean speech). Static timbral flaws remain
    visible because DTW follows phonetic sequence, not absolute timbre.
    Local tempo (slope) >1 means test is locally faster (rushed); <1 slower.
    Test-silence frames get rate 1.0 (tempo inside pauses is meaningless and
    DTW staircases there).
    """
    A, B = baseline["mfcc"], test["mfcc"]
    path, cost = dtw_path(A, B)
    nt = len(test["times"])
    nb = A.shape[1]
    # baseline-index as function of test-index (monotone): average over path,
    # rescaled from mfcc-frame units to test-time-grid units
    scale_b = nb / max(A.shape[1], 1)  # ==1; kept explicit for clarity
    b_of_t = np.zeros(nt)
    counts = np.zeros(nt)
    tb_frames = B.shape[1]
    for i, j in path:
        jj = int(round(j * (nt - 1) / max(tb_frames - 1, 1)))
        jj = int(np.clip(jj, 0, nt - 1))
        b_of_t[jj] += i * (nb - 1) / max(A.shape[1] - 1, 1) * scale_b
        counts[jj] += 1
    b_of_t = b_of_t / np.maximum(counts, 1)
    # fill gaps (unmapped test frames) by interpolation
    mapped = counts > 0
    if mapped.sum() >= 2:
        b_of_t = np.interp(np.arange(nt), np.where(mapped)[0], b_of_t[mapped])
    else:
        b_of_t = np.linspace(0, A.shape[1] - 1, nt)

    def _warp(series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, float)
        idx = np.clip(b_of_t, 0, len(series) - 1)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, len(series) - 1)
        fr = (idx - lo)
        return (1 - fr) * series[lo] + fr * series[hi]

    warped = {
        "f0_var": _warp(np.nan_to_num(baseline["f0_var"])),
        "f0_st_filled": _warp(np.nan_to_num(baseline["f0_st_filled"])),
        "loge_z": _warp(baseline["loge_z"]),
        "loge": _warp(baseline["loge"]),
        "flatness": _warp(baseline["flatness"]),
        "centroid": _warp(baseline["centroid"]),
    }
    # local tempo via Savitzky-Golay slope of b_of_t over ~1 s windows:
    # regression (not 2-point gradient) so DTW staircase quantization does not
    # fabricate rushed/slow bursts. Both series share the same frame hop, so
    # slope IS the local speed ratio: >1 = test faster (rushed), <1 = slower.
    from scipy.signal import savgol_filter
    win = min(nt - (1 - nt % 2), 31)  # ~1 s at 32 ms hop, odd, fits signal
    win = max(5, win if win % 2 == 1 else win - 1)
    try:
        slope = savgol_filter(b_of_t, window_length=win, polyorder=2,
                              deriv=1, delta=1.0)
    except Exception:
        slope = np.gradient(b_of_t)
    local_rate = np.clip(slope, 0.2, 5.0)
    # tempo inside test silence is meaningless: pin to 1.0 so pause zones
    # don't fabricate rushed/slow evidence (inserted pauses are still caught
    # via the energy channel).
    try:
        trms = np.asarray(test["rms"], float)
        if len(trms) >= len(local_rate):
            sil = trms[:len(local_rate)] < (float(np.median(trms)) * 0.15 + 1e-6)
            local_rate[sil] = 1.0
    except Exception:
        pass
    return {"warped": warped, "path": path, "cost": cost,
            "local_rate": local_rate, "b_of_t": b_of_t}

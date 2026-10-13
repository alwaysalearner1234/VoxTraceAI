"""Advanced feature extraction: FFT/STFT, MFCCs, F0, energy, clarity, pauses.

Speaker-agnostic strategy: absolute pitch/energy are NEVER compared directly.
We compare (a) pitch in semitones relative to each speaker's own median F0 and
(b) z-scored log-energy per utterance. This removes biological baseline
differences while preserving expressive *variation* (the thing judges score).
"""
import numpy as np
import librosa
from .utils import SR, moving_average, db

N_FFT = 2048
HOP = 512
N_MFCC = 13


def extract(y: np.ndarray, sr: int = SR) -> dict:
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    times = (np.arange(int(np.ceil(n / HOP))) * HOP / float(sr))

    # --- STFT / FFT magnitude ---
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann"))
    mag = S  # (freq, time)
    logmag = librosa.amplitude_to_db(S + 1e-10)

    # --- MFCCs (+deltas) : the acoustic footprint used for DTW ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_feat = np.vstack([mfcc, mfcc_delta])  # (26, Tm)
    T = len(times)
    # align all frame series to common length T (mfcc/stft/yin can differ by ±1 frame)
    if mfcc_feat.shape[1] < T:
        mfcc_feat = np.pad(mfcc_feat, ((0, 0), (0, T - mfcc_feat.shape[1])), mode="edge")
    else:
        mfcc_feat = mfcc_feat[:, :T]

    # --- F0 contour (YIN). Unvoiced frames -> NaN ---
    fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C6")
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP,
                     frame_length=N_FFT)
    # yin returns a value even for silence; mask low-energy frames as unvoiced
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms = rms[: len(f0)] if len(rms) >= len(f0) else np.pad(rms, (0, len(f0) - len(rms)))
    voiced = rms > (np.median(rms) * 0.15 + 1e-6)
    f0_masked = np.where(voiced, f0, np.nan)

    # speaker-agnostic pitch: semitones above/below speaker median
    med = float(np.nanmedian(f0_masked)) if np.any(voiced) else 120.0
    if not np.isfinite(med) or med <= 0:
        med = 120.0
    f0_st = 12.0 * np.log2(np.maximum(f0_masked, 1e-6) / med)  # NaN where unvoiced
    # deglitch: median-filter (k=5, ~0.16 s) within each voiced run. Kills YIN
    # octave halving/doubling spikes that would otherwise inflate variability
    # differently in two recordings of the same delivery (a major false-flaw
    # source). Slow expressive arcs (the scored signal) pass through untouched.
    f0_st = _voiced_medfilt(f0_st, voiced, k=5)
    f0_st_filled = np.where(np.isnan(f0_st), 0.0, f0_st)

    # rolling pitch variability (expressiveness): std over ~0.5 s window
    win = max(3, int(round(0.5 * sr / HOP)))
    f0_var = _rolling_nanstd(f0_st, win)

    # --- Energy (log RMS) + speaker-agnostic z-score ---
    loge = db(rms[: len(times)] if len(rms) >= len(times) else np.pad(rms, (0, len(times) - len(rms))))
    loge_z = (loge - np.mean(loge)) / (np.std(loge) + 1e-9)
    loge_smooth = moving_average(loge, 5)

    # --- Vocal clarity proxies ---
    flatness = librosa.feature.spectral_flatness(S=S + 1e-10)[0]  # 0 tonal .. 1 noisy
    if len(flatness) < len(times):
        flatness = np.pad(flatness, (0, len(times) - len(flatness)), mode="edge")
    else:
        flatness = flatness[: len(times)]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    if len(zcr) < len(times):
        zcr = np.pad(zcr, (0, len(times) - len(zcr)), mode="edge")
    else:
        zcr = zcr[: len(times)]
    centroid = librosa.feature.spectral_centroid(S=S + 1e-10, sr=sr)[0]
    if len(centroid) < len(times):
        centroid = np.pad(centroid, (0, len(times) - len(centroid)), mode="edge")
    else:
        centroid = centroid[: len(times)]

    # --- Pause intervals (non-speech) ---
    pauses = detect_pauses(y, sr)

    T = len(times)
    def _fit(a):
        a = np.asarray(a).reshape(-1)
        if len(a) < T:
            a = np.pad(a, (0, T - len(a)), mode="edge")
        return a[:T].astype(np.float64)

    return {
        "sr": sr, "times": times.astype(np.float64),
        "mfcc": mfcc_feat.astype(np.float64),       # (26, T) DTW footprint
        "f0_hz": _fit(f0_masked), "f0_st": _fit(f0_st),
        "f0_st_filled": _fit(f0_st_filled), "f0_var": _fit(f0_var),
        "f0_median": float(med),
        "rms": _fit(rms), "loge": _fit(loge), "loge_z": _fit(loge_z),
        "loge_smooth": _fit(loge_smooth),
        "flatness": _fit(flatness), "zcr": _fit(zcr), "centroid": _fit(centroid),
        "logmag": logmag.astype(np.float64), "pauses": pauses,
        "duration": float(n / sr),
    }


def detect_pauses(y: np.ndarray, sr: int = SR, top_db: float = 28.0,
                  min_pause_s: float = 0.25) -> list:
    """Return [(start_s, end_s)] silence regions via energy gating."""
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=N_FFT, hop_length=HOP)
    pauses = []
    prev_end = 0
    for s, e in intervals:
        s_s, e_s = s / sr, e / sr
        if s_s - prev_end >= min_pause_s:
            pauses.append((round(float(prev_end), 3), round(float(s_s), 3)))
        prev_end = e_s
    total = len(y) / sr
    if total - prev_end >= min_pause_s:
        pauses.append((round(float(prev_end), 3), round(float(total), 3)))
    return pauses


def _voiced_medfilt(x: np.ndarray, voiced: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter within each maximal voiced run; NaN stays NaN."""
    from scipy.signal import medfilt
    x = np.asarray(x, dtype=float)
    out = x.copy()
    n = len(x)
    i = 0
    while i < n:
        if not voiced[i] or np.isnan(x[i]):
            i += 1
            continue
        j = i
        while j < n and voiced[j] and not np.isnan(x[j]):
            j += 1
        run = x[i:j]
        if len(run) >= k:
            out[i:j] = medfilt(run, kernel_size=k)
        i = j
    return out


def _rolling_nanstd(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - win // 2), min(len(x), i + win // 2 + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        out[i] = float(np.std(w)) if len(w) >= 3 else 0.0
    return out


def summarize(feats: dict) -> dict:
    """Utterance-level stats used for causal explanations + scoring."""
    v = feats["f0_var"]
    voiced_st = feats["f0_st"][~np.isnan(feats["f0_st"])]
    return {
        "duration": feats["duration"],
        "f0_median": feats["f0_median"],
        "f0_std_st": float(np.nanstd(feats["f0_st"])) if len(voiced_st) else 0.0,
        "f0_range_st": float(np.nanmax(feats["f0_st"]) - np.nanmin(feats["f0_st"])) if len(voiced_st) else 0.0,
        "mean_var": float(np.mean(v)),
        "loge_mean": float(np.mean(feats["loge"])),
        "loge_std": float(np.std(feats["loge"])),
        "flatness_mean": float(np.mean(feats["flatness"])),
        "pause_total": float(sum(e - s for s, e in feats["pauses"])),
        "pause_count": int(len(feats["pauses"])),
        "pause_ratio": float(sum(e - s for s, e in feats["pauses"]) / max(feats["duration"], 1e-6)),
    }

"""Contrastive bad-mirror synthesis: DSP flaw injections (deterministic).

Gradient (same transcript, worsening delivery):
  L0 good          — expressive TTS baseline (edge-tts, en-US-AriaNeural)
  L1 near-perfect  — +6% tempo drift, artifact-free resample (hard to catch)
  L2-rushed        — +28% global tempo (phase-vocoder time stretch)
  L2-slow          — -20% global tempo, resampled (audibly dragged)
  L2-pause         — one 1.6 s inserted mid-utterance silence (exact bounds known)
  L2-mumble        — low-pass + broadband noise (dull articulation)
  L3 egregious     — combined: fast + muffled + inserted pause

Design note: only transforms with *proven* acoustic impact under
speaker-agnostic (z-scored) features are used. Pure global gain changes and
naive envelope flattening are invisible after per-utterance normalization and
were deliberately excluded (see docs/TECHNICAL_DOC.md §2).

All transforms use fixed seeds -> byte-reproducible given the same baseline.
Labels (start/end flaw bounds) are known *by construction*.
"""
import numpy as np
import librosa
import scipy.signal as sig
from .utils import SR, get_rng


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=float(rate))


def respeed(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Artifact-free tempo change (tape-style): rate > 1 = faster/shorter with
    pitch raised by 12*log2(rate) semitones (+1st at 1.06, clearly tense at
    1.28); rate < 1 = slower/longer with lowered pitch. Implemented by
    resampling to sr/rate and reinterpreting at sr (same sample count <-> time
    duality). Used for the near-perfect L1 mirror so it stays clean."""
    y = np.asarray(y, dtype=np.float32)
    fast = librosa.resample(y, orig_sr=sr, target_sr=int(round(sr / float(rate))))
    return np.asarray(fast, dtype=np.float32)


def mumble(y: np.ndarray, sr: int = SR, cutoff: float = 1200.0,
           noise_db: float = -28.0, seed: int = 2026) -> np.ndarray:
    """Low-pass (muffled) + faint broadband noise (unclear articulation)."""
    sos = sig.butter(6, cutoff, btype="lowpass", fs=sr, output="sos")
    y_lp = sig.sosfilt(sos, np.asarray(y, dtype=np.float64)).astype(np.float32)
    rng = get_rng(seed)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    noise *= 10.0 ** (noise_db / 20.0) / (np.std(noise) + 1e-9) * 0.05
    return (y_lp + noise).astype(np.float32)


def insert_pause(y: np.ndarray, sr: int = SR, at_frac: float = 0.45,
                 pause_s: float = 1.4) -> tuple[np.ndarray, dict]:
    """Insert digital silence at fraction of utterance; returns (audio, label)."""
    y = np.asarray(y, dtype=np.float32)
    at = int(len(y) * at_frac)
    n = int(pause_s * sr)
    y2 = np.concatenate([y[:at], np.zeros(n, dtype=np.float32), y[at:]])
    label = {"start": round(at / sr, 2), "end": round((at + n) / sr, 2), "type": "inserted_pause"}
    return y2.astype(np.float32), label


def apply_local_rate(y: np.ndarray, sr: int, start_s: float, end_s: float,
                     rate: float) -> np.ndarray:
    """Time-stretch only [start_s, end_s]; keep rest untouched (localized flaw)."""
    s, e = int(start_s * sr), int(end_s * sr)
    seg = librosa.effects.time_stretch(y[s:e].astype(np.float32), rate=float(rate))
    return np.concatenate([y[:s], seg, y[e:]]).astype(np.float32)


def mirror_spectrum(y: np.ndarray, sr: int = SR, level: str = "L1") -> tuple[np.ndarray, list]:
    """Apply a named gradient level. Returns (audio, injected_labels)."""
    labels = []
    y = np.asarray(y, dtype=np.float32)
    if level == "L1":  # near-perfect: barely-off pacing, artifact-free
        y = respeed(y, sr, 1.06)
        labels.append({"type": "tempo_drift", "detail": "+6% rate (resampled, clean)", "severity": "subtle"})
    elif level == "L2-rushed":
        y = time_stretch(y, 1.28)
        labels.append({"type": "rushed", "detail": "+28% rate globally", "severity": "moderate"})
    elif level == "L2-slow":
        y = respeed(y, sr, 0.78)
        labels.append({"type": "dragged", "detail": "-22% rate globally (resampled, pitch -4.3st)",
                       "severity": "moderate"})
    elif level == "L2-pause":
        y, lab = insert_pause(y, sr, 0.5, 1.6)
        labels.append({**lab, "severity": "moderate"})
    elif level == "L2-mumble":
        y = mumble(y, sr, 1400.0, -26.0)
        labels.append({"type": "mumble", "detail": "LP 1.4kHz + noise", "severity": "moderate"})
    elif level == "L3":  # egregious combo (only effects with proven acoustic impact)
        y = time_stretch(y, 1.32)
        y = mumble(y, sr, 1000.0, -24.0)
        y, lab = insert_pause(y, sr, 0.35, 1.2)
        labels.append({"type": "combined_egregious",
                       "detail": "fast+muffled+pause", "severity": "egregious",
                       "start": lab["start"], "end": lab["end"]})
    else:
        raise ValueError(f"unknown level {level}")
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0.99:
        y = (y / peak * 0.99).astype(np.float32)
    return y, labels

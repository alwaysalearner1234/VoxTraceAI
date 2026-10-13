"""Shared audio I/O + math helpers. Reproducible (fixed seeds)."""
import os
import numpy as np
import soundfile as sf
import librosa

SR = 16000  # canonical sample rate for the whole project
RNG_SEED = 2026


def get_rng(seed: int = RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_mono(path: str, sr: int = SR):
    """Load any audio file as mono float32 at `sr`."""
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = y.astype(np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    # peak-normalize guard against clipping differences (keeps energy comparisons relative)
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak * 0.99
    return y, sr


def save_wav(path: str, y: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)


def db(x: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, floor))


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-9)
    return (x - mu) / sd


def time_grid(n_frames: int, hop: int, sr: int = SR) -> np.ndarray:
    return (np.arange(n_frames) * hop) / float(sr)


def words_of(transcript: str):
    return [w for w in transcript.strip().split() if w.strip()]

# VoxTraceAI — Technical Documentation
*Track C: Contrastive Speech Analytics & Temporal Flaw Grounding (condensed, ≤6 pages)*

## 1. Problem & approach

Judging spoken performance (declamation, oratory, interpretive reading) is subjective:
judges conflate *what* was said with *how* it was delivered. VoxTraceAI fixes the text
and measures delivery contrastively: the participant reads a known transcript; the
system compares their acoustics against an expressive baseline of the **same words**,
grounds deviations to timestamps, explains each flaw mathematically, and scores a fixed
rubric. All comparisons are **speaker-agnostic** (pitch in semitones above the speaker's
own median F0; energy z-scored per utterance), so biology never masquerades as skill.

## 2. Custom contrastive dataset

**Transcripts** (`data/transcripts/`): Gettysburg Address excerpt (71 words, public
domain), "I Have a Dream" excerpt (74 words, short fair-use excerpt for research demo),
and an original persuasive speech on public speaking (73 words).

**Good baselines** (`data/baseline/*__good.wav`): synthesized with `edge-tts`
(`en-US-AriaNeural`, 16 kHz mono) — a reproducible stand-in for champion delivery with
natural prosody. Provenance + durations recorded in `data/labels/*.json`. Fully offline
rebuilds use a deterministic prosodic synthesizer (`--offline`).

**Bad-mirror gradient** (`voxtrace/dataset.py`, all seeded/deterministic):

| level | transform | rationale |
|---|---|---|
| L1 | +6% rate via artifact-free resample | near-perfect; tests false-positive rate |
| L2-rushed | +28% phase-vocoder stretch | compressed pacing |
| L2-slow | −22% resample (pitch −4.3 st, tape-style) | dragged delivery |
| L2-pause | +1.6 s digital silence mid-utterance | exact bounds known by construction |
| L2-mumble | 1.4 kHz low-pass + broadband noise | dull articulation |
| L3 | fast + muffled + inserted pause | egregious combo |

**Deliberately excluded** (verified invisible, documented honestly): global gain drops
and naive envelope flattening — per-utterance normalization erases them, so they cannot
be scored fairly. Two attempted monotone syntheses (windowed pitch correction) were
abandoned after measurement showed they raised F0 variance instead of collapsing it.

## 3. Feature extraction (`voxtrace/features.py`, 16 kHz, n_fft 2048, hop 512)

STFT magnitude; 13 MFCCs + deltas (26-D DTW footprint); YIN F0 (C2–C6) with
silence-masking and **voiced-run median deglitching** (kills octave jumps that otherwise
become false flaws); pitch as semitones re speaker median; rolling F0 variability
(0.5 s std = expressiveness); log-RMS energy + z-score; spectral flatness / ZCR /
centroid (clarity); energy-gated pause intervals (≥0.25 s). Utterance summaries feed
scoring and rationales.

## 4. Forced alignment (`voxtrace/alignment.py`)

Word timestamps: faster-/openai-whisper when installed, else deterministic
length-weighted uniform allocation. Cross-recording warp: DTW (from scratch, Euclidean
on mean/variance-normalized MFCCs, downsampled to ≤400 frames) maps baseline frames to
the test grid; baseline curves are interpolated onto test time. **Local tempo** is the
Savitzky–Golay slope of the warp over ~1 s windows (regression, not gradient — DTW
staircases otherwise fabricate rushed/slow bursts); slope = speed ratio (>1 faster).
Test-silence frames are pinned to 1.0. (Delta-only DTW was tried; full-MFCC warps were
cleaner on near-identical pairs.)

## 5. Temporal grounding & causal explainability (`voxtrace/grounding.py`)

Per-frame deviations: variety loss (relative F0-var drop), |Δ energy-z|, local-tempo
deviation from the utterance median (global drift excluded), |Δ flatness|
(bidirectional: too noisy *or* too dull). Composite = **max over weighted channels**
(a mean dilutes pure single-channel flaws below threshold). Hysteresis (HI 0.9 / LO 0.5,
tunable in UI), ≥0.5 s duration, 0.5 s merge, 0.3 s DTW edge trim. Each region is
classified (Rushed / Dragged / Monotone / Mumbled / Awkward Pause / Flat Projection)
with confidence, quoted words, numeric stats, a **math rationale** (Δvar %, Δenergy σ,
local rate ×, Δflatness, pause-in-region vs baseline), and actionable advice. A
**pervasive-tempo overlay** fires when median rate is ≥15% off (global pace flaws get an
explicit region; contradictory local tempo labels are reconciled to the robust median).

## 6. Rubric scoring (`voxtrace/scoring.py`, deterministic)

Pace 25 (trimmed-mean |log rate|), Variety 25 (pitch-variability ratio), Pauses 20
(pause-ratio + count drift), Clarity 30 (flatness delta + energy-variance loss +
warped-projection mismatch) — each `100·exp(−k·d)` — minus a **flaw-coverage penalty**
(≤30 pts) tying grounding to score. Grades A–F at 85/70/55/40.

## 7. Dashboard (`app.py`)

Streamlit: pick a built-in speech/mirror or upload baseline + transcript + participant
audio; tune HI/LO/min-duration; time-series overlays (pitch-variability, energy,
clarity, local tempo, composite anomaly) with flaw spans; flaw cards with rationales;
audio playback; JSON export. Verified headless (`AppTest`, no exceptions).

## 8. Evaluation (measured)

Stress test (`scripts/demo_stress_test.py`): self-comparison **100.0 / 0 flaws** (all
3); L1 **92–96 / 0–1 flaws**; L2s 43–78; L3 41–55; mean L1 93.4 vs L3 48.3 (PASS).
Grounding audit (`scripts/evaluate_grounding.py`): pause-flaw IoU **0.70–0.72** vs
injected bounds; flaw-label agreement **18/18**; every L2 produces ≥1 flaw, L1 ~none.
Unit tests (`tests/`, 5 passed): determinism, self-score 100, gradient direction,
pause grounding overlap, respeed semantics.

## 9. Reproducibility

Fixed seeds everywhere; pinned `requirements.txt`; one-command dataset rebuild;
whisper-optional (fallback path identical given same inputs); OS-independent (bundled
`imageio-ffmpeg` for mp3 decode, no system ffmpeg). Known limits: whisper-grade word
alignment needs the optional dependency; heavy monotone resynthesis was cut as
unmeasurable; TTS baselines stand in for licensed champion recordings (swap any wav in
as baseline — the pipeline is reference-agnostic).

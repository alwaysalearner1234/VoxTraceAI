# VoxTraceAI — Contrastive Speech Analytics & Temporal Flaw Grounding

Track C · Multimodal AI Hackathon 2026. A speech-evaluation system that contrasts a
participant's delivery of a text against an "ideal" baseline of the **same text**,
grounds delivery flaws to exact timestamps, explains each flaw with its mathematical
rationale, and scores delivery on a reproducible rubric — all behind an interactive dashboard.

## Results (measured, reproducible)

3 speeches × 6-mirror contrastive gradient (`python scripts/demo_stress_test.py`):

| speech | L1 near-perfect | L2-rushed | L2-slow | L2-pause | L2-mumble | L3 egregious |
|---|---|---|---|---|---|---|
| dream_excerpt | **92.4** (0 flaws) | 59.3 | 73.8 | 70.0 | 64.5 | 54.5 |
| gettysburg_excerpt | **95.8** (0) | 58.5 | 68.5 | 69.6 | 69.1 | 48.9 |
| persuasive_original | **92.1** (1) | 58.1 | 70.7 | 77.9 | 43.0 | 41.5 |

- Self-comparison (baseline vs itself): **100.0, 0 flaws** on all 3 speeches.
- Pause-flaw grounding IoU vs injected bounds: **0.70–0.72**; label agreement **18/18**.
- Full numbers: `data/labels/stress_test_results.csv`; grounding audit: `scripts/evaluate_grounding.py`.

## Quickstart

```bash
pip install -r requirements.txt

# 1. (Re)build the contrastive dataset — TTS baselines + 6 DSP mirrors each
python scripts/generate_dataset.py
#    offline fallback (no network): python scripts/generate_dataset.py --offline

# 2. Stress test: score every mirror vs its baseline
python scripts/demo_stress_test.py

# 3. Grounding accuracy audit (IoU vs known bounds, label agreement)
python scripts/evaluate_grounding.py

# 4. Interactive dashboard
streamlit run app.py

# 5. Unit tests
python -m pytest tests/ -q
```

## How it works

1. **Contrastive dataset** (`scripts/generate_dataset.py`, `voxtrace/dataset.py`):
   expressive `edge-tts` baselines of 3 transcripts (2 public-domain excerpts +
   1 original) mirrored into a badness gradient — L1 tempo drift, L2 rushed / slow /
   inserted pause / mumble, L3 combined. Every mirror carries machine-written temporal
   labels in `data/labels/*.json`.
2. **Features** (`voxtrace/features.py`): STFT/FFT, 13 MFCCs + deltas, YIN F0 with
   voiced-run median deglitching, speaker-agnostic semitone pitch (relative to own
   median) and z-scored log-energy, spectral flatness / clarity, pause intervals.
3. **Alignment** (`voxtrace/alignment.py`): whisper word timestamps when installed
   (uniform fallback otherwise) + full-MFCC DTW warp with Savitzky–Golay local-tempo
   (slope = speed ratio; silence-masked).
4. **Grounding + explainability** (`voxtrace/grounding.py`): max-channel composite
   anomaly over variety / energy / local-tempo / clarity deviations, hysteresis
   regions (≥0.5 s), per-region causal math rationale + fix advice, pervasive-tempo
   overlay for globally fast/slow deliveries.
5. **Scoring** (`voxtrace/scoring.py`): Pace 25 / Variety 25 / Pauses 20 /
   Clarity 30 → total 0–100 + grade, minus flaw-coverage penalty. Deterministic.
6. **Dashboard** (`app.py`): upload baseline + transcript + participant audio (or pick
   built-in mirrors), tune sensitivity, inspect F0/energy/clarity/tempo overlays with
   flaw spans, read causal explanations, play audio, download JSON report.

## Repo layout

```
app.py                    # Streamlit dashboard
voxtrace/                 # features | alignment | grounding | scoring | pipeline | dataset | utils
scripts/                  # generate_dataset | demo_stress_test | evaluate_grounding | calibrate | check_labels
data/transcripts/         # 3 transcripts   data/baseline/  # good TTS references
data/mirrors/             # 18 bad-mirror wavs   data/labels/  # JSON labels + results CSV
tests/  docs/TECHNICAL_DOC.md
```

See `docs/TECHNICAL_DOC.md` for dataset construction, model architecture, rubric math,
and evaluation. Dataset audio + labels ship in `data/` (also reproducible from scratch
via one command above).

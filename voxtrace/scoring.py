"""Reproducible rubric scoring (0–100 per dimension, deterministic).

Dimensions (weights sum to 100):
  Pace & Rhythm ......... 25   (local-rate deviation from 1.0)
  Vocal Variety ......... 25   (pitch-variability ratio vs baseline)
  Pauses & Phrasing ..... 20   (pause-ratio + count deviation)
  Clarity & Projection .. 30   (flatness delta + energy-variance loss)

Each maps a deviation d>=0 to score via 100*exp(-k*d). Constants fixed below.
Total = weighted sum. Grade bands: A>=85, B>=70, C>=55, D>=40, else F.
"""
import numpy as np

W = {"pace": 25, "variety": 25, "pauses": 20, "clarity": 30}


def score_utterance(baseline: dict, test: dict, warp: dict,
                    base_sum: dict, test_sum: dict,
                    flaw_coverage: float = 0.0) -> dict:
    lr = np.clip(warp["local_rate"], 0.2, 5.0)
    # trimmed mean of |log rate|: robust to localized DTW staircase artifacts
    # (uniform rushed/slow delivery shifts every frame, so trimming keeps it).
    a = np.abs(np.log(lr))
    lo_p, hi_p = np.percentile(a, [5, 95])
    core = a[(a >= lo_p) & (a <= hi_p)] if len(a) else a
    d_pace = float(np.mean(core)) if len(core) else 0.0                 # 0 perfect
    pace = 100.0 * np.exp(-2.2 * d_pace)

    bv = max(base_sum["mean_var"], 1e-6)
    ratio = float(np.clip(test_sum["mean_var"] / bv, 0.0, 1.5))
    d_var = max(0.0, 1.0 - ratio)                                    # loss of variety
    variety = 100.0 * np.exp(-2.5 * d_var)

    d_pause = abs(test_sum["pause_ratio"] - base_sum["pause_ratio"]) * 4.0 \
        + abs(test_sum["pause_count"] - base_sum["pause_count"]) * 0.06
    pauses = 100.0 * np.exp(-1.6 * d_pause)

    d_clar = max(0.0, test_sum["flatness_mean"] - base_sum["flatness_mean"]) * 6.0
    base_e = max(base_sum["loge_std"], 1e-6)
    e_ratio = float(np.clip(test_sum["loge_std"] / base_e, 0.0, 1.5))
    e_loss = max(0.0, 1.0 - e_ratio)                                 # dynamics flattening
    # per-frame projection mismatch (warped): catches flattened dynamics even
    # when utterance stds look similar
    try:
        pe = float(np.mean(np.abs(np.asarray(warp["warped"]["loge_z"], float)
                                  - np.asarray(test["loge_z"], float)[:len(warp["warped"]["loge_z"])])))
    except Exception:
        pe = 0.0
    d_proj = min(pe / 1.2, 2.0)
    clarity = 100.0 * np.exp(-(1.8 * d_clar + 2.2 * e_loss + 1.1 * d_proj))

    scores = {"pace": round(float(pace), 1), "variety": round(float(variety), 1),
              "pauses": round(float(pauses), 1), "clarity": round(float(clarity), 1)}
    total = round(sum(scores[k] * W[k] for k in scores) / 100.0, 1)
    # flaw-coverage penalty: grounded flaw time directly lowers the total
    # (max -30 pts at 100% coverage). Ties temporal grounding to the score so
    # local flaws (e.g. a dull 30% passage) move the total decisively.
    coverage = float(np.clip(flaw_coverage, 0.0, 1.0))
    total = round(max(0.0, total - 30.0 * coverage), 1)
    grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 \
        else "D" if total >= 40 else "F"
    return {"dimensions": scores, "weights": dict(W), "total": total, "grade": grade,
            "coverage_penalty": round(30.0 * coverage, 1),
            "deviations": {"d_pace": round(d_pace, 4), "variety_ratio": round(ratio, 3),
                           "d_pause": round(float(d_pause), 4), "d_clar": round(float(d_clar), 4),
                           "e_ratio": round(float(e_ratio), 3), "d_proj": round(float(d_proj), 4),
                           "dtw_cost": round(float(warp["cost"]), 4)}}

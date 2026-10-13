"""Temporal grounding + causal explainability.

Pipeline (all speaker-agnostic, all deterministic):
1. Warp baseline features onto test grid (DTW on MFCCs).
2. Per-frame deviation signals:
     d_var    = baseline pitch-variability  - test pitch-variability  (monotone if +)
     d_energy = |baseline loge_z - test loge_z|                        (projection)
     d_rate   = |log(local_rate)|                                      (pacing)
     d_clar   = test flatness - baseline flatness                      (mumble/noise)
3. Composite anomaly = weighted z-scored sum. Hysteresis thresholding ->
   flaw regions (start/end). Merge gaps <0.3 s, drop regions <0.4 s.
4. Classify each region by dominant signal + attach causal math rationale.
"""
import numpy as np
from .utils import zscore, moving_average

WEIGHTS = {"var": 1.0, "energy": 0.9, "rate": 1.1, "clar": 0.8}
HI, LO = 0.90, 0.50          # hysteresis on max-channel composite
MIN_DUR, MERGE_GAP = 0.50, 0.50
EDGE_TRIM_S = 0.30           # ignore DTW edge transients at both ends
SMOOTH_WIN = 13              # ~0.42 s MA: tall-narrow noise spikes die, wide flaws survive
# fixed physical scales: a deviation of `scale` on a channel contributes ~1.0
SCALES = {"var": 0.6, "energy": 1.0, "rate": 0.22, "clar": 0.15}


def composite_anomaly(warped: dict, test: dict, local_rate: np.ndarray) -> tuple[np.ndarray, dict]:
    bv = np.nan_to_num(warped["f0_var"])
    tv = np.nan_to_num(test["f0_var"])
    n = min(len(bv), len(tv), len(test["flatness"]), len(warped["loge_z"]),
            len(test["loge_z"]), len(local_rate))
    bv, tv = bv[:n], tv[:n]
    d_var = np.maximum(bv - tv, 0.0) / (np.mean(bv) + 0.35)   # relative loss of variety
    d_energy = np.abs(warped["loge_z"][:n] - test["loge_z"][:n])
    # LOCAL pacing only: subtract the utterance median log-rate so global
    # tempo drift (L1's +6%, or mic-distance gain-like effects) does not
    # fabricate evidence. Global pace is judged by the rubric (pace dim) and
    # the median-rate gate in ground_flaws fallback.
    lr = np.clip(local_rate[:n], 1e-3, 1e3)
    d_rate = np.abs(np.log(lr) - np.median(np.log(lr)))
    # clarity deviation is bidirectional: too noisy (mumble) OR too dull (flat)
    d_clar = np.abs(test["flatness"][:n] - warped["flatness"][:n])

    def _abs_term(d, scale):
        return moving_average(np.asarray(d, float) / scale, SMOOTH_WIN)

    terms = {"d_var": _abs_term(d_var, SCALES["var"]),
             "d_energy": _abs_term(d_energy, SCALES["energy"]),
             "d_rate": _abs_term(d_rate, SCALES["rate"]),
             "d_clar": _abs_term(d_clar, SCALES["clar"])}
    # MAX over channels (weighted): any single strong deviation is a flaw.
    # (A mean would dilute a pure single-channel flaw below threshold.)
    wmax = max(WEIGHTS.values())
    stacked = np.vstack([(WEIGHTS[k[2:]] / wmax) * terms[k] for k in terms])
    comp = stacked.max(axis=0)
    parts = {k: v for k, v in terms.items()}
    return comp, parts


def threshold_regions(score: np.ndarray, times: np.ndarray,
                      hi: float = HI, lo: float = LO,
                      min_dur: float = MIN_DUR, merge_gap: float = MERGE_GAP) -> list:
    """Hysteresis: turn on at hi, stay on until below lo."""
    on, start, regs = False, 0.0, []
    for k, (s, t) in enumerate(zip(score, times)):
        if not on and s >= hi:
            on, start = True, float(t)
        elif on and s < lo:
            on = False
            regs.append((start, float(t)))
    if on:
        regs.append((start, float(times[-1])))
    # merge close gaps
    merged = []
    for s, e in regs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_dur]


def _overlap(a0, a1, pauses) -> float:
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in pauses)


def classify_region(s, e, times, parts, warped, test, local_rate,
                    test_pauses, base_pauses, words) -> dict:
    m = (times >= s) & (times <= e)
    if m.sum() == 0:
        m = np.ones_like(times, bool)
    dom = max(parts, key=lambda k: float(np.mean(parts[k][m])))
    seg_rate = float(np.mean(local_rate[m]))
    bv, tv = float(np.mean(warped["f0_var"][m])), float(np.mean(test["f0_var"][m]))
    be, te = float(np.mean(warped["loge_z"][m])), float(np.mean(test["loge_z"][m]))
    bf, tf = float(np.mean(warped["flatness"][m])), float(np.mean(test["flatness"][m]))
    var_drop = (bv - tv) / (abs(bv) + 1e-6) * 100.0
    pause_in = _overlap(s, e, test_pauses)
    base_pause_in = _overlap(s, e, base_pauses)
    seg_words = [w["word"] for w in words if not (w["end"] < s or w["start"] > e)][:12]

    if pause_in >= 0.35 and pause_in > base_pause_in + 0.15:
        label, advice = ("Awkward Pause / Poor Phrasing",
                         "Break the pause into shorter breath groups at phrase boundaries; "
                         "keep pauses under ~0.6 s except at sentence ends.")
    elif seg_rate >= 1.28:
        label, advice = ("Rushed / Compressed Pacing",
                         "Slow down ~%.0f%% in this region; elongate vowels and add a "
                         "beat after stressed words." % min(60, (seg_rate - 1) * 100))
    elif seg_rate <= 0.78:
        label, advice = ("Dragged / Too Slow",
                         "Lift the pace ~%.0f%%; reduce inter-word gaps and lean into "
                         "the next stressed syllable." % min(60, (1 / max(seg_rate, 1e-3) - 1) * 100))
    elif dom == "d_var" or var_drop > 45:
        label, advice = ("Monotone / Flat Intonation",
                         "Add pitch contrast: lift stressed words +2–4 semitones and let "
                         "phrase ends fall; practice humming the pitch arc first.")
    elif dom == "d_clar" or (tf - bf) > 0.08:
        label, advice = ("Mumbled / Low Clarity",
                         "Project consonants (especially plosives/fricatives), open the jaw, "
                         "and raise high-frequency energy; check mic distance.")
    else:
        label, advice = ("Flat Projection / Low Energy Variance",
                         "Vary loudness deliberately: crescendo into key words, soften "
                         "asides; mark 2–3 stress peaks in the script.")
    conf = float(np.clip((np.mean(parts[dom][m]) / (np.mean(parts[dom]) + 1e-9)) / 4.0, 0.15, 0.99))

    rationale = (
        f"MATH RATIONALE — Δpitch-var={var_drop:+.0f}% (base {bv:.2f} vs test {tv:.2f} st-std), "
        f"Δenergy-z={te - be:+.2f}σ, local-rate={seg_rate:.2f}× "
        f"({'faster' if seg_rate > 1 else 'slower'} than baseline), "
        f"Δflatness={tf - bf:+.3f} (clarity proxy), "
        f"pause-in-region={pause_in:.2f}s vs baseline {base_pause_in:.2f}s. "
        f"Dominant signal: {dom}. Composite anomaly stayed above threshold for {e - s:.2f}s."
    )
    return {"start": round(float(s), 2), "end": round(float(e), 2),
            "label": label, "confidence": round(conf, 2),
            "words": " ".join(seg_words),
            "stats": {"var_drop_pct": round(float(var_drop), 1),
                      "local_rate": round(seg_rate, 2),
                      "d_energy": round(float(te - be), 3),
                      "d_flatness": round(float(tf - bf), 4),
                      "pause_in_region": round(float(pause_in), 2)},
            "rationale": rationale, "advice": advice}


def ground_flaws(baseline: dict, test: dict, warp: dict,
                 base_words: list, test_pauses: list, base_pauses: list) -> tuple[list, dict]:
    times = np.asarray(test["times"], float)
    comp, parts = composite_anomaly(warp["warped"], test, warp["local_rate"])
    n = min(len(times), len(comp))
    times, comp = times[:n], comp[:n]
    for k in list(parts.keys()):
        parts[k] = np.asarray(parts[k], float)[:n]
    lr = np.asarray(warp["local_rate"], float)[:n]
    # suppress DTW edge transients: pin composite to 0 within EDGE_TRIM_S of ends
    if len(times) > 2:
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.032
        kedge = int(EDGE_TRIM_S / max(dt, 1e-3))
        if kedge > 0 and len(comp) > 2 * kedge:
            comp[:kedge] = 0.0
            comp[-kedge:] = 0.0
    # align warped/test series used by classify_region to the same grid
    warped_n = {k: np.asarray(v, float)[:n] for k, v in warp["warped"].items()}
    test_n = {k: (np.asarray(test[k], float)[:n] if k in test else test[k]) for k in
              ("f0_var", "f0_st_filled", "loge_z", "loge", "flatness", "centroid")}
    regs = threshold_regions(comp, times)
    flaws = [classify_region(s, e, times, parts, warped_n, test_n,
                             lr, test_pauses, base_pauses, base_words)
             for s, e in regs]
    med_rate = float(np.median(lr)) if len(lr) else 1.0
    mean_comp = float(np.mean(comp)) if len(comp) else 0.0
    logmed = float(np.log(max(med_rate, 1e-3)))

    def _peak_window():
        pk = int(np.argmax(comp)) if len(comp) else 0
        pk_t = float(times[pk]) if len(times) else 0.0
        return (max(float(times[0]), pk_t - 3.0),
                min(float(times[-1]), pk_t + 3.0))

    def _force_tempo(fl, direction):
        if direction > 0:  # globally faster
            fl["label"] = "Rushed / Compressed Pacing"
            fl["advice"] = ("The whole delivery runs fast (~%.0f%% above baseline pace). "
                            "Slow down globally; elongate vowels and add beats after "
                            "stressed words." % min(60, (med_rate - 1) * 100))
        else:
            fl["label"] = "Dragged / Too Slow"
            fl["advice"] = ("The whole delivery drags (~%.0f%% below baseline pace). "
                            "Lift the pace globally; reduce inter-word gaps and lean "
                            "into the next stressed syllable." % min(60, (1 / max(med_rate, 1e-3) - 1) * 100))
        fl["stats"]["local_rate"] = round(med_rate, 2)
        return fl

    if not flaws:
        # Global fallback: utterance-wide deviation with no local peak still
        # deserves one grounded region (e.g. uniformly rushed delivery).
        # Window = 6 s centered on the composite peak (not the whole file, so
        # flaw-coverage stays honest for scoring).
        if mean_comp >= 0.60 or abs(logmed) >= 0.12:
            s0, e0 = _peak_window()
            flaws = [classify_region(s0, e0, times, parts, warped_n, test_n,
                                     lr, test_pauses, base_pauses, base_words)]
            flaws[0]["global_fallback"] = True
            flaws[0]["rationale"] += (" Global-fallback region: no single local peak "
                                      "crossed the high threshold, but the utterance-wide "
                                      f"mean anomaly {mean_comp:.2f} / median rate {med_rate:.2f}x "
                                      "indicates a pervasive delivery shift.")
            if abs(logmed) >= 0.12:
                flaws[0] = _force_tempo(flaws[0], 1 if logmed > 0 else -1)
    elif abs(logmed) >= 0.15:
        # Pervasive tempo overlay: the file as a whole runs fast/slow, so add
        # one explicitly global tempo region (local d_rate is median-subtracted
        # and would otherwise never say so). Contradictory local tempo labels
        # are reconciled toward the robust file-level median.
        s0, e0 = _peak_window()
        fl = classify_region(s0, e0, times, parts, warped_n, test_n,
                             lr, test_pauses, base_pauses, base_words)
        fl = _force_tempo(fl, 1 if logmed > 0 else -1)
        fl["pervasive_tempo"] = True
        fl["rationale"] += (f" Pervasive-tempo overlay: median local rate {med_rate:.2f}x "
                            f"(log {logmed:+.2f}) across the whole utterance.")
        flaws.insert(0, fl)
        for f in flaws[1:]:
            if logmed > 0 and f["label"].startswith("Dragged"):
                f["label"] = "Rushed / Compressed Pacing"
                f["rationale"] += (" [Reconciled: file-level median tempo is fast; "
                                   "local slope estimate deferred to it.]")
            elif logmed < 0 and f["label"].startswith("Rushed"):
                f["label"] = "Dragged / Too Slow"
                f["rationale"] += (" [Reconciled: file-level median tempo is slow; "
                                   "local slope estimate deferred to it.]")
    debug = {"composite": comp, "parts": parts, "local_rate": lr,
             "dtw_cost": warp["cost"]}
    return flaws, debug

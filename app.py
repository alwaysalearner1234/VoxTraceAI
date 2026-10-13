"""VoxTraceAI interactive dashboard (Streamlit).

Run:  streamlit run app.py
Features: baseline select OR upload baseline+transcript, upload test audio,
tune thresholds, time-series overlay (F0/energy/clarity/tempo), flaw table
with causal rationales, rubric scores, audio playback, CSV/JSON export.
"""
import io, json, os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import soundfile as sf

from voxtrace.utils import SR, load_mono
from voxtrace.pipeline import analyze_arrays, analyze_files
from voxtrace import grounding as G

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="VoxTraceAI — Contrastive Speech Analytics", layout="wide")
st.title("VoxTraceAI — Contrastive Speech Analytics & Temporal Flaw Grounding")
st.caption("Track C · speaker-agnostic · reproducible rubric scoring · DTW-grounded causal flaws")

# ---------- sidebar: inputs ----------
with st.sidebar:
    st.header("1 · Baseline")
    transcripts = sorted([f for f in os.listdir(os.path.join(ROOT, "data", "transcripts"))
                          if f.endswith(".txt")]) if os.path.exists(os.path.join(ROOT, "data", "transcripts")) else []
    labels = {}
    for jf in (os.listdir(os.path.join(ROOT, "data", "labels"))
               if os.path.exists(os.path.join(ROOT, "data", "labels")) else []):
        if jf.endswith(".json") and not jf.startswith("_"):
            with open(os.path.join(ROOT, "data", "labels", jf), encoding="utf-8") as f:
                labels[os.path.splitext(jf)[0]] = json.load(f)
    choice = st.selectbox("Speech", list(labels.keys()) or ["(upload custom below)"])
    base_wav_path = None
    transcript_text = ""
    if choice in labels:
        item = labels[choice]
        base_wav_path = os.path.join(ROOT, item["baseline"])
        with open(os.path.join(ROOT, "data", "transcripts", item["transcript"]), encoding="utf-8") as f:
            transcript_text = f.read().strip()
        mirrors = {m["level"]: m["file"] for m in item["mirrors"]}
        preset = st.selectbox("Or try a built-in bad mirror", ["(none — upload test audio)"] + list(mirrors))
    else:
        mirrors, preset = {}, "(none — upload test audio)"
    st.markdown("---")
    st.header("2 · Custom inputs (optional)")
    up_base = st.file_uploader("Baseline WAV/MP3/OGG", type=["wav", "mp3", "ogg", "flac"])
    up_trans = st.file_uploader("Transcript TXT", type=["txt"])
    if up_trans is not None:
        transcript_text = up_trans.read().decode("utf-8", errors="replace").strip()
    st.markdown("---")
    st.header("3 · Test audio")
    up_test = st.file_uploader("Participant recording", type=["wav", "mp3", "ogg", "flac"])
    st.markdown("---")
    st.header("4 · Sensitivity")
    hi = st.slider("Anomaly threshold (high)", 0.4, 3.0, 0.90, 0.05)
    lo = st.slider("Release threshold (low)", 0.2, 2.0, 0.50, 0.05)
    min_dur = st.slider("Min flaw duration (s)", 0.2, 1.5, 0.50, 0.05)

transcript_text = st.text_area("Transcript (editable)", transcript_text, height=110)

# resolve test audio bytes
test_bytes, test_name = None, None
if up_test is not None:
    test_bytes, test_name = up_test.read(), up_test.name
elif choice in labels and preset in mirrors:
    with open(os.path.join(ROOT, mirrors[preset]), "rb") as f:
        test_bytes, test_name = f.read(), mirrors[preset]

base_bytes = None
if up_base is not None:
    base_bytes = up_base.read()

def _decode(blob: bytes):
    y, sr = sf.read(io.BytesIO(blob), always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=-1)
    import librosa
    if sr != SR:
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=SR)
    return np.asarray(y, dtype=np.float32)

run = st.button("Analyze delivery", type="primary",
                disabled=not (transcript_text.strip() and (base_wav_path or base_bytes) and test_bytes))

if not run:
    st.info("Select a speech (or upload baseline + transcript), add participant audio, then **Analyze delivery**. "
            "Tip: pick a built-in bad mirror (e.g. `L3`) to see the dashboard catch deviations instantly.")
    st.stop()

# ---------- analysis ----------
G.HI, G.LO, G.MIN_DUR = float(hi), float(lo), float(min_dur)
try:
    if base_bytes is not None:
        yb = _decode(base_bytes)
        import tempfile
        report = analyze_arrays(yb, _decode(test_bytes), transcript_text)
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix="_test.wav", delete=False) as t:
            t.write(test_bytes); tpath = t.name
        report = analyze_files(base_wav_path, tpath, transcript_text)
        os.unlink(tpath)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# ---------- scores ----------
s = report["scores"]
c1, c2, c3, c4, c5 = st.columns(5)
for col, k, w in zip([c1, c2, c3, c4],
                     ["pace", "variety", "pauses", "clarity"],
                     [s["weights"]["pace"], s["weights"]["variety"], s["weights"]["pauses"], s["weights"]["clarity"]]):
    col.metric(f"{k.title()}  ({w}%)", s["dimensions"][k])
c5.metric("TOTAL", f'{s["total"]}  ({s["grade"]})')
st.caption(f'Speech rate — baseline {report["speech_rate"]["baseline_wpm"]} wpm vs test '
           f'{report["speech_rate"]["test_wpm"]} wpm · tempo ratio {report["speech_rate"]["tempo_ratio"]} · '
           f'DTW cost {report["dtw_cost"]}')

# ---------- overlay plots ----------
F = report["features"]
t = F["times"]
tb, tt = F["warped_baseline"], F["test"]

def overlay(title, base_series, test_series, ylab):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=base_series, name="baseline (warped)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=test_series, name="participant", line=dict(width=2)))
    for fl in report["flaws"]:
        fig.add_vrect(x0=fl["start"], x1=fl["end"], fillcolor="red", opacity=0.13,
                      line_width=0, annotation_text=fl["label"].split("/")[0].strip(),
                      annotation_position="top left", annotation_font_size=10)
    fig.update_layout(title=title, xaxis_title="time (s)", yaxis_title=ylab,
                      height=300, margin=dict(l=40, r=20, t=40, b=30),
                      legend=dict(orientation="h", y=1.12))
    return fig

tab1, tab2, tab3, tab4 = st.tabs(["Pitch variability", "Energy", "Clarity", "Tempo + anomaly"])
with tab1:
    st.plotly_chart(overlay("Pitch variability — rolling F0 std (semitones). Drops = monotone",
                            tb["f0_var"], tt["f0_var"], "st-std"), use_container_width=True)
with tab2:
    st.plotly_chart(overlay("Projection — z-scored log-energy (speaker-normalized)",
                            tb["loge_z"], tt["loge_z"], "σ"), use_container_width=True)
with tab3:
    st.plotly_chart(overlay("Clarity — spectral flatness (higher = noisier/muffled)",
                            tb["flatness"], tt["flatness"], "flatness"), use_container_width=True)
with tab4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=F["local_rate"], name="local rate (1 = same pace)"))
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="baseline pace")
    for fl in report["flaws"]:
        fig.add_vrect(x0=fl["start"], x1=fl["end"], fillcolor="red", opacity=0.13, line_width=0)
    fig.update_layout(title="Local tempo from DTW slope", xaxis_title="time (s)",
                      height=220, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=F["composite"], name="composite anomaly",
                              fill="tozeroy", line=dict(color="firebrick")))
    fig2.add_hline(y=hi, line_dash="dash", annotation_text="high threshold")
    for fl in report["flaws"]:
        fig2.add_vrect(x0=fl["start"], x1=fl["end"], fillcolor="red", opacity=0.13, line_width=0)
    fig2.update_layout(title="Composite anomaly score (weighted deviation)", xaxis_title="time (s)",
                       height=220, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- flaws ----------
st.subheader(f"Flaw regions ({len(report['flaws'])}) — temporally grounded + causal")
if not report["flaws"]:
    st.success("No flaw regions above threshold — delivery matches baseline expression.")
for i, fl in enumerate(report["flaws"], 1):
    with st.expander(f'#{i} · {fl["start"]:.2f}–{fl["end"]:.2f}s · {fl["label"]}  (conf {fl["confidence"]})', expanded=(i == 1)):
        st.write(f'**Words:** {fl["words"] or "—"}')
        st.write(f'**Why (math):** {fl["rationale"]}')
        st.write(f'**Fix:** {fl["advice"]}')
        st.json(fl["stats"])

# ---------- audio + export ----------
st.subheader("Audio")
c1, c2 = st.columns(2)
with c1:
    st.caption("Baseline (reference)")
    if base_wav_path and base_bytes is None:
        st.audio(base_wav_path)
    elif base_bytes is not None:
        st.audio(base_bytes)
with c2:
    st.caption(f"Participant — {test_name}")
    st.audio(test_bytes)

st.download_button("Download report (JSON)",
                   json.dumps({k: (v if k != "features" else "omitted-curves") for k, v in report.items()},
                              indent=2, default=str),
                   file_name="voxtrace_report.json")

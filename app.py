# app.py
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchaudio
import soundfile as sf
import tempfile
import os
import io
import math
import json
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Tuple, List, Dict
import re
from datetime import datetime

# Try to set torchaudio backend to soundfile
try:
    torchaudio.utils.set_audio_backend("soundfile")
except Exception:
    pass

# ================================================================
#                STREAMLIT PAGE CONFIG (TOP)
# ================================================================
st.set_page_config(
    page_title="🎧 ALM Audio Intelligence Dashboard (Prototyping)",
    page_icon="🎶",
    layout="wide"
)

# ================================================================
#                FUTURISTIC + ANIMATED THEME (CSS)
# ================================================================
st.markdown("""
<style>

/* GLOBAL BACKGROUND */
.main {
    background: radial-gradient(circle at top left, #00111a, #000c18 70%);
    color: #dff7ff;
    animation: bgShift 18s ease-in-out infinite alternate;
}
@keyframes bgShift {
    0% { background-position: 0% 0%; }
    100% { background-position: 80% 80%; }
}

/* HEADINGS */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #8ae9ff !important;
    text-shadow: 0px 0px 12px #00eaff;
    letter-spacing: 0.8px;
}

/* CARD STYLING */
.block-container {
    padding-top: 2rem;
}
div[data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,255,255,0.08);
    box-shadow: 0 0 25px rgba(0,255,255,0.12);
    animation: floaty 6s ease-in-out infinite;
}
@keyframes floaty {
    0%   { transform: translateY(0px); }
    50%  { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #00cceb, #007bff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s ease;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(0,200,255,0.4);
}
.stButton>button:hover {
    transform: scale(1.06);
    box-shadow: 0 0 20px rgba(0,255,255,0.9);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(0, 15, 30, 0.75);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0,255,255,0.12);
}

/* TEXT INPUT */
.stTextInput>div>div>input {
    background: rgba(0,0,0,0.35);
    color: #dffaff;
    border-radius: 10px;
    border: 1px solid rgba(0,255,255,0.25);
}

/* SLIDERS */
.stSlider>div>div>div {
    background: rgba(255,255,255,0.15);
}
.stSlider>div>div>div>div {
    background: #00eaff;
}

/* ANIMATED NEON DIVIDER */
hr {
    border: none;
    border-top: 1px solid rgba(0,255,255,0.35);
    box-shadow: 0px 0px 8px rgba(0,255,255,0.4);
}

/* small adjustments to audio player glow */
audio {
    box-shadow: 0 0 12px rgba(0,234,255,0.08);
    border-radius: 8px;
}

/* make file uploader look nicer */
[data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 1px dashed rgba(0,255,255,0.12);
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
#                PARTICLES, NEON INTRO, WAVEFORM HELPERS
# ================================================================
particles_html = r"""
<div id="particle-root" style="position:fixed; inset:0; z-index:-1; pointer-events:none;"></div>
<style>
#neon-intro { text-align:center; margin-top:6px; margin-bottom:12px; }
#neon-intro h1 {
  font-family: 'Segoe UI', sans-serif;
  font-size: 34px;
  color: transparent;
  background: linear-gradient(90deg, #00eaff, #7cffb2, #6b8cff);
  -webkit-background-clip: text;
  background-clip: text;
  text-shadow: 0 0 20px rgba(0,234,255,0.25);
  animation: neonPulse 2.5s ease-in-out infinite alternate;
}
@keyframes neonPulse {
  from { text-shadow: 0 0 6px rgba(0,234,255,0.12); transform: translateY(0px); }
  to { text-shadow: 0 0 28px rgba(108,255,212,0.28); transform: translateY(-4px); }
}
</style>
<script>
(function() {
  const root = document.getElementById('particle-root');
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.display = 'block';
  root.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  let particles = [];
  const colors = ['rgba(0,234,255,0.08)','rgba(124,255,178,0.06)','rgba(107,140,255,0.06)'];
  function spawn(n){
    for(let i=0;i<n;i++){
      particles.push({
        x: Math.random()*window.innerWidth,
        y: Math.random()*window.innerHeight,
        vx: (Math.random()-0.5)*0.25,
        vy: (Math.random()-0.5)*0.25,
        r: 0.6 + Math.random()*2.0,
        c: colors[Math.floor(Math.random()*colors.length)]
      });
    }
  }
  spawn(80);
  function resize(){
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
  }
  window.addEventListener('resize', resize);
  function step(){
    ctx.clearRect(0,0, window.innerWidth, window.innerHeight);
    for(let i=0;i<particles.length;i++){
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if(p.x < -10) p.x = window.innerWidth + 10;
      if(p.x > window.innerWidth + 10) p.x = -10;
      if(p.y < -10) p.y = window.innerHeight + 10;
      if(p.y > window.innerHeight + 10) p.y = -10;
      ctx.beginPath();
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r*6);
      g.addColorStop(0, p.c);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.arc(p.x, p.y, p.r*6, 0, Math.PI*2);
      ctx.fill();
    }
    requestAnimationFrame(step);
  }
  step();
})();
</script>
<div id="neon-intro">
  <h1>🌌 Futuristic ALM — Inclusive Audio Intelligence</h1>
</div>
"""
# inject particles (fixed positioned), minimal width/height so it's allowed
components.html(particles_html, height=1, width=1)

# Neon waveform plotting helper
def plot_neon_waveform(waveform: np.ndarray, sr: int, figsize=(10, 1.5)):
    # waveform: 1D numpy array
    if waveform.ndim > 1:
        sig = np.mean(waveform, axis=0)
    else:
        sig = waveform
    max_samples = min(sig.shape[0], sr * 15)  # cap for speed
    sig = sig[:max_samples]
    t = np.linspace(0, len(sig) / sr, num=len(sig))
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.fill_between(t, sig, color='#00eaff', alpha=0.06)
    ax.plot(t, sig, linewidth=0.8, color='#7effd4', alpha=0.95)
    ax.axhline(0, color='#00eaff', alpha=0.12, linewidth=0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig

def display_waveform_from_session():
    if st.session_state.get("waveform", None) is None:
        return
    try:
        wf = st.session_state.waveform.numpy().squeeze()
        if wf.ndim > 1:
            wf = np.mean(wf, axis=0)
        fig = plot_neon_waveform(wf, st.session_state.get("sr", 16000))
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Waveform visualization failed: {e}")

# ================================================================
#                    ORIGINAL BACKEND CODE (MODIFIED)
# ================================================================

@st.cache_resource
def load_asr_pipeline():
    """
    Use Whisper-tiny by default (far smaller memory footprint).
    If you really want to change to a larger model, update the model string here
    but be aware of memory/paging limitations on Windows.
    """
    try:
        # device=-1 forces CPU; adjust if you have GPU (e.g., device=0)
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
        return pipe
    except Exception as e:
        st.warning("Could not load Whisper ASR pipeline: " + str(e))
        return None

@st.cache_resource
def load_emotion_pipeline():
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)
    except Exception:
        return None

@st.cache_resource
def load_t5_summarizer():
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        return model, tokenizer
    except Exception:
        return None, None


def fmt_time(s: float) -> str:
    s = max(0.0, float(s))
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def load_audio_file(path_or_bytes) -> Tuple[torch.Tensor, int]:
    try:
        if isinstance(path_or_bytes, (str, os.PathLike)):
            waveform, sr = torchaudio.load(path_or_bytes)
        else:
            data = path_or_bytes.read()
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            if audio.ndim == 1:
                audio = np.expand_dims(audio, 0)
            waveform = torch.from_numpy(audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1 and waveform.shape[0] <= 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sr
    except Exception as e:
        raise RuntimeError(f"Failed loading audio: {e}")


def resample_waveform(waveform, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return waveform, orig_sr
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform), target_sr


def normalize_waveform(waveform):
    peak = waveform.abs().max()
    if peak > 0:
        return waveform / peak * 0.99
    return waveform


def simple_spectral_gate(waveform, sr, std_threshold=0.5):
    try:
        n = waveform.shape[1]
        noise_frame = waveform[:, : min(int(0.2 * sr), n)]
        noise_floor = noise_frame.mean()
        return waveform - noise_floor * std_threshold
    except Exception:
        return waveform


def write_slice_to_file(waveform, sr, start_s, end_s):
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    end_sample = min(end_sample, waveform.shape[1])
    slice_np = waveform[:, start_sample:end_sample].numpy().T
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, slice_np, sr)
    tmp.flush()
    return tmp.name


def heuristic_word_timestamps(transcript, duration_s):
    words = re.findall(r"\S+", transcript)
    n = len(words)
    timestamps = []
    if n == 0:
        return timestamps
    lengths = [len(w) for w in words]
    total_chars = sum(lengths)
    if total_chars == 0:
        avg = duration_s / n
        cursor = 0.0
        for w in words:
            start = cursor
            end = min(cursor + avg, duration_s)
            timestamps.append({"word": w, "start": start, "end": end})
            cursor = end
        return timestamps
    cursor = 0.0
    for i, w in enumerate(words):
        share = lengths[i] / total_chars
        seg_dur = max(0.05, duration_s * share)
        start = cursor
        end = min(cursor + seg_dur, duration_s if i == n - 1 else cursor + seg_dur)
        timestamps.append({"word": w, "start": start, "end": end})
        cursor = end
    if timestamps:
        timestamps[-1]["end"] = duration_s
    return timestamps


def text_for_time_window(word_ts, start_s, end_s):
    matched = [w["word"] for w in word_ts if w["end"] >= start_s and w["start"] <= end_s]
    return " ".join(matched).strip()


def parse_time_query(question, duration_s):
    q = question.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*s", q)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return max(0, min(a, b)), min(duration_s, max(a, b))
    m = re.search(r"(?:at|around)\s*(\d+(?:\.\d+)?)\s*s", q)
    if m:
        t = float(m.group(1))
        return max(0, t-1), min(duration_s, t+1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*s", q)
    if m:
        t = float(m.group(1))
        return max(0, t-1), min(duration_s, t+1)
    if "middle" in q:
        mid = duration_s / 2
        return mid-1, mid+1
    if "beginning" in q:
        return 0, min(duration_s, 3)
    if "end" in q:
        return max(0, duration_s-3), duration_s
    return 0, duration_s


def answer_generic_question(question, transcription, word_ts, duration_s):
    start, end = parse_time_query(question, duration_s)
    snippet = text_for_time_window(word_ts, start, end)
    if snippet:
        return f"Between {fmt_time(start)} and {fmt_time(end)} the speaker says: \"{snippet}\""
    return "I couldn't understand the time range. Try: 'What is said between 6 and 8 seconds?'"


# ================================================================
#                     STREAMLIT UI (Futuristic)
# ================================================================

st.sidebar.header("⚙️ Processing Options")
enable_emotion = st.sidebar.checkbox("Emotion detection", value=True)
enable_scene_graph = st.sidebar.checkbox("Scene graph", value=True)
enable_auto_process = st.sidebar.checkbox("Auto-process on upload", value=True)
target_sr = st.sidebar.selectbox("Target sample rate", [16000, 22050, 44100], index=0)

# Upload / Record
st.header("🎙 Upload or Record Audio")
col1, col2 = st.columns([2, 1])
with col1:
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac"])
    st.markdown("*or record using microphone*")
    try:
        mic_bytes = st.audio_input("🎤 Record")
    except Exception:
        mic_bytes = None
with col2:
    st.caption("Tip: short clips (<3 minutes) are best for quick demo responses.")

# Session states
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "word_timestamps" not in st.session_state:
    st.session_state.word_timestamps = []
if "duration" not in st.session_state:
    st.session_state.duration = 0.0
if "waveform" not in st.session_state:
    st.session_state.waveform = None
if "sr" not in st.session_state:
    st.session_state.sr = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load models
asr_pipe = load_asr_pipeline()
emotion_pipe = load_emotion_pipeline()
t5_model, t5_tokenizer = load_t5_summarizer()

# PROCESSING
process_now = False
if enable_auto_process and (audio_file or mic_bytes):
    process_now = True
if st.button("🚀 Analyze Audio") and (audio_file or mic_bytes):
    process_now = True

if process_now:
    source = audio_file if audio_file else mic_bytes

    try:
        waveform, sr = load_audio_file(source)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        st.stop()

    waveform = normalize_waveform(waveform)
    waveform = simple_spectral_gate(waveform, sr)
    if sr != target_sr:
        waveform, sr = resample_waveform(waveform, sr, target_sr)

    duration_s = waveform.shape[1] / sr
    st.session_state.waveform = waveform
    st.session_state.sr = sr
    st.session_state.duration = duration_s

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmpfile.name, waveform.numpy().T, sr)
    tmpfile.flush()

    st.audio(tmpfile.name)
    # show neon waveform under the player
    display_waveform_from_session()
    st.caption(f"Duration: {duration_s:.2f}s | SR: {sr} Hz")

    # Transcribe
    st.info("Transcribing...")
    transcription = ""
    if asr_pipe:
        try:
            out = asr_pipe(tmpfile.name)
            transcription = out["text"] if isinstance(out, dict) else str(out)
            st.success("Transcription Ready ✔")
        except Exception as e:
            st.warning(f"ASR failed: {e}")
    else:
        st.warning("ASR pipeline unavailable (model failed to load). Using empty transcription.")
    st.session_state.transcription = transcription

    # Word timestamps
    st.info("Creating rough timestamps…")
    word_ts = heuristic_word_timestamps(transcription, duration_s)
    st.session_state.word_timestamps = word_ts

    st.subheader("📝 Transcript with Estimated Timestamps")
    if word_ts:
        for w in word_ts[:80]:
            st.markdown(f"**{fmt_time(w['start'])}–{fmt_time(w['end'])}:** {w['word']}")
    else:
        st.write("No timestamps detected.")

    # Emotion detection
    if enable_emotion and emotion_pipe and transcription:
        st.info("Detecting emotions…")
        try:
            res = emotion_pipe(transcription[:512])
            emo = res[0]["label"]
            score = res[0]["score"]
            st.success(f"Emotion: **{emo}** ({score:.2f})")
        except Exception:
            st.warning("Emotion detection failed.")

    # Summarizer
    if t5_model and t5_tokenizer and transcription:
        st.info("Generating short summary…")
        try:
            inputs = t5_tokenizer(
                "Summarize: " + transcription,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            outputs = t5_model.generate(**inputs, max_length=60)
            summ = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Summary: " + summ)
            st.session_state.summary = summ
        except Exception:
            st.warning("Failed to summarize.")

    st.session_state.report = {
        "transcription": transcription,
        "duration": duration_s,
        "sr": sr,
        "generated_at": datetime.utcnow().isoformat(),
        "word_timestamps": word_ts
    }

# --------------------------
# CHAT WITH AUDIO
# --------------------------
st.header("💬 Ask Anything About the Audio")

user_q = st.text_input("Your question:")

if st.button("Ask →"):
    if user_q.strip():
        transcription = st.session_state.transcription
        word_ts = st.session_state.word_timestamps
        duration_s = st.session_state.duration

        ans = answer_generic_question(user_q, transcription, word_ts, duration_s)
        st.session_state.chat_history.append(("user", user_q))
        st.session_state.chat_history.append(("assistant", ans))

# Show chat
for role, text in st.session_state.chat_history[-20:]:
    if role == "user":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 ALM:** {text}")

# --------------------------
# SEGMENT PLAYER
# --------------------------
st.header("🎛 Interactive Segment Player")

if st.session_state.waveform is not None:
    dur = st.session_state.duration
    start = st.slider("Start", 0.0, dur, 0.0, step=0.1, key="seg_start")
    end = st.slider("End", 0.0, dur, min(start + 0.5, dur), step=0.1, key="seg_end")

    if st.button("▶️ Play Segment"):
        if start < end:
            tmp = write_slice_to_file(st.session_state.waveform, st.session_state.sr, start, end)
            st.audio(tmp)
        else:
            st.warning("Start must be < End")

# --------------------------
# EXPORT REPORT
# --------------------------
st.header("📦 Export Analysis")

if "report" in st.session_state:
    rep = json.dumps(st.session_state.report, indent=2)
    st.download_button("Download JSON Report", rep, "alm_report.json")

st.divider()
st.caption("✨ Futuristic ALM Dashboard — Developed by Prachi Deore")

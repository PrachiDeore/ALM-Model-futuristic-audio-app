# scripts/preprocess.py
import torchaudio
import torch
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa

OUTDIR = Path("data/features")
OUTDIR.mkdir(exist_ok=True)

def compute_mel(path, sr=16000, n_mels=128, hop=160, win=400):
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, hop_length=hop, win_length=win)
    log_mel = librosa.power_to_db(mel)
    return log_mel.astype(np.float32)

if __name__ == "__main__":
    scenes = list(Path("data/scenes").glob("*.wav"))
    for s in scenes:
        mel = compute_mel(str(s))
        out = OUTDIR / (s.stem + ".npy")
        np.save(out, mel)
    print("Computed mel for", len(scenes))

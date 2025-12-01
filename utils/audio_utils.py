import torchaudio
import torch
import numpy as np

# -------------------------------
# 🎧 AUDIO FEATURE EXTRACTION UTILS
# -------------------------------

def load_audio(file_path, target_sr=16000):
    """
    Loads an audio file and resamples it to the target sampling rate.
    Returns waveform tensor and sample rate.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sr


def extract_features(waveform, sr, n_mfcc=40):
    """
    Extracts MFCC + log-mel spectrogram features for downstream model.
    """
    # Compute MFCCs
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
    )(waveform)

    # Compute log-mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=400,
        hop_length=160,
        n_mels=64
    )(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)

    # Flatten both features and concatenate
    mfcc_flat = mfcc.mean(dim=-1).squeeze()
    mel_flat = mel_spec.mean(dim=-1).squeeze()
    combined = torch.cat((mfcc_flat, mel_flat), dim=0)

    # Ensure consistent shape for model
    if combined.ndim == 1:
        combined = combined.unsqueeze(0)

    return combined


def detect_audio_events(waveform, sr):
    """
    Simple heuristic detector to print what kind of sound might exist.
    You can replace this with a pretrained AudioSet classifier later.
    """
    energy = torch.mean(waveform ** 2).item()
    duration = waveform.shape[1] / sr
    avg_db = 10 * np.log10(energy + 1e-9)

    event = "quiet environment"
    if avg_db > -20:
        event = "very loud sound (vehicle, music, or machinery)"
    elif avg_db > -35:
        event = "moderate sound (speech or indoor noise)"
    elif avg_db > -50:
        event = "soft sound (background chatter)"
    
    return {
        "duration_sec": round(duration, 2),
        "avg_db": round(avg_db, 2),
        "detected_event": event
    }


# -------------------------------
# 🧪 TEST
# -------------------------------
if __name__ == "__main__":
    test_file = "sample.wav"  # replace with an existing .wav for testing
    waveform, sr = load_audio(test_file)
    feats = extract_features(waveform, sr)
    events = detect_audio_events(waveform, sr)
    print("Feature shape:", feats.shape)
    print("Audio summary:", events)

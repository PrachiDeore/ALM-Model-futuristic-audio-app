# scripts/create_scenes.py
import random
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import json

DATA_DIR = Path("data")
OUT_DIR = Path("data/scenes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_audio(path):
    return AudioSegment.from_file(path)

def random_mix(speech_path, event_paths, out_path, snr_db=5):
    speech = load_audio(speech_path)
    # start with environmental background (choose first event as background)
    bg = load_audio(event_paths[0]).apply_gain(-10)  # quieter background
    combined = bg
    # overlay speech at random position
    pos = random.randint(0, max(0, len(bg) - len(speech)))
    combined = combined.overlay(speech, position=pos)
    # overlay other events at random times
    event_info = []
    for ev in event_paths[1:]:
        ev_audio = load_audio(ev)
        start = random.randint(0, max(0, len(combined) - len(ev_audio)))
        combined = combined.overlay(ev_audio, position=start)
        event_info.append({"file": str(ev), "start_ms": start, "duration_ms": len(ev_audio)})
    combined.export(out_path, format="wav")
    return pos, event_info

if __name__ == "__main__":
    # Example usage — replace paths with real data
    speech = "data/example_speech.wav"
    events = ["data/airport_ambient.wav", "data/airplane_passby.wav", "data/subway.wav"]
    out = OUT_DIR/"scene1.wav"
    speech_pos,event_info = random_mix(speech, events, out)
    meta = {
        "scene_file": str(out),
        "speech_start_ms": speech_pos,
        "events": event_info
    }
    with open(OUT_DIR/"scene1.json","w") as f:
        json.dump(meta,f,indent=2)
    print("created", out)

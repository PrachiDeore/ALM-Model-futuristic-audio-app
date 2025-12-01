# scripts/download_manifests.py
import os
import csv
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_manifest(name, rows, header):
    path = DATA_DIR / f"{name}.tsv"
    with open(path, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print("Saved:", path)

def make_dummy_audioset_manifest():
    header = ["youtube_id", "start", "end", "label"]
    # Add rows for later replacement with real AudioSet IDs.
    rows = [
        ("dQw4w9WgXcQ", 0, 10, "airplane"),
    ]
    save_manifest("audioset_small", rows, header)

if __name__ == "__main__":
    make_dummy_audioset_manifest()

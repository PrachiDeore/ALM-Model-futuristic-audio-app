# scripts/generate_qa.py
import json
from pathlib import Path

SCENE_DIR = Path("data/scenes")
OUT = SCENE_DIR/"qa_pairs.jsonl"

def template_questions(meta):
    # meta: dict produced by create_scenes
    qas = []
    # Closed-ended: event presence
    for ev in meta["events"]:
        label = Path(ev["file"]).stem
        q = f"Is there {label.replace('_',' ')} sound?"
        a = "yes"
        qas.append({"question": q, "answer": a, "type":"closed"})
    # Location inference example
    qas.append({"question":"Where is the person likely located?","answer":"near an airport/boarding area","type":"open"})
    return qas

if __name__ == "__main__":
    files = list(SCENE_DIR.glob("*.json"))
    with open(OUT, "w", encoding="utf-8") as out:
        for f in files:
            meta = json.load(open(f))
            qas = template_questions(meta)
            for qa in qas:
                record = {"scene": meta["scene_file"], **qa}
                out.write(json.dumps(record, ensure_ascii=False)+"\n")
    print("QA saved to", OUT)

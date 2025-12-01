# training/train.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

class SceneDataset(Dataset):
    def __init__(self, features_dir, qa_file):
        self.features = list(Path(features_dir).glob("*.npy"))
        self.qa = [json.loads(l) for l in open(qa_file, encoding="utf-8")]
        # naive mapping: assume same order; in practice build an index by scene filename
        self.scene2qa = {}
        for q in self.qa:
            scene = Path(q["scene"]).stem
            self.scene2qa.setdefault(scene, []).append(q)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        mel = np.load(f)
        stem = f.stem
        qas = self.scene2qa.get(stem, [])
        # Return mel and first QA as sample
        qa = qas[0] if qas else {"question":"", "answer":""}
        return {"mel": torch.tensor(mel), "qa": qa}

def collate_fn(batch):
    # pad mel to same shape, minimal example
    maxT = max(b["mel"].shape[1] for b in batch)
    B = len(batch)
    X = torch.zeros(B, batch[0]["mel"].shape[0], maxT)
    for i,b in enumerate(batch):
        T = b["mel"].shape[1]
        X[i,:,:T] = b["mel"]
    return X, [b["qa"] for b in batch]

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    for X, qas in dataloader:
        X = X.to(device)
        # dummy forward; integrate actual label conversion to token ids for text head
        out = model(X)
        loss = out.get('aed_loss', torch.tensor(0.0, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    from models.multimodal_aln import ALMModel
    ds = SceneDataset("data/features", "data/scenes/qa_pairs.jsonl")
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALMModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(10):
        train_one_epoch(model, dl, optimizer, device)
        print("Epoch", epoch, "done")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# -------------------------------
# 🧠 Simple Audio QA Model
# -------------------------------
class AudioQAModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_classes=10):
        super(AudioQAModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # You can later replace classifier with LLM or reasoning head

    def forward(self, feats):
        x = self.encoder(feats)
        logits = self.classifier(x)
        return logits

    # -------------------------------
    # 🗣️ QA Reasoning Interface
    # -------------------------------
    def answer_question(self, feats, question, tokenizer=None):
        """
        Given preprocessed audio features and a question string,
        returns a natural-language answer.
        Replace with real reasoning later (LLM + audio embeddings).
        """
        # 🔸 Example heuristic reasoning
        q_lower = question.lower()

        if "airport" in q_lower or "plane" in q_lower:
            return "There’s an airplane and announcements — likely at an airport."
        elif "music" in q_lower:
            return "Background sounds suggest someone is playing music or singing."
        elif "street" in q_lower or "traffic" in q_lower:
            return "You can hear vehicles and horns — sounds like a street or highway."
        elif "dog" in q_lower or "bark" in q_lower:
            return "The audio has dog barking sounds — maybe outdoors or in a park."
        else:
            return "The environment seems to have mixed sounds, possibly a public place."


# -------------------------------
# ⚙️ Training Skeleton (Optional)
# -------------------------------
def train_model(train_loader, val_loader, num_epochs=5, lr=1e-3, device="cpu"):
    model = AudioQAModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/qa_model.pt")
    print("✅ Model saved to checkpoints/qa_model.pt")
    return model


# -------------------------------
# 🔁 Model Loader (used by Streamlit app)
# -------------------------------
def load_model(checkpoint_path="checkpoints/qa_model.pt", device="cpu"):
    model = AudioQAModel()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print("⚠️ No trained model found — using default weights.")
    model.eval()
    tokenizer = None  # Placeholder (for LLM or text embedding model)
    return model, tokenizer


# -------------------------------
# 🧪 Quick Test
# -------------------------------
if __name__ == "__main__":
    # Simulate dummy features and question
    dummy_feats = torch.randn(1, 128)
    question = "Where is the person?"
    model, tok = load_model()
    answer = model.answer_question(dummy_feats, question)
    print("Answer:", answer)

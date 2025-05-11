import torch
import torch.nn as nn
from torch.optim import Adam
import json

from model.carlai_model import CarlAI

# Lade Daten
with open("data/encoded.txt", "r") as f:
    tokens = list(map(int, f.read().split()))

with open("data/vocab.json", "r") as f:
    vocab = json.load(f)

stoi, itos = vocab["stoi"], vocab["itos"]
vocab_size = len(stoi)

# Hyperparameter
seq_len = 64
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Batches vorbereiten
def get_batch():
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.tensor(tokens[i+1:i+seq_len+1]) for i in ix])
    return x.to(device), y.to(device)

# Modell initialisieren
model = CarlAI(vocab_size=vocab_size).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Speichern
torch.save(model.state_dict(), "carlai_model.pt")

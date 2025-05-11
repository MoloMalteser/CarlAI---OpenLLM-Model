# Datei: export/export_torchscript.py
import torch
from model.carlai_model import CarlAI
import json

with open("data/vocab.json") as f:
    vocab = json.load(f)
vocab_size = len(vocab["stoi"])

model = CarlAI(vocab_size=vocab_size)
model.load_state_dict(torch.load("carlai_model.pt"))
model.eval()

# Beispielinput (Batchgröße 1, Sequenzlänge 64)
dummy_input = torch.randint(0, vocab_size, (1, 64))

# Exportieren
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("export/carlai_torchscript.pt")

print("✅ Export abgeschlossen: export/carlai_torchscript.pt")

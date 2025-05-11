import torch
from model.carlai_model import CarlAI
import json

with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
stoi, itos = vocab["stoi"], vocab["itos"]
vocab_size = len(stoi)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CarlAI(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("carlai_model.pt"))
model.eval()

def generate(start_token, length=100):
    tokens = [stoi[start_token]]
    for _ in range(length):
        x = torch.tensor(tokens[-64:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        next_token = torch.argmax(logits[0, -1]).item()
        tokens.append(next_token)
    return ''.join([itos[str(i)] for i in tokens])

print(generate("H", 200))

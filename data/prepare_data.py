import string
import json

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def tokenize(text, stoi):
    return [stoi[ch] for ch in text if ch in stoi]

def detokenize(tokens, itos):
    return ''.join([itos[i] for i in tokens])

if __name__ == "__main__":
    with open("data/dataset.txt", "r") as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    encoded = tokenize(text, stoi)

    with open("data/vocab.json", "w") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)

    with open("data/encoded.txt", "w") as f:
        f.write(" ".join(map(str, encoded)))

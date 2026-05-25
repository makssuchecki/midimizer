import torch
import json
from pathlib import Path

from src.model.lstm import LSTMModel
from src.model.gru import GRUModel
from src.model.transformer import TransformerModel


from src.utils.midi_writer import events_to_midi

def load_vocab():
    with open("data/index/vocab.json") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    return vocab, id_to_token


def sample(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate(model, start_token, length=200):
    tokens = [start_token]

    for _ in range(length):
        x = torch.tensor(tokens).unsqueeze(0)
        logits = model(x)

        next_token = sample(logits[0, -1])
        tokens.append(next_token)

    return tokens


def tokens_to_events(tokens, id_to_token):
    events = []

    for t in tokens:
        tok = id_to_token[t]

        if tok.startswith("time_shift"):
            value = int(tok.split("_")[2])
            events.append({"type": "time_shift", "value": value})

        elif tok.startswith("note_on"):
            _, _, pitch, vel = tok.split("_")
            events.append({
                "type": "note_on",
                "pitch": int(pitch),
                "velocity": int(vel)
            })

        elif tok.startswith("note_off"):
            _, _, pitch = tok.split("_")
            events.append({
                "type": "note_off",
                "pitch": int(pitch)
            })

    return events


if __name__ == "__main__":
    vocab, id_to_token = load_vocab()

    model = TransformerModel(len(vocab))
    model.load_state_dict(torch.load("outputs/tra10-3000.pt"))
    model.eval()

    tokens = generate(model, start_token=0, length=300)

    events = tokens_to_events(tokens, id_to_token)

    print(events[:20])
    events_to_midi(events, "outputs/midi/generated.mid")
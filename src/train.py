import torch
from torch.utils.data import DataLoader
import json
import os

from src.data.dataset import MidiDataset
from src.model.lstm import LSTMModel


def main():
    with open("data/index/vocab.json") as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)

    dataset = MidiDataset("data/processed/train_chunks.jsonl", limit=3000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = LSTMModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(5):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = loss_fn(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss:.4f}")
        torch.save(model.state_dict(), f"outputs/lstm_epoch_{epoch}.pt")
        
if __name__ == "__main__":
    main()
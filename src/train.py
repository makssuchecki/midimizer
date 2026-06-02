import os
import torch
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

from src.data.dataset import MidiDataset

from src.model.lstm import LSTMModel
from src.model.gru import GRUModel
from src.model.transformer import TransformerModel

'''
Trening
Pipeline:
- load vocab
- load dataset
- DataLoader
- model
- training loop
'''

def main():
    with open("data/index/vocab.json") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    dataset = MidiDataset(
        "data/processed/train_chunks.jsonl",
        limit=30000,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = GRUModel(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs("outputs", exist_ok=True)

    best_loss = float("inf")
    loss_history = []

    for epoch in range(20):
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

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "outputs/best_model_tr.pt")

    print("Training finished.")


    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
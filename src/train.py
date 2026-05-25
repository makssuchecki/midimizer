import os
import torch
from torch.utils.data import DataLoader
import json

from src.data.dataset import MidiDataset

from src.model.lstm import LSTMModel
from src.model.gru import GRUModel
from src.model.transformer import TransformerModel


def main():
    # wczytaj vocab
    with open("data/index/vocab.json") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    # dataset (fast mode)
    dataset = MidiDataset(
        "data/processed/train_chunks.jsonl",
        limit=3000,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = TransformerModel(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # folder na modele
    os.makedirs("outputs", exist_ok=True)

    # best model tracking
    best_loss = float("inf")

    # trening
    for epoch in range(10):
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

        # średni loss (lepszy niż suma)
        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # zapis najlepszego modelu
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "outputs/best_model.pt")

    print("Training finished.")


if __name__ == "__main__":
    main()
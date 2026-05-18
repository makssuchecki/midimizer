import json
import torch
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, path, limit=3000):
        self.data = []

        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                self.data.append(json.loads(line)["tokens"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1])
        y = torch.tensor(tokens[1:])
        return x, y
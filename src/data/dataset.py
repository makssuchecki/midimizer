import json
import torch
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, path):
        self.data = []

        with open(path) as f:
            for line in f:
                self.data.append(json.loads(line)["tokens"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1])
        y = torch.tensor(self.data[idx][1:])
        return x, y
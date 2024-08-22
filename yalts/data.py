import numpy as np
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.zeros = torch.zeros((2048), dtype=torch.long)

    def __len__(self):
        return int(1e8)

    def __getitem__(self, index):
        return self.zeros


class MemMapBinaryDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

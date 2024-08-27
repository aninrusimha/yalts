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
    def __init__(self, file) -> None:
        super().__init__()
        self.file = np.memmap(file, dtype=np.uint16, mode="r").reshape(-1, 2048)

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, index):
        # in reality, we probably won't have time to download data and preprocess it
        return torch.tensor(self.file[index], dtype=torch.long)

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    def __init__(self, path: Path | str, sequence_length: int):
        self._tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self._sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self._tokens) // self._sequence_length

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self._sequence_length
        row = self._tokens[start : start + self._sequence_length]

        return torch.from_numpy(row.astype(np.int64))


def create_data_loader(path: Path | str, sequence_length: int, batch_size: int, train: bool, seed: int = 0) -> DataLoader:
    return DataLoader(
        dataset=TokenDataset(path, sequence_length),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        generator=torch.Generator().manual_seed(seed),
    )

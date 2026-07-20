"""CIFAR-10 torch Dataset backed by the pre-decoded numpy arrays
(prepare_cifar10.py) -- avoids any further network access during training."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE.replace("/scripts", "/data")

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


class CIFAR10Numpy(Dataset):
    def __init__(self, split="train", augment=False):
        assert split in ("train", "test")
        self.images = np.load(f"{DATA_DIR}/cifar10_{split}_images.npy")  # (N,32,32,3) uint8
        self.labels = np.load(f"{DATA_DIR}/cifar10_{split}_labels.npy")
        # augment=True (opt-in, default off for backward compat) applies the
        # standard CIFAR train-time random crop(pad 4)+horizontal flip on the
        # raw image before normalization -- needed for a genuinely good clean
        # baseline (phase3_guardbias/train_clean_baseline.py).
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            padded = torch.nn.functional.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            img = padded[:, top:top + 32, left:left + 32]
        img = (img - MEAN) / STD
        return img, int(self.labels[idx])

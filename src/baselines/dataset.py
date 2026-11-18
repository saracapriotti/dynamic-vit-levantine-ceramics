import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FabricsDataset(Dataset):
    """
    CSV with ';' separator and latin-1 encoding:
      column 0 -> image path relative to IMG_DIR
      column 1 -> class label (1-20), shifted to 0-19
    """

    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file, delimiter=";", encoding="latin-1")
        self.img_dir = img_dir
        self.transform = transform

        self.df.dropna(subset=[self.df.columns[1]], inplace=True)
        self.df[self.df.columns[1]] = self.df[self.df.columns[1]].astype(int)

        self.image_paths = self.df.iloc[:, 0].apply(
            lambda x: os.path.join(img_dir, x.replace("\\", "/"))
        )
        self.labels = self.df.iloc[:, 1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths.iloc[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = self.labels.iloc[idx] - 1  # 0-based
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), -1   # removed by collate_fn


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_transform, val_transform


def collate_fn(batch):
    """Drop images that could not be loaded (label = -1)."""
    batch = [(x, y) for x, y in batch if y >= 0]
    if not batch:
        return torch.empty(0, 3, 224, 224, dtype=torch.float32), torch.empty(0, dtype=torch.int64)
    return torch.utils.data.default_collate(batch)


def split_70_15_15(n: int, seed: int):
    """
    70/15/15 train/val/test split, identical to the one used for the Dynamic ViT
    (notebooks/01_training.ipynb): after setting the seed, random_split is called
    twice and the second call defines the split.
    """
    set_seed(seed)
    lengths = [int(config.TRAIN_RATIO * n), int(config.VAL_RATIO * n)]
    lengths.append(n - sum(lengths))
    torch.utils.data.random_split(range(n), lengths)
    train, val, test = torch.utils.data.random_split(range(n), lengths)
    return train.indices, val.indices, test.indices


def build_dataloaders(
    csv_file: str = config.CSV_FILE,
    img_dir: str = config.IMG_DIR,
    seed: int = config.SEED,
):
    """Data loaders for one seed. The seed defines the split, exactly as for the Dynamic ViT."""
    train_transform, val_transform = get_transforms()

    # Separate dataset objects for training and evaluation, so that each subset
    # gets its own transform
    train_base = FabricsDataset(csv_file, img_dir,
                                transform=train_transform if config.USE_AUGMENTATION else val_transform)
    eval_base = FabricsDataset(csv_file, img_dir, transform=val_transform)

    n = len(train_base)
    train_idx, val_idx, test_idx = split_70_15_15(n, seed)
    train_size, val_size, test_size = len(train_idx), len(val_idx), len(test_idx)

    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(eval_base, val_idx)
    test_ds  = Subset(eval_base, test_idx)

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"Dataset split — train: {train_size}, val: {val_size}, test: {test_size}")
    print(f"Number of classes: {train_base.labels.nunique()}")

    return train_loader, val_loader, test_loader, train_base, train_ds, val_ds, test_ds

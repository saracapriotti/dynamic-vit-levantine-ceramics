import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FabricsDataset(Dataset):
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
            return torch.zeros(3, 224, 224), -1


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
    batch = [(x, y) for x, y in batch if y >= 0]
    if not batch:
        return torch.empty(0, 3, 224, 224, dtype=torch.float32), torch.empty(0, dtype=torch.int64)
    return torch.utils.data.default_collate(batch)


def build_dataloaders(
    csv_file: str = config.CSV_FILE,
    img_dir: str = config.IMG_DIR,
    seed: int = config.SEED,
):
    set_seed(seed)

    train_transform, val_transform = get_transforms()

    dataset = FabricsDataset(csv_file=csv_file, img_dir=img_dir, transform=None)

    train_size = int(config.TRAIN_RATIO * len(dataset))
    val_size   = int(config.VAL_RATIO   * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform
    test_ds.dataset.transform  = val_transform

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
    print(f"Number of classes: {len(dataset.labels.unique())}")

    return train_loader, val_loader, test_loader, dataset, train_ds, val_ds, test_ds

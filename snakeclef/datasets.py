"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

switch to v2 of transforms
https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms.v2 import InterpolationMode

from augmentations import Grid
from paths import METADATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR

# dataset loading issue
# https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
WORKER_TIMEOUT_SECS = 120


def get_train_transform(cfg, image_size, pretrained):
    """
    training transformations
    : param image_size: Image size of resize when applying transforms.
    """
    resize = max(768, image_size * 2)  # resized dataset to 768

    train_augs = [
        v2.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.RandomCrop(image_size),
    ]

    if cfg["train_aug"]["trivial_aug"]:
        train_augs.append(v2.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC))
    elif cfg["train_aug"]["auto_aug"]:
        train_augs.append(v2.AutoAugment(interpolation=InterpolationMode.BICUBIC))
    elif cfg["train_aug"]["random_aug"]:
        train_augs.append(v2.RandAugment(interpolation=InterpolationMode.BICUBIC))

    train_augs.extend([
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        normalize_transform(pretrained)
    ])

    if cfg["train_aug"]["gridmask_prob"] is not None and cfg["train_aug"]["gridmask_prob"] > 0.0:
        train_augs.append(Grid(360, 0, cfg["train_aug"]["gridmask_prob"]))

    train_transform = v2.Compose(train_augs)
    return train_transform


# Validation transforms
def get_valid_transform(image_size, pretrained, fivecrop=False):
    """
    validation transformations
    """
    if fivecrop:
        valid_transform = v2.Compose([
            v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.FiveCrop(image_size),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            normalize_transform(pretrained)
        ])
    else:
        valid_transform = v2.Compose([
            v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(image_size),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            normalize_transform(pretrained)
        ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:  # Normalization when training from scratch.
        normalize = v2.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


def get_datasets(cfg, pretrained, image_size):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """

    original_train_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "SnakeCLEF2023-TrainMetadata-iNat.csv",
        img_dir=TRAIN_DATA_DIR,
        transform=(get_train_transform(cfg, image_size, pretrained)),
    )
    supp_train_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "SnakeCLEF2023-TrainMetadata-HM.csv",
        img_dir=TRAIN_DATA_DIR,
        transform=(get_train_transform(cfg, image_size, pretrained)),
    )
    val_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "SnakeCLEF2023-ValMetadata.csv",
        img_dir=VAL_DATA_DIR,
        transform=(get_valid_transform(image_size, pretrained)),
    )

    train_dataset = ConcatDataset([original_train_dataset, supp_train_dataset])
    train_dataset.target = np.array(list(original_train_dataset.target) + list(supp_train_dataset.target))

    return train_dataset, val_dataset


class CustomImageDataset(Dataset):
    def __init__(self, label_file_path: str, img_dir: str,
                 transform=None,
                 target_transform=None):

        self.classes = None
        self.target = None
        self.image_filenames = None
        self.labels = None
        self.img_dir = img_dir
        self._setup_from_df(label_file_path)
        self.transform = transform
        self.target_transform = target_transform

    def _setup_from_df(self, label_file_path):
        df = pd.read_csv(label_file_path, dtype={"class_id": "int64"})

        df = df[["image_path", "class_id"]]

        self.classes = df["class_id"].unique()
        self.target = df["class_id"].values
        self.image_filenames = df["image_path"].values
        self.labels = df["class_id"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """If an image can't be read, print the error and return None"""
        try:
            img_path = os.path.join(self.img_dir, self.image_filenames[idx])
            # from torchvision.io import read_image
            # image = read_image(img_path)
            # https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
            with open(img_path, "rb") as f:
                image = Image.open(f).convert('RGB')  # hopefully this handles greyscale cases
            # image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
        except Exception as e:
            print("issue loading image")
            print(e)
            return None

        return image, label


def collate_fn(batch):
    """Filter None from the batch"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders(train_dataset, val_dataset, batch_size, num_workers, balanced_sampler,
                     timeout=WORKER_TIMEOUT_SECS):
    """
    Prepares the training and validation data loaders.
    :param train_dataset: The training dataset.
    :param val_dataset: The validation dataset.
    :param batch_size: batch_size.
    :param num_workers: Number of parallel processes for data preparation.
    Returns the training and validation data loaders.

    References:
    - https://pytorch.org/docs/stable/data.html
    - https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
    """

    if balanced_sampler:
        target = train_dataset.target
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight_per_class = 1. / class_sample_count
        weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
        weight_per_sample = torch.from_numpy(weight_per_sample)
        weight_per_sample = weight_per_sample.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                                  num_workers=num_workers,
                                  timeout=timeout,
                                  collate_fn=collate_fn,
                                  )

    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            timeout=timeout,
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=True, num_workers=max(1, int(num_workers * 0.25)),
        timeout=timeout,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader

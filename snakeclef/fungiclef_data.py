"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

switch to v2 of transforms
https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
"""

import math
import os
import re
from collections import Counter
from math import radians, cos, sin, pi
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.transforms.v2 import InterpolationMode

from augmentations import Grid
from paths import DATA_DIR, METADATA_DIR

countryCode = [
    'FO', 'DE', 'GA', 'CR', 'SJ', 'AT', 'IE', 'JP', 'FR', 'AU', 'PL', 'GR', 'US',
    'RU', 'PT', 'RE', 'AL', 'CA', 'IT', 'NP', 'HR', 'GL', 'GB', 'FI', 'NO', 'CH',
    'SE', 'HU', 'NL', 'ES', 'IS', 'BE', 'CZ', 'DK',
    # new ones:
    # 'LV', 'LI', 'BA', 'RO', 'MT', 'EE', 'ID',
]

Substrate = [
    'fruits', 'remains of vertebrates (e.g. feathers and fur)', 'other substrate',
    'dead wood (including bark)', 'wood', 'stems of herbs, grass etc', 'dead stems of herbs, grass etc',
    'stone', 'wood and roots of living trees', 'mycetozoans', 'bark', 'calcareous stone',
    'building stone (e.g. bricks)', 'faeces', 'wood chips or mulch', 'living leaves', 'fungi',
    'living stems of herbs, grass etc', 'leaf or needle litter',
    'siliceous stone', 'mosses', 'bark of living trees', 'insects', 'spiders', 'living flowers', 'catkins', 'lichens',
    'soil', 'liverworts', 'peat mosses', 'cones', 'fire spot',
    # new ones:
    # 'šišky', 'mechorosty', 'půda', 'kůra živých stromů', 'houby', 'odumřelé dřevo (včetně kůry)',
    # 'dřevní štěpka nebo mulč', 'listí nebo jehličí', 'dřevo a kořeny živých stromů',
]  # contains nan

Habitat = [
    'bog', 'Unmanaged deciduous woodland', 'Unmanaged coniferous woodland', 'Acidic oak woodland',
    'ditch', 'wooded meadow, grazing forest', 'natural grassland', 'heath', 'dune', 'park/churchyard',
    'fertilized field in rotation', 'gravel or clay pit', 'Deciduous woodland', 'Bog woodland',
    'coniferous woodland/plantation', 'Forest bog', 'Mixed woodland (with coniferous and deciduous trees)',
    'lawn', 'salt meadow', 'Willow scrubland', 'improved grassland', 'rock', 'garden', 'Thorny scrubland',
    'other habitat', 'fallow field', 'masonry', 'roadside', 'hedgerow', 'meadow', 'roof',
    # new ones:
    # 'zahrada', 'louka/trávník', 'listnatý les', 'park/hřbitov', 'louka', 'jehličnatý les s přirozeným charakterem',
    # 'lesní rašeliniště', 'krajnice', 'trávník', 'acidofilní / kyselá doubrava', 'listnatý les s přirozeným charakterem',
    # 'smíšený les', 'pole', 'příkop', 'jehličnatý les / monokultura', 'živý plot',

]  # contains nan

# dataset loading issue
# https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
WORKER_TIMEOUT_SECS = 120


# Training transforms
def get_train_transform(cfg, image_size, pretrained):
    """
    training transformations
    : param image_size: Image size of resize when applying transforms.
    """
    # resize = max(image_size + 8, max_image_size)
    resize = max(384 * 2, image_size * 2)  # resized dataset to 768

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
def get_valid_transform(image_size, pretrained):
    """
    validation transformations
    """
    resize = image_size
    valid_transform = v2.Compose([
        v2.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
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


def get_datasets(cfg, pretrained, image_size, validation_frac, oversample=False, undersample=False,
                 oversample_prop=0.1, equal_undersampled_val=True, seed=42, include_metadata=False, training_augs=True):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """

    if cfg["train"]["new_data_split"]:
        print("using new data split. ignoring other variables such as `equal_undersampled_val` and `undersample`.")

        train_20 = CustomImageDataset(
            label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            img_dir=DATA_DIR / "DF20",
            transform=((get_train_transform(cfg, image_size, pretrained)) if training_augs else
                       (get_valid_transform(image_size, pretrained))),  # this is the difference
            include_metadata=include_metadata,
        )
        train_21 = CustomImageDataset(
            label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            img_dir=DATA_DIR / "DF21",
            transform=((get_train_transform(cfg, image_size, pretrained)) if training_augs else
                       (get_valid_transform(image_size, pretrained))),  # this is the difference
            include_metadata=include_metadata,
        )
        val_20 = CustomImageDataset(
            label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            img_dir=DATA_DIR / "DF20",
            transform=(get_valid_transform(image_size, pretrained)),  # only difference
            include_metadata=include_metadata,
        )
        val_21 = CustomImageDataset(
            label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            img_dir=DATA_DIR / "DF21",
            transform=(get_valid_transform(image_size, pretrained)),  # only difference
            include_metadata=include_metadata,
        )
        total_train = ConcatDataset(train_20, train_21)
        total_val = ConcatDataset(val_20, val_21)
        total_train.target = np.array([list(train_20.target) + list(train_21.target)])
        total_val.target = np.array([list(train_20.target) + list(train_21.target)])

        # get the indices for the train and val datasets where val is max(10% total samples, 3)
        train_indices = []
        val_indices = []
        for cls in np.unique(val_20.target):
            cls_count_20 = np.sum(val_20[val_20.target == cls])
            cls_count_21 = np.sum(val_21[val_21.target == cls])
            cls_count_total = cls_count_20 + cls_count_21
            if cls_count_21 > 0.1 * cls_count_total:
                # add all train samples to train
                train_indices.extend(np.where(val_20.target == cls)[0])
                # move some additional samples from val to train
                # only add as many as needed to reach 10% of total samples in val, 90% in test
                train_indices.extend(np.where(val_21.target == cls)[0][:int(0.1 * cls_count_total)])
                val_indices.extend(np.where(val_21.target == cls)[0][int(0.1 * cls_count_total):])
            elif cls_count_20 > 0.9 * cls_count_total or cls_count_21 < 3:
                # add all val samples to val
                val_indices.extend(np.where(val_21.target == cls)[0])
                # move some additional samples from train to val
                # only add as many as needed to reach 10% of total samples in val, 90% in test or 3 samples in val
                n_samples_move_to_val = max(3 - cls_count_21, int(0.1 * cls_count_total))
                val_indices.extend(np.where(val_20.target == cls)[0][:n_samples_move_to_val])
                train_indices.extend(np.where(val_20.target == cls)[0][n_samples_move_to_val:])

        # data integrity checks
        # assert no overlap
        assert len(train_indices) + len(val_indices) == len(total_train)
        # assert val has at least 3 samples per class
        val_counts = Counter(total_val.target)
        assert all([v >= 3 for v in val_counts.values()])
        # assert train is 90% of each class except for classes for which val has 3 samples
        train_counts = Counter(total_train.target)
        for k, v in train_counts.items():
            if val_counts[k] == 3:
                assert train_counts[k] > 3
                continue
            assert v >= 0.9 * (val_counts[k] + v)

        train_dataset = Subset(total_train, indices=train_indices)
        train_dataset.target = total_train.target[train_indices]
        val_dataset = Subset(total_val, indices=val_indices)
        val_dataset.target = total_val.target[val_indices]
        return train_dataset, val_dataset, total_train.classes

    dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF20",
        transform=((get_train_transform(cfg, image_size, pretrained)) if training_augs else
                   (get_valid_transform(image_size, pretrained))),  # this is the difference
        include_metadata=include_metadata,
    )
    val_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF20",
        transform=(get_valid_transform(image_size, pretrained)),  # only difference
        include_metadata=include_metadata,
    )
    targets = dataset.target

    if equal_undersampled_val:
        # get 4 samples per class for validation
        sample_dict = Counter(targets)
        sample_dict = {k: 4 for k in sample_dict.keys()}
        under = RandomUnderSampler(sampling_strategy=sample_dict, random_state=seed)
        all_indices = np.array(list(np.arange(targets.shape[0])))
        test_indices, _ = under.fit_resample(all_indices.reshape(-1, 1), targets)
        test_indices = test_indices.squeeze()
        train_indices = np.delete(all_indices, test_indices)
    else:
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets,
                                                       test_size=validation_frac, random_state=seed)

    if undersample:
        under = RandomUnderSampler(random_state=seed)
        if oversample:
            sample_dict = Counter(targets)
            majority = max(sample_dict.values())
            sample_dict = {k: max(v, int(majority * oversample_prop)) for k, v in sample_dict.items()}
            over = RandomOverSampler(sampling_strategy=sample_dict, random_state=seed)
            train_indices, _ = over.fit_resample(train_indices.reshape(-1, 1), targets[train_indices])
            train_indices = train_indices.squeeze()
            print(train_indices.shape)
        train_indices, _ = under.fit_resample(train_indices.reshape(-1, 1), targets[train_indices])
        train_indices = train_indices.squeeze()

    train_dataset = Subset(dataset, indices=train_indices)
    train_dataset.target = targets[train_indices]
    val_dataset = Subset(val_dataset, indices=test_indices)
    val_dataset.target = targets[test_indices]
    return train_dataset, val_dataset, dataset.classes


class CustomImageDataset(Dataset):
    def __init__(self, label_file_path: str, img_dir: str,
                 keep_only: set | None = None,
                 exclude: set | None = None,
                 transform=None,
                 target_transform=None,
                 include_metadata=False,
                 metadata_from_cache=True):

        self.classes = None
        self.target = None
        self.image_filenames = None
        self.labels = None
        self.metadata = []
        self.include_metadata = include_metadata
        self.metadata_from_cache = metadata_from_cache
        self.img_dir = img_dir
        self._setup_from_df(exclude, keep_only, label_file_path)
        self.transform = transform
        self.target_transform = target_transform

    def _setup_from_df(self, exclude, keep_only, label_file_path):
        df = pd.read_csv(label_file_path, dtype={"class_id": "int64"})

        if self.include_metadata:
            create_metadata = True
            write_cache = False
            if self.metadata_from_cache:
                cache_location = Path(self.img_dir).parent / f"{Path(self.img_dir).name}.npy"
                if cache_location.exists():
                    with open(cache_location, 'rb') as f:
                        self.metadata = np.load(f)
                    create_metadata = False
                else:
                    write_cache = True

            if create_metadata:
                for idx, row in df.iterrows():
                    encoded_metadata = encode_metadata_row(row)
                    self.metadata.append(encoded_metadata)

                self.metadata = np.array(self.metadata).astype(np.float32)

            if write_cache:
                with open(cache_location, 'wb') as f:
                    np.save(f, self.metadata)

        df = df[["image_path", "class_id"]]

        if keep_only is not None:
            df = df[df["class_id"].isin(keep_only)]
        if exclude is not None:
            df = df[~df["class_id"].isin(exclude)]
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
            if self.include_metadata:
                metadata = self.metadata[idx]
                metadata = torch.tensor(metadata, dtype=torch.float32)
        except Exception as e:
            print("issue loading image")
            print(e)
            return None

        if self.include_metadata:
            return (image, metadata), label
        return image, label


def encode_metadata_row(row):
    temporal_info = encode_temporal_info(row["month"], row["day"])
    # spatial_info = encode_spatial_info(row["Latitude"], row["Longitude"])
    countryCode_onehot_info = encode_onehot(row["countryCode"], countryCode)
    Substrate_onehot_info = encode_onehot(row["Substrate"], Substrate)
    Habitat_onehot_info = encode_onehot(row["Habitat"], Habitat)
    onehot_info = countryCode_onehot_info + Substrate_onehot_info + Habitat_onehot_info
    encoded_metadata = temporal_info + onehot_info
    return encoded_metadata


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
    """

    if balanced_sampler:
        target = train_dataset.target
        # https://pytorch.org/docs/stable/data.html
        # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight_per_class = 1. / class_sample_count
        weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
        weight_per_sample = torch.from_numpy(weight_per_sample)
        weight_per_sample = weight_per_sample.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
        # class_sample_counts = Counter(train_dataset.target)
        # class_sample_counts = [v for k, v in sorted(class_sample_counts.items(), key=lambda x: x[0])]
        # weights = 1 / torch.Tensor(class_sample_counts)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  # pin_memory_device=device,
                                  timeout=timeout,
                                  # persistent_workers=True,
                                  collate_fn=collate_fn,
                                  )

    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            # pin_memory=True,
            # pin_memory_device=device,
            timeout=timeout,
            # persistent_workers=True,
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=max(1, int(num_workers * 0.25)),
        # pin_memory=True,
        # pin_memory_device=device,
        timeout=timeout,
        # persistent_workers=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def get_openset_datasets(cfg, pretrained, image_size, n_train=2000, n_val=200, seed=42, training_augs=True,
                         include_metadata=False):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    :param training_augs: Whether to use training augmentations for the training dataset. This is useful if fine-tuning
        on the unknown/open dataset, but probably not desired if creating embeddings for openGAN.
    Returns the training, validation, and test sets for the openset dataset.
    """

    # set up the dataset twice but with different transformations
    train_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        keep_only={-1},
        transform=((get_train_transform(cfg, image_size, pretrained)) if training_augs else
                   (get_valid_transform(image_size, pretrained))),  # this is the difference
        include_metadata=include_metadata
    )
    val_test_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        keep_only={-1},
        transform=(get_valid_transform(image_size, pretrained)),  # this is the difference
        include_metadata=include_metadata
    )

    indices = list(range(train_dataset.target.shape[0]))
    test_indices, train_indices = train_test_split(indices, test_size=n_train, random_state=seed)
    test_indices, val_indices = train_test_split(test_indices, test_size=n_val, random_state=seed)

    train_dataset = Subset(train_dataset, indices=train_indices)
    val_dataset = Subset(val_test_dataset, indices=val_indices)
    test_dataset = Subset(val_test_dataset, indices=test_indices)
    train_dataset.target = np.array([-1] * len(train_dataset))
    val_dataset.target = np.array([-1] * len(val_dataset))
    test_dataset.target = np.array([-1] * len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def get_closedset_test_dataset(pretrained, image_size, include_metadata=False):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training, validation, and test sets for the openset dataset.
    """

    # drop the unknown label
    val_test_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        exclude={-1},
        transform=(get_valid_transform(image_size, pretrained)),
        include_metadata=include_metadata
    )

    return val_test_dataset


def get_dataloader_combine_and_balance_datasets(dataset_1, dataset_2, batch_size, num_workers=16,
                                                timeout=WORKER_TIMEOUT_SECS,
                                                persistent_workers=False, unknowns=False):
    """
    https://pytorch.org/docs/stable/data.html
    https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
    """
    dataset = ConcatDataset([dataset_1, dataset_2])
    target = np.array(list(dataset_1.target) + list(dataset_2.target))
    target = target + 1 if unknowns else target  # -1 is now 0 to satisfy logic below
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight_per_class = 1. / class_sample_count
    weight_per_class[0] = 0.5 if unknowns else weight_per_class[0]
    weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
    weight_per_sample = torch.from_numpy(weight_per_sample)
    weight_per_sample = weight_per_sample.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers,
                            timeout=timeout,
                            persistent_workers=persistent_workers,
                            )
    return dataloader


# the imagefolder dataset will sort by class then filename, so for data splitting purposes, we can sort the
# image_path per class and get back the order in the dataset
# full_dataset = ImageFolder(root=DATA_FROM_FOLDER_DIR)
# def get_openset_datasets(pretrained, image_size, n_train=2000, n_val=200, seed=42):
#     """
#     Function to prepare the Datasets.
#     :param pretrained: Boolean, True or False.
#     Returns the training, validation, and test sets for the openset dataset.
#     """
#
#     # TODO: revisit the idea of training transformations for the open set discriminator training
#
#     # set up the dataset twice but with different transformations
#     train_dataset = CustomImageDataset(
#         label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
#         img_dir=DATA_DIR / "DF21",
#         keep_only={-1},
#         transform=(get_train_transform(image_size, pretrained))
#     )
#     val_test_dataset = CustomImageDataset(
#         label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
#         img_dir=DATA_DIR / "DF21",
#         keep_only={-1},
#         transform=(get_valid_transform(image_size, pretrained))  # this is the difference
#     )
#
#     indices = list(range(train_dataset.target.shape[0]))
#     test_indices, train_indices = train_test_split(indices, test_size=n_train, random_state=seed)
#     test_indices, val_indices = train_test_split(test_indices, test_size=n_val, random_state=seed)
#
#     train_dataset = Subset(train_dataset, indices=train_indices)
#     val_dataset = Subset(val_test_dataset, indices=val_indices)
#     test_dataset = Subset(val_test_dataset, indices=test_indices)
#     train_dataset.target = np.array([-1] * len(train_dataset))
#     val_dataset.target = np.array([-1] * len(val_dataset))
#     test_dataset.target = np.array([-1] * len(test_dataset))
#
#     return train_dataset, val_dataset, test_dataset


def get_spatial_info(latitude, longitude):
    if latitude and longitude:
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude) * cos(longitude)
        y = cos(latitude) * sin(longitude)
        z = sin(latitude)
        return [x, y, z]
    else:
        return [0, 0, 0]


def get_temporal_info(date, miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = sin(2 * pi * month / 12)
                y_month = cos(2 * pi * month / 12)
                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2 * pi * hour / 24)
                    y_hour = cos(2 * pi * hour / 24)
                return [x_month, y_month, x_hour, y_hour]
            else:
                return [0, 0, 0, 0]
        else:
            return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]


def encode_temporal_info(month, day):
    if math.isnan(month):
        x_month = -2
        y_month = -2
    else:
        month = int(month)
        x_month = sin(2 * pi * month / 12)
        y_month = cos(2 * pi * month / 12)
    if math.isnan(day):
        x_day = -2
        y_day = -2
    else:
        day = int(day)
        x_day = sin(2 * pi * day / 31)
        y_day = cos(2 * pi * day / 31)
    temporal_info = [x_month, y_month, x_day, y_day]
    return temporal_info


def encode_spatial_info(latitude, longitude):
    if latitude and longitude:
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude) * cos(longitude)
        y = cos(latitude) * sin(longitude)
        z = sin(latitude)
        return [x, y, z]
    else:
        return [0, 0, 0]


def encode_onehot(attr, attr_list, additional_unknown_idx=False):
    if additional_unknown_idx:
        code = [0] * (len(attr_list) + 1)  # add unknown idx
        if isinstance(attr, str):
            try:
                idx = attr_list.index(attr)
            except ValueError:
                idx = len(attr_list)  # add to final position, unknown idx
            code[idx] = 1

    else:
        code = [0] * len(attr_list)
        if isinstance(attr, str):
            try:
                idx = attr_list.index(attr)
                code[idx] = 1
            except ValueError:
                pass

    return code


def fungi_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    imgs, targets, genus_targets, auxs, img_names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = torch.tensor(targets, dtype=torch.int64)
    genus_targets = torch.tensor(genus_targets, dtype=torch.int64)
    auxs = [torch.tensor(aux, dtype=torch.float64) for aux in auxs]
    auxs = torch.stack(auxs, dim=0)
    return imgs, targets, genus_targets, auxs, img_names

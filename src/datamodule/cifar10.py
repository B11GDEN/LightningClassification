import cv2

from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional
from pathlib import Path

import albumentations as albu
from albumentations.pytorch import ToTensorV2


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Cifar10Dataset(BaseDataset):

    def __init__(self,
                 data_dir: Path,
                 train: bool,
                 transform: callable = None,
                 ) -> None:
        self.labels = []
        self.imgs = []

        if train:
            files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
            for file in files:
                data = unpickle(data_dir / file)
                self.imgs.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        else:
            data = unpickle(data_dir / 'test_batch')
            self.imgs.extend(data[b'data'])
            self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):

        img = self.imgs[idx].reshape(3, 32, 32).transpose([1, 2, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

class Cifar10DataModule(LightningDataModule):

    def __init__(self,
                 data_dir: Path,
                 batch_size: int = 32,
                 image_size: int = 32,
                 num_workers: int = 0,
                 persistent_workers: bool = False,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_ds: Optional[Cifar10Dataset] = None
        self.val_ds: Optional[Cifar10Dataset] = None
        self.test_ds: Optional[Cifar10Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.train_ds and not self.val_ds and not self.test_ds:
            train_transform = albu.Compose(
                [
                    albu.HorizontalFlip(),
                    albu.ShiftScaleRotate(),
                    albu.ColorJitter(),
                    albu.Normalize(),
                    ToTensorV2()
                ]
            )
            test_transform = albu.Compose(
                [
                    albu.Normalize(),
                    ToTensorV2()
                ]
            )
            self.train_ds = Cifar10Dataset(
                data_dir=self.data_dir,
                train=True,
                transform=train_transform,
            )
            self.val_ds = Cifar10Dataset(
                data_dir=self.data_dir,
                train=False,
                transform=train_transform,
            )
            self.test_ds = Cifar10Dataset(
                data_dir=self.data_dir,
                train=False,
                transform=test_transform,
            )

    @property
    def num_classes(self):
        return 10

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          )
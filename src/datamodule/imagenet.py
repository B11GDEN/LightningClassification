import csv
import cv2

from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional
from pathlib import Path

import albumentations as albu
from albumentations.pytorch import ToTensorV2


class ImagenetDataset(BaseDataset):

    def __init__(self,
                 data_path: Path,
                 conf_path: Path,
                 transform: callable = None,
                 train: bool = False,
                 ) -> None:

        self.data_path = data_path
        self.conf_path = conf_path
        self.transform = transform
        self.train = train

        self.names = []
        self.classes = []

        with open(self.conf_path, 'r') as csv_file:
            data_csv = csv.reader(csv_file)
            next(data_csv)
            for line in data_csv:
                self.names.append(line[0])
                self.classes.append(line[1].split(' ')[0])

        self.uniq_classes = sorted(list(set(self.classes)))

    def __len__(self):

        return len(self.names)

    def __getitem__(self, idx: int):

        if self.train:
            im = self.data_path / self.classes[idx] / f"{self.names[idx]}.JPEG"
        else:
            im = self.data_path / f"{self.names[idx]}.JPEG"
        img = cv2.imread(str(im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.uniq_classes.index(self.classes[idx])

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label


class ImagenetDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: Path,
                 config_dir: Path,
                 batch_size: int = 32,
                 image_size: int = 224,
                 num_workers: int = 0,
                 persistent_workers: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_ds: Optional[ImagenetDataset] = None
        self.val_ds: Optional[ImagenetDataset] = None
        self.test_ds: Optional[ImagenetDataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.train_ds and not self.val_ds and not self.test_ds:
            transform = albu.Compose(
                [
                    albu.Resize(self.image_size, self.image_size),
                    albu.Normalize(),
                    ToTensorV2()
                ]
            )
            self.train_ds = ImagenetDataset(
                data_path=Path(self.data_dir / "train"),
                conf_path=Path(self.config_dir / "LOC_train_solution.csv"),
                transform=transform,
                train=True,
            )
            self.val_ds = ImagenetDataset(
                data_path=Path(self.data_dir / "val"),
                conf_path=Path(self.config_dir / "LOC_val_solution.csv"),
                transform=transform,
            )
            self.test_ds = ImagenetDataset(
                data_path=Path(self.data_dir / "test"),
                conf_path=Path(self.config_dir / "LOC_sample_submission.csv"),
                transform=transform,
            )

    @property
    def num_classes(self):
        return 1000

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

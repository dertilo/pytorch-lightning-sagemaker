import os
import shutil

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST

# based on: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/mnist_datamodule.py

class MNISTDataModule(LightningDataModule):

    name = 'mnist'

    def __init__(
            self,
            data_dir: str,
            output_data_dir:str,
            val_split: int = 5000,
            num_workers: int = 16,
            normalize: bool = False,
            seed: int = 42,
            batch_size=32,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.output_data_dir = output_data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):

        try:
            os.system("tar xzf %s -C %s" % (self.data_dir+"/output.tar.gz", self.data_dir)) #TODO(tilo): handle failure more explicitly
            MNIST(self.data_dir, train=True, download=False, transform=transform_lib.ToTensor())
            MNIST(self.data_dir, train=False, download=False, transform=transform_lib.ToTensor())
        except:
            m = MNIST(self.output_data_dir, train=True, download=True, transform=transform_lib.ToTensor())
            MNIST(self.output_data_dir, train=False, download=True, transform=transform_lib.ToTensor())
            shutil.rmtree(m.raw_folder)
            self.data_dir = self.output_data_dir

    def train_dataloader(self, transforms=None):
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, transforms=None):
        transforms = transforms or self.val_transforms or self._default_transforms()
        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, transforms=None):
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms

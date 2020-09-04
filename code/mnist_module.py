import os
from argparse import ArgumentParser

import torch
from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# based on: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/mnist_module.py


class LitMNIST(LightningModule):
    def __init__(
        self,
        hidden_dim=128,
        learning_rate=1e-3,
        batch_size=32,
        num_workers=4,
        data_dir="",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()  # this saves all the params into a hparams field, what kind of magic is this?

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

        self.mnist_train = None
        self.mnist_val = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        progress_bar_metrics = tensorboard_logs
        return {
            "loss": loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar_metrics,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {"val_loss": F.cross_entropy(y_hat, y), "acc": acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar_metrics,
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {"test_loss": F.cross_entropy(y_hat, y), "acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss, "test_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar_metrics,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt:off
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--data_dir', type=str, default='mnist_data')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        # fmt:on
        return parser

seed_everything(42)

def run_cli():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)
    args = parser.parse_args()

    args.max_epochs = 2
    model = LitMNIST(**vars(args))

    dm = MNISTDataModule(num_workers=args.num_workers, data_dir=args.data_dir)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
    """
    {'test_acc': tensor(0.8718), 'test_loss': tensor(0.3433)}
    """

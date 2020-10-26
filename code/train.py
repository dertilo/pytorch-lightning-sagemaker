import argparse
import os
from pprint import pprint
import logging

import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import ec2_metadata
from ec2_metadata import ec2_metadata as ec2metadata
from mnist_module import LitMNIST
from mnist_datamodule import MNISTDataModule

logging.basicConfig(level=logging.DEBUG)
seed_everything(42)
DEBUG = False
# INSTANCE_ACTION = 'TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && curl -H "X-aws-ec2-metadata-token: $TOKEN" –v http://169.254.169.254/latest/meta-data/spot/instance-action'  # see: https://aws.amazon.com/blogs/compute/best-practices-for-handling-ec2-spot-instance-interruptions/
# INSTANCE_META_DATA = 'TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/'  # see: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
# see ec2_metadata package!

with open("secrets.env", "r") as f:
    key = f.readline().strip("\n").replace("WANDB_API_KEY=", "")
os.environ["WANDB_API_KEY"] = key


class InterruptionWarning(Callback):
    ALL_GOOD = 0
    INTERRUPTION_WARNING = 1

    def __init__(self,check_status_interval = 200) -> None:
        super().__init__()
        self.check_status_interval = check_status_interval

    def on_train_batch_end(
        self, trainer, pl_module: LitMNIST, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % self.check_status_interval == 0:
            status = self._get_status()
            trainer.callback_metrics["interruption_warning"] = status
        else:
            trainer.callback_metrics["interruption_warning"] = self.ALL_GOOD

    def _get_status(self):
        try:
            resp = ec2metadata._get_url(
                ec2_metadata.METADATA_URL + "instance-action", allow_404=True
            )
            # see: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
            if resp.status_code == 404:
                # This URI returns a 404 response code when the instance is not marked for interruption
                status = self.ALL_GOOD
            elif resp.status_code == 200:
                # If the instance is marked for interruption, you receive a 200 response code.
                status = self.INTERRUPTION_WARNING
            else:
                status = self.INTERRUPTION_WARNING
        except:
            status = self.ALL_GOOD
        return status


class ModelCheckpointOnBatchEnd(ModelCheckpoint):
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_validation_end(trainer, pl_module)


if __name__ == "__main__":
    if DEBUG:
        os.environ["SM_OUTPUT_DATA_DIR"] = "/tmp/output_data_dir"
        os.environ["SM_MODEL_DIR"] = "/tmp/model_dir"
        os.environ["SM_CHANNEL_TRAINING"] = "/tmp/channel_train"
        os.environ["SM_CHANNEL_TEST"] = "/tmp/channel_test"

    parser = argparse.ArgumentParser()
    # fmt:off
    output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']
    checkpoint_path = output_data_dir + "/checkpoints/"
    parser.add_argument('-o','--output-data-dir', type=str, default=output_data_dir)
    parser.add_argument('--data_dir', type=str,default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)
    # fmt:on

    wandb.init(project="mnist")
    assert wandb.api.api_key is not None

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)

    default_kwargs = (
        {"batch_size": 32, "max_epochs": 2, "gpus": 0, "hidden_dim": 32}
        if DEBUG
        else {"default_root_dir": output_data_dir}
    )
    args, _ = parser.parse_known_args(namespace=argparse.Namespace(**default_kwargs))
    pprint(args.__dict__)

    dm = MNISTDataModule(
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        output_data_dir=output_data_dir,
        batch_size=args.batch_size,
    )
    model = LitMNIST(**vars(args))
    args.logger = WandbLogger(
        project="mnist",
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(InterruptionWarning(500))
    checkpoint_callback = ModelCheckpointOnBatchEnd(
        prefix=args.checkpoint_path,
        monitor="interruption_warning",
        save_top_k=2,
        mode="max",
        verbose=True,
    )

    trainer.callbacks.append(trainer.configure_checkpoint_callback(checkpoint_callback))

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

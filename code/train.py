import argparse
import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mnist_module import LitMNIST
from mnist_datamodule import MNISTDataModule

seed_everything(42)
DEBUG = True


class InterruptionWarning(Callback):
    def on_train_batch_end(
        self, trainer, pl_module: LitMNIST, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 10:
            print("interrupt training")
            trainer.callback_metrics["interruption_warning"] = 1.0
        else:
            trainer.callback_metrics["interruption_warning"] = 0.0


class ModelCheckpointOnBatchEnd(ModelCheckpoint):
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_validation_end(trainer, pl_module)


if __name__ == "__main__":
    if DEBUG:
        os.environ["SM_OUTPUT_DATA_DIR"] = "/tmp/output_data_dir"
        os.environ["SM_MODEL_DIR"] = "/tmp/model_dir"
        os.environ["SM_CHANNEL_TRAINING"] = "/tmp/channel_train"
        os.environ["SM_CHANNEL_TEST"] = "/tmp/channel_test"
        checkpoint_path = "/tmp/checkpoints/"
    else:
        checkpoint_path = "/opt/ml/checkpoints/"

    parser = argparse.ArgumentParser()
    # fmt:off
    output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']
    parser.add_argument('-o','--output-data-dir', type=str, default=output_data_dir)
    parser.add_argument('--data_dir', type=str,default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)

    # fmt:on

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)

    kwargs = (
        {"batch_size": 32, "max_epochs": 2, "gpus": 0, "hidden_dim": 128}
        if DEBUG
        else {"default_root_dir": output_data_dir}
    )
    args, _ = parser.parse_known_args(namespace=argparse.Namespace(**kwargs))
    pprint(args.__dict__)

    dm = MNISTDataModule(
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        output_data_dir=output_data_dir,
        batch_size=args.batch_size,
    )
    model = LitMNIST(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(InterruptionWarning())
    checkpoint_callback = ModelCheckpointOnBatchEnd(
        prefix=args.checkpoint_path,
        monitor="interruption_warning",
        save_top_k=2,
        mode="max",
        verbose=True,
    )
    # trainer.checkpoint_callback = [
    #     trainer.configure_checkpoint_callback(checkpoint_callback)
    # ]
    trainer.callbacks.append(trainer.configure_checkpoint_callback(checkpoint_callback))

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

import argparse
import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from mnist_module import LitMNIST
from mnist_datamodule import MNISTDataModule

seed_everything(42)
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        os.environ["SM_OUTPUT_DATA_DIR"] = "/tmp/output_data_dir"
        os.environ["SM_MODEL_DIR"] = "/tmp/model_dir"
        os.environ["SM_CHANNEL_TRAIN"] = "/tmp/channel_train"
        os.environ["SM_CHANNEL_TEST"] = "/tmp/channel_test"

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # fmt:on

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)

    kwargs = (
        {"batch_size": 32, "max_epochs": 2, "gpus": 0, "hidden_dim": 128}
        if DEBUG
        else {}
    )
    args, _ = parser.parse_known_args(namespace=argparse.Namespace(**kwargs))
    pprint(args.__dict__)

    dm = MNISTDataModule(
        num_workers=args.num_workers, data_dir=args.data_dir, batch_size=args.batch_size
    )
    model = LitMNIST(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

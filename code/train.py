import argparse
import os
from pprint import pprint

import pytorch_lightning as pl
from pl_bolts import LitMNIST
from pl_bolts.datamodules import MNISTDataModule

if __name__ == '__main__':
    os.environ['SM_OUTPUT_DATA_DIR'] = "/tmp/output_data_dir"
    os.environ['SM_MODEL_DIR'] = "/tmp/model_dir"
    os.environ['SM_CHANNEL_TRAIN'] = "/tmp/channel_train"
    os.environ['SM_CHANNEL_TEST'] = "/tmp/channel_test"

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--gpus', type=int, default=0) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr','--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('-te','--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    # args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)
    # args = parser.parse_args()
    kwargs = {"batch_size": 32, "max_epochs": 2,"gpus":0}
    args, _ = parser.parse_known_args(namespace=argparse.Namespace(**kwargs))
    pprint(args.__dict__)

    dm = MNISTDataModule(num_workers=args.num_workers, data_dir=args.train)

    model = LitMNIST(**vars(args))

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(args.batch_size), dm.val_dataloader(args.batch_size))
    # trainer.test(model,datamodule=dm)
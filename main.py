import json
import boto3
import sagemaker
import wandb
from sagemaker.pytorch import PyTorch
# based on: https://github.com/aletheia/mnist_pl_sagemaker/blob/master/main.py
source_dir = 'code'
wandb.sagemaker_auth(path=source_dir)

sagemaker_session = sagemaker.Session()
# bucket_name = sagemaker_session.default_bucket()
bucket_name = "sagemaker-eu-central-1-706022464121/pytorch-training-2020-10-26-19-02-00-900/output"
bucket = f's3://{bucket_name}'

role = 'arn:aws:iam::706022464121:role/SageMakerRole_MNIST' # sagemaker.get_execution_role()

estimator = PyTorch(
  entry_point='train.py',
  source_dir=source_dir,
  role=role,
  framework_version='1.4.0',
  py_version="py3",
  instance_count=1,
  # instance_type="local",# 'ml.p2.xlarge',
  instance_type="ml.c5.xlarge",#"ml.g4dn.xlarge",# 'ml.p2.xlarge',
  use_spot_instances = True,
  max_wait = 24 * 60 * 60, # seconds; see max_run
  # checkpoint_s3_uri = ... #TODO(tilo)
  hyperparameters={
  'max_epochs': 2,
  'batch_size': 32,
  })

estimator.fit(f"{bucket}")

# [ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.c5n.xlarge, ml.p3.16xlarge, ml.m5.large, ml.p2.16xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.c5n.18xlarge, ml.g4dn.xlarge, ml.g4dn.12xlarge, ml.c4.8xlarge, ml.g4dn.2xlarge, ml.c5.9xlarge, ml.g4dn.4xlarge, ml.c5.xlarge, ml.g4dn.16xlarge, ml.c4.xlarge, ml.g4dn.8xlarge, ml.c5n.2xlarge, ml.c5n.4xlarge, ml.c5.18xlarge, ml.p3dn.24xlarge, ml.p3.2xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.c5n.9xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.p2.8xlarge, ml.m5.2xlarge, ml.p3.8xlarge, ml.m4.4xlarge]
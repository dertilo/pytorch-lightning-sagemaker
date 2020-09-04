# MNIST on SageMaker with PyTorch Lightning
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
# based on: https://github.com/aletheia/mnist_pl_sagemaker/blob/master/main.py

# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()

# The bucket containig our input data
bucket = 's3://dataset.mnist'

# The IAM Role which SageMaker will impersonate to run the estimator
# Remember you cannot use sagemaker.get_execution_role()
# if you're not in a SageMaker notebook, an EC2 or a Lambda
# (i.e. running from your local PC)

role = 'arn:aws:iam::682411330166:role/SageMakerRole_MNIST' # sagemaker.get_execution_role()

# Creates a new PyTorch Estimator with params
estimator = PyTorch(
  # name of the runnable script containing __main__ function (entrypoint)
  entry_point='train.py',
  # path of the folder containing training code. It could also contain a
  # requirements.txt file with all the dependencies that needs
  # to be installed before running
  source_dir='code',
  role=role,
  framework_version='1.4.0',
  train_instance_count=1,
  train_instance_type='ml.p2.xlarge',
  # these hyperparameters are passed to the main script as arguments and
  # can be overridden when fine tuning the algorithm
  hyperparameters={
  'epochs': 6,
  'batch-size': 128,
  })

# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container
estimator.fit({
    'train': bucket+'/training',
    'test': bucket+'/testing'
})
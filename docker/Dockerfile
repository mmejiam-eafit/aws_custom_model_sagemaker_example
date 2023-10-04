# all tf dockers here https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&ordering=last_updated
# nothing for 2.3 !!!
#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-gpu-py38-cu116-ubuntu20.04-e3-v1.0
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
COPY requirements.txt /build/

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get install python3 -y \
    && apt-get install python3-pip -y

RUN pip3 install --upgrade pip

RUN pip3 install -r /build/requirements.txt

# Install sagemaker-training toolkit to enable SageMaker Python SDK
RUN pip3 install sagemaker-training

RUN mkdir -p /opt/ml/code

# Copies the training code inside the container
#COPY train.py /opt/ml/code/train.py
COPY *.py /opt/ml/code/

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py

#{
#    "repository": {
#        "repositoryArn": "arn:aws:ecr:us-east-1:056057680849:repository/chexnet-pruning-experiments",
#        "registryId": "056057680849",
#        "repositoryName": "chexnet-pruning-experiments",
#        "repositoryUri": "056057680849.dkr.ecr.us-east-1.amazonaws.com/chexnet-pruning-experiments",
#        "createdAt": "2022-07-21T20:51:48-05:00",
#        "imageTagMutability": "MUTABLE",
#        "imageScanningConfiguration": {
#            "scanOnPush": false
#        },
#        "encryptionConfiguration": {
#            "encryptionType": "AES256"
#        }
#    }
#}
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 056057680849.dkr.ecr.us-east-1.amazonaws.com
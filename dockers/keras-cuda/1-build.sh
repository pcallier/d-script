#!/bin/bash
set -e

# image name
__image=lab41/cudnn-keras

# build image
echo "Building caffe-cuda-cudnn"
docker build -t $__image .


#!/bin/bash -ex

docker build -t orts_origin -f Dockerfile.server --build-arg ONNXRUNTIME_REPO=https://github.com/zhijxu-MS/onnxruntime .

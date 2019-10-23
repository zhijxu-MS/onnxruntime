#!/bin/bash -ex

docker build -t orts_ngraph -f Dockerfile.server.ngraph --build-arg ONNXRUNTIME_REPO=https://github.com/zhijxu-MS/onnxruntime .

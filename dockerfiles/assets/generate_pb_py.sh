#!/bin/bash -ex

protoc --python_out=. predict.proto
protoc --python_out=. prediction_service.proto
protoc --python_out=. onnx-ml.proto

#!/bin/bash -ex

docker run -v /home/zhijxu/onnxruntime/dockerfiles:/models -p 9001:9001 -d orts_ngraph \
           --model_path /models/ssd.onnx --http_port 9001

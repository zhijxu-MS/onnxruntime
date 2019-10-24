#!/bin/bash -ex

docker build -t orts_mkldnn_mklml -f Dockerfile.server.mkldnn_mklml .

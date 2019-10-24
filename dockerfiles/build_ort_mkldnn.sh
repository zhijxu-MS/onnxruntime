#!/bin/bash -ex

docker build -t orts_mkldnn -f Dockerfile.server.mkldnn .

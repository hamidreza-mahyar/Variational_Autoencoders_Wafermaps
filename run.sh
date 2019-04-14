#!/bin/bash

docker run \
    -v "$(pwd)/jupyter":/jupyter \
    -p 8888:8888 \
    --workdir /jupyter \
    --entrypoint /jupyter/run_jupyter.sh \
    python:2

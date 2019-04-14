#!/bin/bash

pip install --trusted-host pypi.python.org -r requirements.txt
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export TF_CPP_MIN_LOG_LEVEL=2

python=/home/n_redwan/workspace2/r_keras/bin/python
$python -u ./main_ae.py


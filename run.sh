#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export TF_CPP_MIN_LOG_LEVEL=2
declare -a files=("./main_cnn.py" "./main_ae.py")
for i in "${files[@]}"
do
  echo $i
  python -u $i
done

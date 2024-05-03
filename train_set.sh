#!/bin/bash

source /opt/miniconda/bin/activate auramask
export TF_CPP_MIN_LOG_LEVEL="2"
export AURAMASK_LOG_FREQ=10
python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C rgb --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 1.0 1.0 -L ssim mse -a 2e-4
python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C rgb --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 0.6 0.4 -L ssim mse -a 2e-4

python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C yuv --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 1.0 1.0 -L ssim mse -a 2e-4
python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C yuv --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 0.6 0.4 -L ssim mse -a 2e-4

python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C hsv --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 1.0 1.0 -L ssim mse -a 2e-4
python train.py -v 0 -E 100 --t-split train[:25%] --v-split train[26%:27%] -B 32 --n-filters 32 --depth 5 -C hsv --aesthetic -g 1.0 --no-note -e 1.0 -D lfw -l 0.6 0.4 -L ssim mse -a 2e-4
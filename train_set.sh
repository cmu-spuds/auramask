source /opt/miniconda/bin/activate unet
export TF_CPP_MIN_LOG_LEVEL="2"
python train.py --no-note -v 0 -E 500 --t_split train[0:2048] --v_split train[2049:2176] -F arcface -B 32 --n_filters 32 --depth 3 -L none || { echo 'Training failed' ; exit 1; }
python train.py --no-note -v 0 -E 500 --t_split train[0:2048] --v_split train[2049:2176] -F arcface -B 32 --n_filters 64 --depth 3 -L none || { echo 'Training failed' ; exit 1; }
python train.py --no-note -v 0 -E 500 --t_split train[0:2048] --v_split train[2049:2176] -F arcface -B 32 --n_filters 16 --depth 5 -L none || { echo 'Training failed' ; exit 1; }
python train.py --no-note -v 0 -E 500 --t_split train[0:2048] --v_split train[2049:2176] -F arcface -B 32 --n_filters 32 --depth 5 -L none || { echo 'Training failed' ; exit 1; }
python train.py --no-note -v 0 -E 500 --t_split train[0:2048] --v_split train[2049:2176] -F arcface -B 32 --n_filters 64 --depth 5 -L none || { echo 'Training failed' ; exit 1; }
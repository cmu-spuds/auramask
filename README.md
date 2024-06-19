# Requirements
- Python: 3.11
- ffmpeg
- libsm6
- libxext6

# Setup
To run the notebooks in this repository, first clone it with submodules.
```sh
git clone git@github.com:PrettyLilLiars/face-benchmark.git
```
Then, install requirements in the requirements.txt file
```sh
pip install -r requirements.txt
```

Needs the following (https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-2016019354)

Automating the variable set/unset process with Anaconda (one-time setup)

Activate your environement in which TF 2.16.1 is installed
- Two files to be created in "anaconda3/envs/<ENV_NAME>/etc/conda"
- anaconda3/envs/<ENV_NAME>/etc/conda/activate.d/env_vars.sh
```sh
#!/bin/sh
export NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
export LD_LIBRARY_PATH=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

anaconda3/envs/<ENV_NAME>/etc/conda/deactivate.d/env_vars.sh
```sh
#!/bin/sh
unset NVIDIA_DIR
unset LD_LIBRARY_PATH
```

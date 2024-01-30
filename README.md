# Requirements
- Python: 3.11
- ffmpeg
- libsm6
- libxext6

# Setup
To run the notebooks in this repository, first clone it with submodules.
```sh
git clone --recurse-submodules git@github.com:PrettyLilLiars/face-benchmark.git
```
Then, install requirements in the requirements.txt file
```sh
pip install -r requirements.txt
```
Finally, get the pairwise LFW dataset
```sh
cd lfw_pairs
tfds build --overwrite
```
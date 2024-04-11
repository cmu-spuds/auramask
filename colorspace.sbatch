#!/bin/bash
#SBATCH --job-name=colorspace_exp               # Job name
#SBATCH --account=gts-sdas7                     # charge account
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=logasja@gatech.edu          # Where to send mail
#SBATCH --nodes=1                               # How many nodes to use
#SBATCH --mem-per-gpu=256gb                     # Memory per GPU
#SBATCH --gres=gpu:RTX_6000:1                   # Number of GPUs per node
#SBATCH -q inferno                              # QOS
#SBATCH --time=12:00:00                         # Time limit hrs:min:sec
#SBATCH --output=../batchruns/Report_%A-%a.out  # Standard output and error log
#SBATCH --array=1-6                               # Array range
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

module load anaconda3/2022.05.0.1
conda activate unet

# export WANDB_API_KEY=uncommentandaddapikey
export WANDB_CONSOLE=off
# export WANDB_RUN_GROUP="${SLURM_JOB_NAME} <Unique>"

DATASET=instagram
EPOCH=500
TSPLIT=train[:25%]
VSPLIT=train[26%:28%]
F=(arcface vggface)
BATCH_SIZE=32
N_FILTERS=32
DEPTH=5
L=ssim
L_W=(0.5 1.0)
COL=(rgb hsv yuv)
TB_LOGS=/storage/home/hcoda1/0/plogas3/tb-logs/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/
SEED=IAMGROOT

# Compute the index for L to use on this test
L_W_IDX=$(( (($SLURM_ARRAY_TASK_ID - 1) / 3) ))
COL_IDX=$(( (($SLURM_ARRAY_TASK_ID - 1) % 3) ))

echo Running task $SLURM_ARRAY_TASK_ID, which will train with a loss weight of ${L_W[$L_W_IDX]} and using the ${COL[$COL_DIX]} colorspace.
echo Training against F = ${F[@]}

echo srun python -u train.py --no-note -v 0 -E $EPOCH --t-split $TSPLIT --v-split $VSPLIT -F ${F[@]} -B $BATCH_SIZE --n-filters $N_FILTERS --depth $DEPTH -L $L -l ${L_W[$L_W_IDX]} --color-space ${COL[$COL_IDX]} -S $SEED
srun python -u train.py --no-note -v 0 -E $EPOCH --t-split $TSPLIT --v-split $VSPLIT -F ${F[@]} -B $BATCH_SIZE --n-filters $N_FILTERS --depth $DEPTH -L $L -l ${L_W[$L_W_IDX]} --color-space ${COL[$COL_IDX]} -S $SEED
date

#!/bin/bash
#SBATCH --job-name=lfw-vggface-unet-grid        # Job name
#SBATCH --account=gts-sdas7                     # charge account
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=logasja@gatech.edu          # Where to send mail
#SBATCH --nodes=1                               # How many nodes to use
#SBATCH --mem-per-gpu=256gb                     # Memory per GPU
#SBATCH --gres=gpu:RTX_6000:1                   # Number of GPUs per node
#SBATCH -q inferno                              # QOS
#SBATCH --time=08:00:00                         # Time limit hrs:min:sec
#SBATCH --output=../batchruns/Report_%A-%a.out  # Standard output and error log
#SBATCH --array=1-5                             # Array range
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

module load anaconda3/2022.05.0.1
conda activate unet

DATASET=lfw
EPOCH=500
TSPLIT=train[:20%]
VSPLIT=train[21%:30%]
F=(vggface)
BATCH_SIZE=32
N_FILTERS=(32 64 16 32 64)
DEPTH=(3 3 5 5 5)
L=none
TB_LOGS=/storage/home/hcoda1/0/plogas3/tb-logs/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/

RUN_IDX=$(( ($SLURM_ARRAY_TASK_ID - 1) ))
echo Running task $SLURM_ARRAY_TASK_ID, which will train with ${N_FILTERS[$RUN_IDX]} starting filters and depth of ${DEPTH[$RUN_IDX]}

echo srun python -u train.py --no-note -v 0 -E $EPOCH --t-split $TSPLIT --v-split $VSPLIT -F ${F[@]} -B $BATCH_SIZE --n-filters ${N_FILTERS[$RUN_IDX]} --depth ${DEPTH[$RUN_IDX]} -L $L --log-dir $TB_LOGS
srun python -u train.py --no-note -v 0 -E $EPOCH --t-split $TSPLIT --v-split $VSPLIT -F ${F[@]} -B $BATCH_SIZE --n-filters ${N_FILTERS[$RUN_IDX]} --depth ${DEPTH[$RUN_IDX]} -L $L --log-dir $TB_LOGS

date
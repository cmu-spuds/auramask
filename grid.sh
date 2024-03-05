#!/bin/bash
DATASET=lfw
EPOCH=500
TSPLIT=train[:20%]
VSPLIT=train[21%:30%]
F=(arcface vggface)
BATCH_SIZE=32
N_FILTERS=(32 64)
DEPTH=(3 5)
NUM_JOBS=$((${#N_FILTERS[@]} + ${#DEPTH[@]}))
L=none
SERIALIZATION_DIR=/storage/home/hcoda1/0/plogas3/experiments
TB_LOGS=/storage/home/hcoda1/0/plogas3/tb-logs/
NUM_GPUS_PER_NODE=1
NUM_NODES=1
SLURM_PARTITION=gpu-rtx6000
SLURM_ACCOUNT=gts-sdas7
JOB_NAME=${DATASET}_${F}
CODE_FILE=./train.sh

CMD='
trap_handler () {
  echo "Caught signal: " $1
  # SIGTERM must be bypassed
  if [ "$1" = "TERM" ]; then
      echo "bypass sigterm"
  else
    # Submit a new job to the queue
    echo "Requeuing " $SLURM_JOB_ID
    scontrol requeue $SLURM_JOB_ID
  fi
}

# Install signal handler
trap '"'"'trap_handler USR1'"'"' USR1
trap '"'"'trap_handler TERM'"'"' TERM

'

for d in "${DEPTH[@]}"
do
  for nfil in "${N_FILTERS[@]}"
  do
    run_id=\"${d}_${nfil}\"
    CMD+='if [ "$SLURM_ARRAY_TASK_ID" ='
    CMD+=" $run_id ]; then
    srun --job-name ${JOB_NAME}/${CLUSTER_NUMBER} \
    --output ${SERIALIZATION_DIR}/${JOB_NAME}/${run_id}/train.log \
    --error ${SERIALIZATION_DIR}/${JOB_NAME}/${run_id}/train.stderr.%A_%a \
    --open-mode append --unbuffered --cpu-bind=none \
    bash ${CODE_FILE} $EPOCH $TSPLIT $VSPLIT '${F[@]}' $BATCH_SIZE $nfil $d $L $TB_LOGS \
    ; fi ; "
  done
done

echo $CMD

# mkdir ${SERIALIZATION_DIR}/${JOB_NAME}

# sbatch --job-name $JOB_NAME --gpus-per-node $NUM_GPUS_PER_NODE --nodes $NUM_NODES --ntasks-per-node $NUM_GPUS_PER_NODE --cpus-per-task 10 \
# --account $SLURM_ACCOUNT --partition $SLURM_PARTITION --time 4320 --mem 0 --open-mode append --signal B:USR1@180  \
# --output "${SERIALIZATION_DIR}/${JOB_NAME}/%a/slrm_stdout.%A_%a" --error "${SERIALIZATION_DIR}/${JOB_NAME}/%a/slrm_stderr.%A_%a" \
# --array 0-$(($NUM_JOBS-1))  \
# --wrap "$CMD"

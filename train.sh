#!/bin/bash
echo module load anaconda3/2022.05.0.1
echo conda activate unet

EPOCH=$1
TSPLIT=$2
VSPLIT=$3
F=$4
BATCH_SIZE=$5
nfil=$6
d=$7
L=$8
TB_LOGS=$9


echo python train.py --no-note -v 0 -E $EPOCH --t-split $TSPLIT --v-split $VSPLIT -F $F -B $BATCH_SIZE --n-filters $nfil --depth $d -L $L --log-dir $TB_LOGS
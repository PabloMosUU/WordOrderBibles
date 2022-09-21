#!/bin/sh
#SBATCH --time=72:00:00
#SBATCH --mem=250G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

CFG_FILENAME="$1"
CFG_NAME="$2"

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
MAIN_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles
MODEL_NAME=$CFG_NAME
echo "python ${MAIN_DIR}/train.py ${MAIN_DIR}/eng-x-bible-world.txt ${MAIN_DIR}/configs/${CFG_FILENAME} ${CFG_NAME} ${MODEL_NAME} ${MAIN_DIR}/output/ False"
python ${MAIN_DIR}/train.py ${MAIN_DIR}/eng-x-bible-world.txt ${MAIN_DIR}/configs/${CFG_FILENAME} ${CFG_NAME} ${MODEL_NAME} ${MAIN_DIR}/output/ False

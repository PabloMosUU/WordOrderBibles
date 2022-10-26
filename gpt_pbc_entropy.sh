#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH --mem=250G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

DEVICE="$1"

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
MAIN_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles
echo "python ${MAIN_DIR}/gpt_pbc_entropy.py ${DEVICE}"
python ${MAIN_DIR}/gpt_pbc_entropy.py ${DEVICE}

#!/bin/sh
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

BIBLE_FILENAME="$1"
TEMP_DIR="$2"
ENTROPIES_FILENAME="$3"

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
MAIN_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles/
echo "python3 ${MAIN_DIR}/word_splitting.py ${BIBLE_FILENAME} ${TEMP_DIR} ${ENTROPIES_FILENAME} /hpc/uu_ics_ads/pmosteiro/KoplenigEtAl/shortestmismatcher.jar"
python3 ${MAIN_DIR}/word_splitting.py ${BIBLE_FILENAME} ${TEMP_DIR} ${ENTROPIES_FILENAME} /hpc/uu_ics_ads/pmosteiro/KoplenigEtAl/shortestmismatcher.jar

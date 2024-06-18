#!/bin/sh
#SBATCH --time=05:00:00
#SBATCH --mem=250G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

BIBLE_FILENAME="$1"

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
MAIN_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles/
BIBLES_DIR=/hpc/uu_ics_ads/pmosteiro/paralleltext/bibles/corpus/
OUTPUT_DIR=${MAIN_DIR}/output/KoplenigEtAlSpace
ENTROPIES_DIR=${OUTPUT_DIR}/WordPasting/
TEMP_DIR=${ENTROPIES_DIR}/temp/
python3 ${MAIN_DIR}/compression_entropy.py ${BIBLES_DIR}/${BIBLE_FILENAME} ${TEMP_DIR} ${ENTROPIES_DIR}/entropies_${BIBLE_FILENAME}.json /hpc/uu_ics_ads/pmosteiro/KoplenigEtAl/shortestmismatcher.jar

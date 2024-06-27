#!/bin/sh
#SBATCH --time=05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

BIBLE_FILENAME="$1"
MAX_MERGES="$2"
BIBLE_ENTROPIES_FILE="entropies_${BIBLE_FILENAME}.json"
HPC_HOME=/hpc/uu_ics_ads/pmosteiro
MAIN_DIR=${HPC_HOME}/WordOrderBibles/
BIBLE_DIR=${HPC_HOME}/paralleltext/bibles/corpus
JAR_FILE=${HPC_HOME}/KoplenigEtAl/shortestmismatcher.jar
OUTPUT_DIR=${MAIN_DIR}/output/KoplenigEtAl/WordSplitting
TEMP_DIR=${OUTPUT_DIR}/temp

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
cd ${MAIN_DIR}
echo "python word_splitting.py ${BIBLE_DIR}/${BIBLE_FILENAME} ${TEMP_DIR} ${OUTPUT_DIR}/${BIBLE_ENTROPIES_FILE} ${JAR_FILE} ${MAX_MERGES}"
python word_splitting.py ${BIBLE_DIR}/${BIBLE_FILENAME} ${TEMP_DIR} ${OUTPUT_DIR}/${BIBLE_ENTROPIES_FILE} ${JAR_FILE} ${MAX_MERGES}


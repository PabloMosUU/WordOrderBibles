#!/bin/sh
#SBATCH --time=05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=p.mosteiro@uu.nl

BIBLE_FILENAME="$1"
TEMP_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/KoplenigEtAl/WordSplitting/temp

source /hpc/uu_ics_ads/anaconda3/etc/profile.d/conda.sh
conda activate word_order_bibles
MAIN_DIR=/hpc/uu_ics_ads/pmosteiro/WordOrderBibles/
cd ${MAIN_DIR}
echo "./30_run_word_splitting.sh ${BIBLE_FILENAME} /hpc/uu_ics_ads/pmosteiro/paralleltext/bibles/corpus /hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/KoplenigEtAl/WordSplitting/temp /hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/KoplenigEtAl/WordSplitting /hpc/uu_ics_ads/pmosteiro/KoplenigEtAl/shortestmismatcher.jar"
./30_run_word_splitting.sh ${BIBLE_FILENAME} /hpc/uu_ics_ads/pmosteiro/paralleltext/bibles/corpus /hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/KoplenigEtAl/WordSplitting/temp /hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/KoplenigEtAl/WordSplitting /hpc/uu_ics_ads/pmosteiro/KoplenigEtAl/shortestmismatcher.jar

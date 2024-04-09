BIBLE_FILE=$1
BIBLE_DIR=$2
TEMP_DIR=$3
OUTPUT_DIR=$4
JAR_FILE=$5
BIBLE_ENTROPIES_FILE="entropies_${BIBLE_FILE}.json"
COMPLETED_ENTROPIES_FILE="${OUTPUT_DIR}/completed_entropies.txt"
if grep -Fxq "${BIBLE_ENTROPIES_FILE}" ${COMPLETED_ENTROPIES_FILE}
then
    echo "found; skipping ${BIBLE_FILE}"
else
    echo "not found; will run word splitting on ${BIBLE_FILE}"
    echo "python word_splitting.py ${BIBLE_DIR}/${BIBLE_FILE} ${TEMP_DIR} ${OUTPUT_DIR}/${BIBLE_ENTROPIES_FILE} ${JAR_FILE} 10000"
    python word_splitting.py ${BIBLE_DIR}/${BIBLE_FILE} ${TEMP_DIR} ${OUTPUT_DIR}/${BIBLE_ENTROPIES_FILE} ${JAR_FILE} 10000
    if [ -f "${OUTPUT_DIR}/${BIBLE_ENTROPIES_FILE}" ]; then
	echo "${BIBLE_ENTROPIES_FILE}" >> ${COMPLETED_ENTROPIES_FILE}
    fi
fi

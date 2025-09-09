GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="create_in_memory_dataset_blob"
DATASET_NAME="${1:-sss2-1b_1p}"

SCRIPT="scripts/${EXPERIMENT_NAME}.sh"
LOG_DIR="/nfs/team361/am84/VQNiche/logs/${DATASET_NAME}/${EXPERIMENT_NAME}"
CONFIG_FILE="config/${EXPERIMENT_NAME}/${DATASET_NAME}.yaml"
JOB_NAME="${EXPERIMENT_NAME}_${DATASET_NAME}"

echo "Log directory: ${LOG_DIR}"

case $DATASET_NAME in
    mmb0-4b_1p)
        RAM="25G"
        TIME="0:30"
        NUM_GPUS=2
        CORES=4
        QUEUE="gpu-lotfollahi"
        ;;
    sss2-1b_1p)
        RAM="10G"
        TIME="0:30"
        NUM_GPUS=1
        CORES=1
        QUEUE="gpu-lotfollahi"
        ;;
    xhs1000-39b_1p)
        RAM="40G"
        TIME="4:00"
        NUM_GPUS=2
        CORES=8
        QUEUE="gpu-lotfollahi"
        ;;
    xhk1020-CV1-CV2-5b_1p)
        RAM="40G"
        TIME="1:00"
        NUM_GPUS=1
        CORES=8
        QUEUE="gpu-lotfollahi"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"

# Set the output and error log files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.err"

echo "Creating an in-memory Dataset-Blob of ${DATASET_NAME} using ${CORES} cores..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -gpu "num=${NUM_GPUS}:mode=exclusive_process:block=yes" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}"
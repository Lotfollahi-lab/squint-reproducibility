GROUP="team361"

EXPERIMENT_NAME="sweep_train_model"
SCRIPT="scripts/${EXPERIMENT_NAME}.sh"

# Set the experiment parameters
DATASET_NAME="${1:-sss2-1b_1p}"
SPLIT_NAME="${2:-1-test-patch-split}"
MODEL_NAME="${3:-vqniche_graphsage}"
SWEEP_NAME="${4:-backbone_gnn}"

LOG_DIR="/nfs/team361/am84/VQNiche/logs/${DATASET_NAME}/sweep/job_log/${MODEL_NAME}/${SWEEP_NAME}"

BASE_CONFIG_FILE="config/train_model/${DATASET_NAME}_${SPLIT_NAME}_${MODEL_NAME}.yaml"
SWEEP_CONFIG_FILE="config/sweeps/${SWEEP_NAME}.yaml"
JOB_NAME="${DATASET_NAME}_${SPLIT_NAME}_${MODEL_NAME}_${SWEEP_NAME}"

echo "Log directory: ${LOG_DIR}"

case $DATASET_NAME in
    mmb0-4b_1p)
        RAM="25G"
        TIME="0:30"
        NUM_GPUS=1
        CORES=4
        QUEUE="gpu-lotfollahi"
        ;;
    sss2-1b_1p)
        RAM="20G"
        TIME="0:30"
        NUM_GPUS=1
        CORES=1
        QUEUE="gpu-lotfollahi"
        ;;
    xhs1000-39b_1p-oriented-3)
        RAM="40G"
        TIME="1:00"
        NUM_GPUS=2
        CORES=8
        QUEUE="gpu-lotfollahi"
        ;;
    xhk1020-CV1-CV2-5b_1p)
        RAM="40G"
        TIME="2:00"
        NUM_GPUS=1
        CORES=8
        QUEUE="gpu-lotfollahi"
        ;;
    xhs1021-15b_1p)
        RAM="40G"
        TIME="2:00"
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
CORES="${5:-$CORES}"
QUEUE="${6:-$QUEUE}"

# Set the output and error log files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.err"

echo "Training ${MODEL_NAME} on ${DATASET_NAME} with ${SPLIT_NAME} using ${CORES} cores..."

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
    "${SCRIPT}" "${BASE_CONFIG_FILE}" "${SWEEP_CONFIG_FILE}"
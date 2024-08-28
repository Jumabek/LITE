#!/bin/bash

INPUT_RESOLUTION="1280"
MIN_CONFIDENCE=".25"
DATASET=""
SPLIT=""

usage() {
    echo "Usage: $0 -d DATASET -s SPLIT"
    echo "  -d DATASET   Dataset name (e.g., MOT17)"
    echo "  -s SPLIT     Split (e.g., train)"
    exit 1
}

# Parse command-line options
while getopts "d:s:" opt; do
    case ${opt} in
        d)
            DATASET=${OPTARG}
            ;;
        s)
            SPLIT=${OPTARG}
            ;;
        \?)
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if all mandatory variables are set
if [ -z "$DATASET" ] || [ -z "$SPLIT" ]; then
    echo "Error: Missing required arguments."
    usage
fi

BASE_CMD="python3 run.py --dataset ${DATASET} --split ${SPLIT}"

run_tracker() {
    TRACKER_NAME=$1
    echo "-----------------------------------"
    echo "Running tracker: ${TRACKER_NAME}"

    DIR_SAVE="results/${DATASET}/${TRACKER_NAME}__input_${INPUT_RESOLUTION}__conf_${MIN_CONFIDENCE}"
    mkdir -p "${DIR_SAVE}"

    CMD_OPTIONS="--dir_save ${DIR_SAVE} --input_resolution ${INPUT_RESOLUTION} --min_confidence ${MIN_CONFIDENCE}"

    case ${TRACKER_NAME} in
        "SORT")
            ${BASE_CMD} --tracker_name "SORT" ${CMD_OPTIONS}
            ;;
        "LITEDeepSORT")
            ${BASE_CMD} --tracker_name "LITEDeepSORT" ${CMD_OPTIONS} --woC --appearance_feature_layer "layer0"
            ;;
        "DeepSORT")
            ${BASE_CMD} --tracker_name "DeepSORT" ${CMD_OPTIONS}
            ;;
        "StrongSORT")
            ${BASE_CMD} --tracker_name "StrongSORT" ${CMD_OPTIONS} --BoT --ECC --NSA --EMA --MC --woC
            ;;
        *)
            echo "Invalid tracker name"
            exit 1
            ;;
    esac
    echo "Experiment completed for ${TRACKER_NAME}!"
}

# TRACKERS=("SORT" "LITEDeepSORT" "DeepSORT" "StrongSORT")
TRACKERS=("StrongSORT")

for TRACKER in "${TRACKERS[@]}"; do
    run_tracker "${TRACKER}"
done

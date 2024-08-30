#!/bin/bash

INPUT_RESOLUTION="1280"
CONFIDENCES=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)   
DATASET=""
SPLIT=""

usage() {
    echo "Usage: $0 -d DATASET -s SPLIT"
    echo "  -d DATASET   Dataset name (e.g., MOT17)"
    echo "  -s SPLIT     Split (e.g., train)"
    echo "  -q SEQUENCE   Sequence name (e.g., 'MOT17-02-FRCNN')"
    exit 1
}

# Parse command-line options
while getopts "d:s:q:" opt; do
    case ${opt} in
        d)
            DATASET=${OPTARG}
            ;;
        s)
            SPLIT=${OPTARG}
            ;;
        q)
            SEQUENCE=${OPTARG}
            ;;
        \?)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

# Check if all mandatory variables are set
if [ -z "$DATASET" ] || [ -z "$SPLIT" ] || [ -z "$SEQUENCE" ]; then
    echo "Error: Missing required arguments."
    usage
fi

run_tracker() {
    CMD="python3 run_fps.py \
        --sequence ${SEQUENCE} \
        --dataset ${DATASET} \
        --split ${SPLIT} \
        --input_resolution ${INPUT_RESOLUTION} \
        --min_confidence ${MIN_CONFIDENCE}"

    CMD_YOLO_TRACKING="python3 yolo_tracking/tracking/run_fps.py \
        --sequence ${SEQUENCE} \
        --dataset ${DATASET} \
        --split ${SPLIT} \
        --imgsz ${INPUT_RESOLUTION} \
        --conf ${MIN_CONFIDENCE}"
        
    TRACKER_NAME=$1
    MIN_CONFIDENCE=$2
    echo "-----------------------------------"
    echo "Running tracker: ${TRACKER_NAME}"

    DIR_SAVE="results/${DATASET}-FPS/${SEQUENCE}/${TRACKER_NAME}__input_${INPUT_RESOLUTION}__conf_${MIN_CONFIDENCE}"
    mkdir -p "${DIR_SAVE}" || { echo "Error creating directory ${DIR_SAVE}"; exit 1; }

    case ${TRACKER_NAME} in
        "SORT")
            ${CMD} --tracker_name "SORT" --dir_save "${DIR_SAVE}"
            ;;
        "LITEDeepSORT")
            ${CMD} --tracker_name "LITEDeepSORT" --woC --appearance_feature_layer "layer0" --dir_save "${DIR_SAVE}"
            ;;
        "DeepSORT")
            ${CMD} --tracker_name "DeepSORT" --dir_save "${DIR_SAVE}"
            ;;
        "StrongSORT")
            ${CMD} --tracker_name "StrongSORT" --BoT --ECC --NSA --EMA --MC --woC --dir_save "${DIR_SAVE}"
            ;;
        "OCSORT")
            ${CMD_YOLO_TRACKING} --tracking-method "ocsort" --project "${DIR_SAVE}"
            ;;
        "Bytetrack")
            ${CMD_YOLO_TRACKING} --tracking-method "bytetrack" --project "${DIR_SAVE}"
            ;;
        "DeepOCSORT")
            ${CMD_YOLO_TRACKING} --tracking-method "deepocsort" --project "${DIR_SAVE}"
            ;;
        "LITEDeepOCSORT")
            ${CMD_YOLO_TRACKING} --tracking-method "deepocsort" --project "${DIR_SAVE}" --appearance-feature-layer "layer0"
            ;;
        "BoTSORT")
            ${CMD_YOLO_TRACKING} --tracking-method "botsort" --project "${DIR_SAVE}"
            ;;
        "LITEBoTSORT")
            ${CMD_YOLO_TRACKING} --tracking-method "botsort" --project "${DIR_SAVE}" --appearance-feature-layer "layer0"
            ;;
        *)
            echo "Invalid tracker name"
            exit 1
            ;;
    esac
    echo "Experiment completed for ${TRACKER_NAME}!"
}

TRACKERS=("SORT" "LITEDeepSORT" "DeepSORT" "StrongSORT" "OCSORT" "Bytetrack" "DeepOCSORT" "LITEDeepOCSORT" "BoTSORT" "LITEBoTSORT")

for MIN_CONFIDENCE in "${CONFIDENCES[@]}"; do
    for TRACKER in "${TRACKERS[@]}"; do
        run_tracker "${TRACKER}" "${MIN_CONFIDENCE}"
    done
done

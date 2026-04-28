#!/bin/bash
set -euo pipefail

DATASET="MOT20"
SPLIT="train"
TRACKER="LITEStrongSORT"
YOLO_MODEL="yolov8m"
INPUT_RESOLUTION="1280"
CONFIDENCE="0.25"
DIR_ROOT="results/paper"

usage() {
    local status="${1:-1}"
    echo "Usage: $0 [-d DATASET] [-s SPLIT] [-t TRACKER] [-m YOLO_MODEL] [-r INPUT_RESOLUTION] [-c CONFIDENCE] [-o DIR_ROOT]"
    echo "  -d DATASET           Dataset name, e.g. MOT17 or MOT20 (default: ${DATASET})"
    echo "  -s SPLIT             Split, e.g. train or test (default: ${SPLIT})"
    echo "  -t TRACKER           Tracker name (default: ${TRACKER})"
    echo "  -m YOLO_MODEL        YOLO model stem, e.g. yolov8m or yolo11m (default: ${YOLO_MODEL})"
    echo "  -r INPUT_RESOLUTION  Input image size (default: ${INPUT_RESOLUTION})"
    echo "  -c CONFIDENCE        Detection confidence threshold (default: ${CONFIDENCE})"
    echo "  -o DIR_ROOT          Output root directory (default: ${DIR_ROOT})"
    echo
    echo "Tracker options: SORT, LITEDeepSORT, DeepSORT, StrongSORT, LITEStrongSORT,"
    echo "                 OCSORT, Bytetrack, DeepOCSORT, LITEDeepOCSORT, BoTSORT, LITEBoTSORT"
    exit "${status}"
}

while getopts "d:s:t:m:r:c:o:h" opt; do
    case "${opt}" in
        d) DATASET=${OPTARG} ;;
        s) SPLIT=${OPTARG} ;;
        t) TRACKER=${OPTARG} ;;
        m) YOLO_MODEL=${OPTARG} ;;
        r) INPUT_RESOLUTION=${OPTARG} ;;
        c) CONFIDENCE=${OPTARG} ;;
        o) DIR_ROOT=${OPTARG} ;;
        h) usage 0 ;;
        \?) usage ;;
    esac
done

DIR_SAVE="${DIR_ROOT}/${DATASET}-${SPLIT}/${TRACKER}__input_${INPUT_RESOLUTION}__conf_${CONFIDENCE}__model_${YOLO_MODEL}"
mkdir -p "${DIR_SAVE}"

CMD=(
    python3 run.py
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --input_resolution "${INPUT_RESOLUTION}"
    --min_confidence "${CONFIDENCE}"
    --dir_save "${DIR_SAVE}"
    --yolo_model "${YOLO_MODEL}"
)

CMD_YOLO_TRACKING=(
    python3 yolo_tracking/tracking/run.py
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --imgsz "${INPUT_RESOLUTION}"
    --conf "${CONFIDENCE}"
    --project "${DIR_SAVE}"
    --yolo-model "${YOLO_MODEL}"
)

echo "Running ${TRACKER} on ${DATASET}-${SPLIT}"
echo "Output: ${DIR_SAVE}"

case "${TRACKER}" in
    "SORT")
        "${CMD[@]}" --tracker_name "SORT"
        ;;
    "LITEDeepSORT")
        "${CMD[@]}" --tracker_name "LITEDeepSORT" --woC --appearance_feature_layer "layer14"
        ;;
    "DeepSORT")
        "${CMD[@]}" --tracker_name "DeepSORT"
        ;;
    "StrongSORT")
        "${CMD[@]}" --tracker_name "StrongSORT" --BoT --ECC --NSA --EMA --MC --woC
        ;;
    "LITEStrongSORT")
        "${CMD[@]}" --tracker_name "LITEStrongSORT" --BoT --ECC --NSA --EMA --MC --woC --appearance_feature_layer "layer14"
        ;;
    "OCSORT")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "ocsort"
        ;;
    "Bytetrack")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "bytetrack"
        ;;
    "DeepOCSORT")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "deepocsort"
        ;;
    "LITEDeepOCSORT")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "deepocsort" --appearance-feature-layer "layer14"
        ;;
    "BoTSORT")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "botsort"
        ;;
    "LITEBoTSORT")
        "${CMD_YOLO_TRACKING[@]}" --tracking-method "botsort" --appearance-feature-layer "layer14"
        ;;
    *)
        echo "Invalid tracker name: ${TRACKER}"
        usage
        ;;
esac

echo "Experiment completed."

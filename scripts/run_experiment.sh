#!/bin/bash

# Resolutions and other constants
INPUT_RESOLUTIONS=(1280)
<<<<<<< Updated upstream
CONFIDENCE_LEVELS=(0.25)
DATASETS=("MOT20")
SPLIT="train"
MODELS=('yolov8m')

=======
CONFIDENCE_LEVELS=(0.01)
DATASETS=("MOT20")
SPLIT="train"
MODELS=('yolov8m')
>>>>>>> Stashed changes

#n' 'ablation_17s
usage() {
    echo "Usage: $0 -s SPLIT"
    echo "  -s SPLIT     Split (e.g., train)"
    exit 1
}

# Parse command-line options
while getopts "s:" opt; do
    case ${opt} in
        s)
            SPLIT=${OPTARG}
            ;;
        \?)
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if mandatory variables are set
if [ -z "$SPLIT" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Loop through datasets
for DATASET in "${DATASETS[@]}"
do
    for INPUT_RESOLUTION in "${INPUT_RESOLUTIONS[@]}"
    do
        for CONFIDENCE in "${CONFIDENCE_LEVELS[@]}"
        do # chnage to run_parallel.py if you want to run multiple videos in parallel
           CMD="python3 run.py \
                    --dataset ${DATASET} \
                    --split ${SPLIT} \
                    --input_resolution ${INPUT_RESOLUTION} \
                    --min_confidence ${CONFIDENCE}"

            CMD_YOLO_TRACKING="python3 yolo_tracking/tracking/run.py \
                    --dataset ${DATASET} \
                    --split ${SPLIT} \
                    --imgsz ${INPUT_RESOLUTION} \
                    --conf ${CONFIDENCE}"
            # print CMD_YOLO_TRACKING
            echo $CMD

            run_tracker() {
                TRACKER_NAME=$1
                YOLO_MODEL=$2
                echo "-----------------------------------"
                echo "Running tracker: ${TRACKER_NAME} with YOLO model: ${YOLO_MODEL} at confidence: ${CONFIDENCE}"

<<<<<<< Updated upstream
                DIR_SAVE="results/fps-exp/${DATASET}-${SPLIT}/${TRACKER_NAME}__input_${INPUT_RESOLUTION}__conf_${CONFIDENCE}__model_${YOLO_MODEL}"
=======
                DIR_SAVE="results/exp_conf_0.01/model/${DATASET}-${SPLIT}/${TRACKER_NAME}__input_${INPUT_RESOLUTION}__conf_${CONFIDENCE}__model_${YOLO_MODEL}"
>>>>>>> Stashed changes
                mkdir -p "${DIR_SAVE}"

                case ${TRACKER_NAME} in
                    "SORT")
                        ${CMD} --tracker_name "SORT" --dir_save ${DIR_SAVE} --yolo_model ${YOLO_MODEL}
                        ;;
                    "LITEDeepSORT")
                        ${CMD} --tracker_name "LITEDeepSORT" --woC --appearance_feature_layer "layer14" --dir_save ${DIR_SAVE} --yolo_model ${YOLO_MODEL}
                        ;;
                    "DeepSORT")
                        ${CMD} --tracker_name "DeepSORT" --dir_save ${DIR_SAVE} --yolo_model ${YOLO_MODEL}
                        ;;
                    "StrongSORT")
                        ${CMD} --tracker_name "StrongSORT" --BoT --ECC --NSA --EMA --MC --woC --dir_save ${DIR_SAVE} --yolo_model ${YOLO_MODEL}
                        ;;
                    "LITEStrongSORT")
                        ${CMD} --tracker_name "LITEStrongSORT" --BoT --ECC --NSA --EMA --MC --woC --dir_save ${DIR_SAVE} --yolo_model ${YOLO_MODEL} --appearance_feature_layer "layer14"
                        ;;
                    "OCSORT")
                        ${CMD_YOLO_TRACKING} --tracking-method "ocsort" --project ${DIR_SAVE}
                        ;;
                    "Bytetrack")
                        ${CMD_YOLO_TRACKING} --tracking-method "bytetrack" --project ${DIR_SAVE}
                        ;;
                    "DeepOCSORT")
                        ${CMD_YOLO_TRACKING} --tracking-method "deepocsort" --project ${DIR_SAVE}
                        ;;
                    "LITEDeepOCSORT")
                        ${CMD_YOLO_TRACKING} --tracking-method "deepocsort" --project ${DIR_SAVE} --appearance-feature-layer "layer14" --yolo-model ${YOLO_MODEL}
                        ;;
                    "BoTSORT")
                        ${CMD_YOLO_TRACKING} --tracking-method "botsort" --project ${DIR_SAVE} --yolo-model ${YOLO_MODEL}
                        ;;
                    "LITEBoTSORT")  
                        ${CMD_YOLO_TRACKING} --tracking-method "botsort" --project ${DIR_SAVE} --appearance-feature-layer "layer14" --yolo-model ${YOLO_MODEL}
                        ;;
                    *)
                        echo "Invalid tracker name"
                        exit 1
                        ;;
                esac
                echo "Experiment completed for ${TRACKER_NAME} with YOLO model: ${YOLO_MODEL} at confidence: ${CONFIDENCE}!"
            }

            # Loop through models and trackers
<<<<<<< Updated upstream
            TRACKERS=("LITEStrongSORT")

=======
            #TRACKERS=("SORT" )
            #TRACKERS=("LITEBoTSORT" "LITEDeepOCSORT")
            #TRACKERS=('Bytetrack' 'OCSORT' 'LITEDeepSORT' '')
            #TRACKERS=("DeepOCSORT" "LITEDeepOCSORT" "BoTSORT" "LITEBoTSORT")
            TRACKERS=("LITEDeepSORT")
>>>>>>> Stashed changes
            for YOLO_MODEL in "${MODELS[@]}"; do
                for TRACKER in "${TRACKERS[@]}"; do
                    run_tracker "${TRACKER}" "${YOLO_MODEL}"
                done
            done
        done
    done
done
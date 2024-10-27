#!/bin/bash

# Hardcoded Variables
EXPERIMENT_NAME="scenarios"
DATASET="MOT20"

# Base Command
BASE_CMD="python run_parallel.py --dataset ${DATASET} --split train"

# Function to run tracker with timing
run_tracker() {
    TRACKER_NAME=$1
    echo "-----------------------------------"
    echo "Running tracker: ${TRACKER_NAME} with Input Resolution: ${INPUT_RESOLUTION} and Confidence Threshold: ${MIN_CONFIDENCE}"  # Debug message

    DIR_SAVE="results/${EXPERIMENT_NAME}/${DATASET}-train/${TRACKER_NAME}__input_${INPUT_RESOLUTION}__conf_${MIN_CONFIDENCE}/data/"
    if [ ! -d "${DIR_SAVE}" ]; then
        mkdir -p ${DIR_SAVE}
    fi

    CMD=""
    case ${TRACKER_NAME} in
        "SORT")
            CMD="${BASE_CMD} --tracker_name SORT --dir_save ${DIR_SAVE} --input_resolution ${INPUT_RESOLUTION} --min_confidence ${MIN_CONFIDENCE}"
            ;;
        "LITEDeepSORT")
            CMD="${BASE_CMD} --tracker_name LITEDeepSORT --dir_save ${DIR_SAVE} --input_resolution ${INPUT_RESOLUTION} --min_confidence ${MIN_CONFIDENCE} --appearance_feature_layer layer0 --woC"
            ;;
        "DeepSORT")
            CMD="${BASE_CMD} --tracker_name DeepSORT --dir_save ${DIR_SAVE} --input_resolution ${INPUT_RESOLUTION} --min_confidence ${MIN_CONFIDENCE}"
            ;;
        "StrongSORT")
            CMD="${BASE_CMD} --tracker_name StrongSORT --dir_save ${DIR_SAVE} --input_resolution ${INPUT_RESOLUTION} --min_confidence ${MIN_CONFIDENCE} --BoT --ECC --NSA --EMA --MC --woC"
            ;;
        *)
            echo "Invalid tracker name"
            exit 1
            ;;
    esac

    echo "Executing: ${CMD}"  # Print command
    START_TIME=$(date +%s)  # Capture start time
    ${CMD}  # Execute the command
    END_TIME=$(date +%s)  # Capture end time

    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "Experiment completed for ${TRACKER_NAME} in ${ELAPSED_TIME} seconds!"
}


# Define the resolutions and confidence thresholds you want to iterate over
RESOLUTIONS=(640 960 1280 1600 1920 2240)
CONFIDENCES=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)
# continue from 640 at 0.15 confidence
# Iterate over multiple resolutions and confidence thresholds
for MIN_CONFIDENCE in "${CONFIDENCES[@]}"; do
    for INPUT_RESOLUTION in "${RESOLUTIONS[@]}"; do
        # Run experiments for all trackers
        run_tracker "SORT"        
        run_tracker "DeepSORT" # first seq: time: 75s | Avg FPS: 5.7
        run_tracker "LITEDeepSORT"
        run_tracker "StrongSORT"
    done
done

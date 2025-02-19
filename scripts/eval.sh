#!/bin/bash

BENCHMARK="MOT20" 

# Base Command
BASE_CMD="python TrackEval/scripts/run_mot_challenge.py"

# Parent directory where all trackers' results are stored
PARENT_TRACKERS_FOLDER="results/exp_conf_0.01/model"

echo "Starting the evaluation script..."

# Debug messages
echo "-------------------------------------------"
echo "Processing resolution: ${res}, and confidence: ${conf}"

# Construct the command 
EVAL_CMD="${BASE_CMD} \
            --BENCHMARK ${BENCHMARK} \
            --TRACKERS_FOLDER ${PARENT_TRACKERS_FOLDER} \
            --SPLIT_TO_EVAL train \
            --METRICS HOTA CLEAR Identity VACE \
            --USE_PARALLEL True \
            --NUM_PARALLEL_CORES 16 \
            --GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt.txt' \
            --OUTPUT_SUMMARY True \
            --OUTPUT_EMPTY_CLASSES False \
            --OUTPUT_DETAILED True \
            --PLOT_CURVES True"

# Print the evaluation command
echo "Evaluation command: ${EVAL_CMD}"

# Run the evaluation for the current combination of resolution and confidence
eval $EVAL_CMD #> "${PARENT_TRACKERS_FOLDER}/evaluation_output_res_${res}_conf_${conf}.txt"

echo "Evaluation script completed."

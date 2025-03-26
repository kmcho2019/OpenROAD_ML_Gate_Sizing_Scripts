#!/bin/bash

# Define the log file
#LOG_FILE="gate_sizing_logs.txt"
#LOG_FILE="mempool_gate_sizing_logs.txt"
LOG_FILE="model_based_sizing_results_20250318.log"

# Define the designs to be run
DESIGNS=("NV_NVDLA_partition_m" "NV_NVDLA_partition_p" "ariane136" "aes_256" "mempool_tile_wrap")
# DESIGNS=("mempool_tile_wrap")

# Create or clear the log file
> "$LOG_FILE"

# Loop through each design
for design_name in "${DESIGNS[@]}"; do
  echo "Running design: $design_name" | tee -a "$LOG_FILE"

  # Construct the command
  #command="../OpenROAD_ML_Gate_Sizing/g7_build/src/openroad -python getEndpointAndCriticalPaths_debug.py --design_name ${design_name} -exit"
  command="../OpenROAD_ML_Gate_Sizing/build/src/openroad -python getEndpointAndCriticalPaths_debug_with_postsize_eval.py --design_name ${design_name} -exit"

  # Execute the command and capture both stdout and stderr, redirecting to the log file
  { $command 2>&1; } | tee -a "$LOG_FILE"

  echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "Finished running all designs. Logs are saved in: $LOG_FILE"

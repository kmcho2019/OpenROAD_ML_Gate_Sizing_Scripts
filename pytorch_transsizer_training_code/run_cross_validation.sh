#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_SCRIPT="train_model.py" # CHANGE THIS if your python script has a different name
EPOCHS=30
LR="1e-3"
WARMUP_RATIO=0.2
OUTPUT_PARAMS="transformer_params.bin"
OUTPUT_PLOT="training_plot.png"

# Define the base directories in an array
declare -a base_dirs=(
    "./train_data/backup_20250327/aes_256"
    "./train_data/backup_20250327/ariane136"
    "./train_data/backup_20250327/mempool_tile_wrap"
    "./train_data/backup_20250327/NV_NVDLA_partition_m"
    "./train_data/backup_20250327/NV_NVDLA_partition_p"
)

# Get the total number of directories
num_dirs=${#base_dirs[@]}

echo "Starting cross-validation..."
echo "Total designs: $num_dirs"
echo "-------------------------------------"

# Loop through each directory, using it once as the validation set
for (( i=0; i<${num_dirs}; i++ )); do
    # Select the validation directory
    val_dir=${base_dirs[$i]}
    design_name=$(basename "$val_dir") # Extract the design name (e.g., aes_256)

    echo "Running validation for: $design_name (Run $((i+1))/$num_dirs)"

    # Build the list of training directories (all except the current validation one)
    train_dirs=()
    for (( j=0; j<${num_dirs}; j++ )); do
        if [[ $i -ne $j ]]; then
            train_dirs+=("${base_dirs[$j]}")
        fi
    done

    echo "  Validation set: $val_dir"
    echo "  Training sets: ${train_dirs[@]}"
    echo "  Running Python script..."

    # Construct and run the python command
    # Note: Using "${train_dirs[@]}" ensures paths with spaces are handled correctly
    python "$PYTHON_SCRIPT" \
        --train_base_dir "${train_dirs[@]}" \
        --val_base_dir "$val_dir" \
        --num_epochs $EPOCHS \
        --lr $LR \
        --warmup_ratio $WARMUP_RATIO \
        # --batch_size can be added here if needed, otherwise uses default from python script

    echo "  Python script finished for $design_name."

    # Rename the output files, checking if they exist first
    echo "  Renaming output files..."
    if [ -f "$OUTPUT_PARAMS" ]; then
        new_params_name="validation_${design_name}_${OUTPUT_PARAMS}"
        cp "$OUTPUT_PARAMS" "$new_params_name"
        echo "    Renamed $OUTPUT_PARAMS to $new_params_name"
    else
        echo "    Warning: $OUTPUT_PARAMS not found after training with $design_name."
    fi

    if [ -f "$OUTPUT_PLOT" ]; then
        new_plot_name="validation_${design_name}_${OUTPUT_PLOT}"
        cp "$OUTPUT_PLOT" "$new_plot_name"
        echo "    Renamed $OUTPUT_PLOT to $new_plot_name"
    else
        echo "    Warning: $OUTPUT_PLOT not found after training with $design_name."
    fi

    echo "-------------------------------------"

done

echo "Cross-validation finished."
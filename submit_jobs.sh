#!/bin/bash
#SBATCH --account=smearlab
#SBATCH --job-name=PGAM
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-11
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sidr@uoregon.edu



# Print start time and allocated resources
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Partition: $SLURM_JOB_PARTITION"

# Define mice, targets, and sessions
MICE=("6000" "6001" "6002" "6003")
SESSIONS=("1" "2" "24")




# Dimensions
NUM_MICE=${#MICE[@]}
NUM_SESSIONS=${#SESSIONS[@]}

# Total combinations per dimension
TOTAL_COMBINATIONS=$((NUM_MICE * NUM_SESSIONS))

# Get indices
MOUSE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SESSIONS))
SESSION_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SESSIONS))

# Get values for this array job
MOUSE=${MICE[$MOUSE_INDEX]}
SESSION=${SESSIONS[$SESSION_INDEX]}

# Define the save directory for figures
SAVE_DIR="/projects/smearlab/shared/clickbait-ephys(3-20-25)/figures/PGAM/6ms_good"
DATA_DIR="/projects/smearlab/shared/clickbait-ephys(3-20-25)"

# Print to log
echo "Running decoding for mouse $MOUSE in session $SESSION on node $SLURM_NODELIST"

# Run the decoding script
python fit_pgam.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --mouse $MOUSE \
    --session $SESSION \
    --window_size 0.006 \
    --window_step 0.006 \
    --use_units 'good' \
    --order 4 \
    --frac_eval 0.05

echo "Job finished at: $(date)"

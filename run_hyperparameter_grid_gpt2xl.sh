#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=4G    # memory per cpu-core
#SBATCH -t 1:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-1999       # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /om2/user/zaho/private_llm_bm/reports/slurm-%A_%a.out # STDOUT
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

# Define arrays for each hyperparameter
n_top_pc_llm_array=(100 400 800 -1)
weight_decay_array=(0.0 0.001 0.005 0.01 0.02)
convolve_size_array=(1 4 8 16)
n_delay_embedding_llm_array=(1 2 4 8 16)
llm_model_index_array=(0 1)

# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
n_top_pc_llm_index=$((index % 4))
index=$((index / 4))
weight_decay_index=$((index % 5))
index=$((index / 5))
convolve_size_index=$((index % 4))
index=$((index / 4))
n_delay_embedding_llm_index=$((index % 5))
index=$((index / 5))
random_index=$((index % 5))

# Get the actual parameter values
n_top_pc_llm=${n_top_pc_llm_array[$n_top_pc_llm_index]}
weight_decay=${weight_decay_array[$weight_decay_index]}
convolve_size=${convolve_size_array[$convolve_size_index]}
n_delay_embedding_llm=${n_delay_embedding_llm_array[$n_delay_embedding_llm_index]}
llm_model_index=0

echo "Running with parameters:"
echo "n_top_pc_llm: $n_top_pc_llm"
echo "weight_decay: $weight_decay"
echo "convolve_size: $convolve_size"
echo "n_delay_embedding_llm: $n_delay_embedding_llm"
echo "llm_model_index: $llm_model_index"

# Run the Python script with the selected parameters
python train_analyze_fmri_potter.py \
    --n_top_pc_llm $n_top_pc_llm \
    --weight_decay $weight_decay \
    --convolve_size $convolve_size \
    --n_delay_embedding_llm $n_delay_embedding_llm \
    --llm_model_index $llm_model_index \
    --random $random_index
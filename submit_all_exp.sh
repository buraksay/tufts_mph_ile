#!/bin/bash
#SBATCH --job-name=tufts_mph_ile_xgboost
#SBATCH --output=tufts_mph_ile_xgboost_%A_%a.out
#SBATCH --error=tufts_mph_ile_xgboost_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu,preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=24:00:00

#/cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/results
module purge
module load miniforge
module load cuda
module load memforge
module load glib
module load gcc

# Activate your conda environment
source activate
conda activate nlp4ori
conda info
# Set up environment variables
#export PYTHONPATH=$PYTHONPATH:/path/to/your/code


# export WANDB_PROJECT=bert-finetuning-temporal-split-nlp4ori
# export WANDB_NAME="SLURM_JOB_${SLURM_JOB_ID}_TRIAL_${SLURM_ARRAY_TASK_ID}"
# export WANDB_API_KEY="bd54b12223aa5e3c4cfc59aaf9a15dfb7084ce51"

# Run the Python script, passing the SLURM array task ID as the trial number
# srun python /cluster/home/bsay01/aphaproject/NLP-for-ORI/training_demo/bert_finetuning_script.py --trial ${SLURM_ARRAY_TASK_ID}
#srun python /cluster/home/bsay01/aphaproject/NLP-for-ORI/training_demo/bert_finetuning_script.py --trial ${SLURM_ARRAY_TASK_ID}
srun python /cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/runner/all_tasks.py --file /cluster/tufts/shresthaapha/bsay01/data/nlp_experiment_matrix_xgb_only.csv

conda deactivate

#!/bin/bash
#SBATCH --job-name=bert_finetune
#SBATCH --output=bert_finetune_%A_%a.out
#SBATCH --error=bert_finetune_%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=gpu,preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=24:00:00
#SBATCH --array=0-79  # Run 20 jobs, numbered 0 to 19

module purge
module load miniforge/25.3.0-3

# Activate your conda environment
source activate nlp4ori2

# Set up environment variables
#export PYTHONPATH=$PYTHONPATH:/path/to/your/code


# export WANDB_PROJECT=bert-finetuning-temporal-split-nlp4ori
# export WANDB_NAME="SLURM_JOB_${SLURM_JOB_ID}_TRIAL_${SLURM_ARRAY_TASK_ID}"
# export WANDB_API_KEY="bd54b12223aa5e3c4cfc59aaf9a15dfb7084ce51"

# Run the Python script, passing the SLURM array task ID as the trial number
# srun python /cluster/home/bsay01/aphaproject/NLP-for-ORI/training_demo/bert_finetuning_script.py --trial ${SLURM_ARRAY_TASK_ID}
srun python /cluster/home/bsay01/aphaproject/NLP-for-ORI/training_demo/bert_finetuning_script.py --trial ${SLURM_ARRAY_TASK_ID}


conda deactivate

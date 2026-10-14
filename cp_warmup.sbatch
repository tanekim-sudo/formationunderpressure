#!/bin/bash
#SBATCH --job-name=cp_warm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=cp_warmup_%j.log
# Code version, step 2: teach only the tool format (the model's own well-formed attempts,
# right or wrong, in their natural proportions), shared by every arm. Saves
# results_code/warmup/<model name>/model, which cp_run.sbatch uses as MODEL.
#   sbatch --export=ALL,EXTRA="--families ... --ood ..." cp_warmup.sbatch
export PYTHONUNBUFFERED=1
source ~/venv/bin/activate
BASE=${BASE:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
python -u code_pressure.py --stage warmup --model $BASE ${EXTRA}

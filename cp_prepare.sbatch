#!/bin/bash
#SBATCH --job-name=cp_prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=cp_prepare_%j.log
# Code version, step 1: is the task set up right on this model? Screens every trap
# family (does the model write the obvious code, and fix it when shown a FAIL?).
#   sbatch cp_prepare.sbatch                                         (0.5B coder)
#   sbatch --export=ALL,BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct cp_prepare.sbatch
export PYTHONUNBUFFERED=1
source ~/venv/bin/activate
BASE=${BASE:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
python -u code_pressure.py --self_check || exit 1
python -u code_pressure.py --stage validate --model $BASE ${EXTRA}

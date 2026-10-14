#!/bin/bash
#SBATCH --job-name=cp_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cp_press_%a_%A.log
# Code version, step 3: the headline arms, formation then drift, one job per (arm, seed).
# Task index -> arm = index % 4, seed = index / 4.
#   sbatch --array=0-11  --export=ALL,MODEL=<warm model>,EXTRA="--families ... --ood ..." cp_run.sbatch   (seeds 0-2)
#   sbatch --array=12-19 --export=ALL,MODEL=<warm model>,EXTRA="--families ... --ood ..." cp_run.sbatch   (seeds 3-4)
export PYTHONUNBUFFERED=1
source ~/venv/bin/activate
ARMS=(exp spec_exp demo rule)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=$((SLURM_ARRAY_TASK_ID / 4))
MODEL=${MODEL:?set MODEL to the warmed-up code model}
python -u code_pressure.py --stage form --arm $ARM --seed $SEED --model $MODEL ${EXTRA} || exit 1
python -u code_pressure.py --stage press --arm $ARM --seed $SEED --pressure drift --model $MODEL ${EXTRA}

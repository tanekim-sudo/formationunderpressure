#!/bin/bash
#SBATCH --job-name=fp_ctrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=fp_press_ctrl_%a_%A.log
# Controls for the identity result: formation, then drift, in one job per (arm, seed).
# Task index -> arm = index % 3, seed = index / 3. Submit seeds 0-2, then 3-4:
#   sbatch --array=0-8  fp_controls.sbatch
#   sbatch --array=9-14 fp_controls.sbatch
# Same model, settings and drift as the existing five-seed runs (no EXTRA flags).
export PYTHONUNBUFFERED=1
source ~/venv/bin/activate
ARMS=(neutral_exp exp_spec relational_exp)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=$((SLURM_ARRAY_TASK_ID / 3))
MODEL=${MODEL:-results_fp/warmup/Qwen2.5-0.5B-Instruct/model}
python -u formation_pressure.py --stage form --arm $ARM --seed $SEED --model $MODEL ${EXTRA} || exit 1
python -u formation_pressure.py --stage press --arm $ARM --seed $SEED --pressure drift --model $MODEL ${EXTRA}

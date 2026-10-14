#!/bin/bash
#SBATCH --job-name=fp_press
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=fp_press_a%a_%A.log
#SBATCH --array=0-18
# Phases 2-3, one job per arm. PRESSURE, SEED and MODEL from the environment.
export PYTHONUNBUFFERED=1
source ~/venv/bin/activate
ARMS=(none harness rule paid demo spec distill exp exp_noreturn exp_reliable calib spec_exp distill_exp given invite full neutral_exp exp_spec relational_exp)
python -u formation_pressure.py --stage press --arm ${ARMS[$SLURM_ARRAY_TASK_ID]} \
  --pressure ${PRESSURE:-drift} --seed ${SEED:-0} --model ${MODEL:-results_fp/warmup/Qwen2.5-0.5B-Instruct/model} ${EXTRA}

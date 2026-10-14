# Where a Habit Is Anchored Decides How It Fails: Verification Under Reward Drift

Code for the paper of the same name ([paper](https://claude.ai/artifact/Atgz8bD6CQEaSdyd3gp3WW)).

We train Qwen2.5-0.5B-Instruct to call an always-correct checker before answering arithmetic problems, a quarter of which are traps where the obvious answer is wrong. The habit is installed in different ways (outcome RL, reward shaping, imitation, first-person character text before or after RL, and matched controls), brought to a matched starting strength, and then put under 200 steps of reward drift toward a proxy that pays for the wrong answer.

## Results

- **Installation method decides how the habit fails.** All 25 runs whose habit was built by RL alone stopped checking. Of 25 runs whose habit passed through a supervised stage, 13 were still checking (one-sided Fisher p ≈ 1.5 × 10⁻⁵), and 4 of those were hollow: the model called the checker, was told the right answer, and submitted the wrong one (p ≈ 0.055 against 0 of 25).
- **Self-description made the habit fragile.** First-person statements of the habit before RL made the RL-built habit 3.6× easier to break (4 of 4 seeds where it formed; paired t = 12.5, df = 3). Matched neutral text had no effect (0.92×), and statements about the people relying on the answer did not weaken it.
- **RL could not reach what the model never sampled.** Retained runs absorbed 112 to 352 learning-signal samples over drift, against 544 to 712 for outcome-RL runs. No eroded habit recovered within 30 steps of restored reward.

## Requirements

Python 3.10+, PyTorch, Hugging Face `transformers` and `peft`, one GPU. The paper's runs used Stanford FarmShare with slurm (the `.sbatch` files).

## Reproduce

```bash
# 1. Check the base model on the trap families
python -u formation_pressure.py --stage validate

# 2. Shared tool-format warm-up (saves the merged model used by every arm)
python -u formation_pressure.py --stage warmup
MODEL=results_fp/warmup/Qwen2.5-0.5B-Instruct/model

# 3. Form a habit, then apply drift and restoration (one arm, one seed)
python -u formation_pressure.py --stage form  --arm spec_exp --seed 0 --model $MODEL
python -u formation_pressure.py --stage press --arm spec_exp --seed 0 --pressure drift --model $MODEL

# On slurm: all arms for one seed, then the three control arms for seeds 0-4
SEED=0 sbatch fp_form.sbatch
SEED=0 PRESSURE=drift sbatch fp_press.sbatch
sbatch --array=0-14 fp_controls.sbatch

# 4. Analysis
python dose_fp.py drift        # dose to break per run, paired comparisons against exp
python endstates_fp.py         # end states (abandoned / intact / hollow) and Fisher tests
python analyze_fp.py           # retention, half-life, recovery
```

Arms used in the paper: `exp`, `rule`, `paid`, `calib`, `exp_noreturn`, `demo`, `spec_exp`, `exp_spec`, `neutral_exp`, `relational_exp`, `spec`, `harness`. Every arm and seed uses the same held-out evaluation items.

## Files

| File | Purpose |
|---|---|
| `formation_pressure.py` | Environment, all formation arms, drift and other pressures, restoration, evaluation |
| `dose_fp.py` | Dose to break: learning-signal samples absorbed before trap-checking halves |
| `endstates_fp.py` | End-state classification and Fisher tests |
| `analyze_fp.py` | Retention, half-life, recovery, paired tests |
| `code_pressure.py` | A code version of the environment (hidden tests, sandboxed); models at 0.5B and 1.5B could not repair code from a failing test, so it is released for larger models |
| `fp_*.sbatch`, `cp_*.sbatch` | Slurm jobs for the arithmetic and code versions |

## Citation

```
@misc{kim2026anchored,
  title  = {Where a Habit Is Anchored Decides How It Fails: Verification Under Reward Drift},
  author = {Kim, Tane},
  year   = {2026}
}
```

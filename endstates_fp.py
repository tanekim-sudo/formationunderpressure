"""
endstates_fp.py: how did each run end after drift?

Reads results_fp/press/drift/<arm>_s<seed>/curve.json and classifies the last
drift evaluation on held-out traps:
  abandoned  trap-checking <= 0.1
  intact     trap-checking >= 0.5 and trap accuracy >= 0.5
  hollow     trap-checking >= 1/3 and trap accuracy == 0 (checks, then ignores the result)
  partial    anything else
Then compares RL-only arms with arms that include a supervised stage
(one-sided Fisher exact test when one group has zero events).

Usage:  python endstates_fp.py [seeds, default 0-4]
"""

import json
import os
import sys
from math import comb

ROOT = "results_fp/press/drift"
RL_ONLY = ["exp", "rule", "paid", "calib", "exp_noreturn"]
SUPERVISED = ["demo", "spec_exp", "exp_spec", "relational_exp", "neutral_exp"]


def state(e):
    c, t = e["trap_check"], e["trap_true"]
    if c >= 1 / 3 and t == 0:
        return "hollow"
    if c >= 0.5 and t >= 0.5:
        return "intact"
    if c <= 0.1:
        return "abandoned"
    return "partial"


def end_state(arm, seed):
    try:
        c = json.load(open(os.path.join(ROOT, f"{arm}_s{seed}", "curve.json")))
    except Exception:
        return None
    p2 = c.get("phase2") or []
    if not p2 or p2[-1]["step"] < c["args"].get("p2_steps", 200):
        return None  # missing or still running
    return state(p2[-1])


def fisher_zero(n_zero, k, n_other):
    """One-sided p that the zero-event group has 0 of the k + 0 events by chance."""
    return comb(n_zero + n_other - k, n_zero) / comb(n_zero + n_other, n_zero)


def main():
    seeds = range(5)
    if len(sys.argv) > 1:
        a, b = sys.argv[1].split("-")
        seeds = range(int(a), int(b) + 1)
    groups = {}
    print(f"{'arm':<16}{'runs':>6}{'abandoned':>11}{'intact':>8}{'hollow':>8}{'partial':>9}")
    for arm in RL_ONLY + SUPERVISED:
        states = [s for s in (end_state(arm, sd) for sd in seeds) if s]
        counts = {k: states.count(k) for k in ("abandoned", "intact", "hollow", "partial")}
        g = "rl_only" if arm in RL_ONLY else "supervised"
        for k, v in counts.items():
            groups.setdefault(g, {}).setdefault(k, 0)
            groups[g][k] += v
        groups[g]["runs"] = groups[g].get("runs", 0) + len(states)
        print(f"{arm:<16}{len(states):>6}{counts['abandoned']:>11}{counts['intact']:>8}"
              f"{counts['hollow']:>8}{counts['partial']:>9}")
    rl, sv = groups.get("rl_only", {}), groups.get("supervised", {})
    if rl.get("runs") and sv.get("runs"):
        kept_rl = rl["intact"] + rl["hollow"] + rl["partial"]
        kept_sv = sv["intact"] + sv["hollow"] + sv["partial"]
        print(f"\nStill checking: RL-only {kept_rl}/{rl['runs']}, supervised stage {kept_sv}/{sv['runs']}")
        if kept_rl == 0:
            print(f"  one-sided Fisher p = {fisher_zero(rl['runs'], kept_sv, sv['runs']):.2e}")
        print(f"Hollow: RL-only {rl['hollow']}/{rl['runs']}, supervised stage {sv['hollow']}/{sv['runs']}")
        if rl["hollow"] == 0:
            print(f"  one-sided Fisher p = {fisher_zero(rl['runs'], sv['hollow'], sv['runs']):.3f}")


if __name__ == "__main__":
    main()

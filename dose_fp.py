"""
dose_fp.py: how much pressure did it take to break each arm?

Reads the Phase 2 logs (fp_press_*.log) and, per arm, reports:
  first signal   first step where any sampled rollout departed (a learning signal exists)
  signal->half   cumulative learning-signal samples until training check rate fell to
                 half its starting value (the "dose" needed to break the habit)
  break step     the step where that happened
  total signal   all learning-signal samples in Phase 2

Separates "resists pressure" (large dose to break) from "rarely explores leaving"
(late first signal, but breaks quickly once signal arrives).

Usage:  python dose_fp.py [pressure] [log pattern]
        defaults: drift, fp_press_*.log   (code version: python dose_fp.py drift "cp_press_*.log")
"""

import glob
import os
import re
import sys

HEAD = re.compile(r"stage=press arm=(\S+) pressure=(\S+) seed=(\d+)")
STEP = re.compile(r"\[p2 (\S+)\] step (\d+)\s+lam [\d.]+\s+reward [-\d.]+\s+check ([\d.]+)"
                  r".*learning-signal samples (\d+)")


def parse(path):
    arm = pressure = seed = None
    steps = []
    with open(path, errors="ignore") as f:
        for line in f:
            if arm is None:
                h = HEAD.search(line)
                if h:
                    arm, pressure, seed = h.group(1), h.group(2), int(h.group(3))
            m = STEP.search(line)
            if m:
                steps.append((int(m.group(2)), float(m.group(3)), int(m.group(4))))
    return arm, pressure, seed, steps


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "drift"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "fp_press_*.log"
    runs = {}
    for p in glob.glob(pattern):
        arm, pressure, seed, steps = parse(p)
        if arm is None or pressure != want or not steps:
            continue  # old-format logs (no learning-signal) are skipped
        key = (arm, seed)
        if key not in runs or os.path.getmtime(p) > runs[key][0]:
            runs[key] = (os.path.getmtime(p), steps)
    if not runs:
        print(f"No new-format Phase 2 logs for pressure '{want}' yet.")
        return
    print(f"\n=== Dose to break, pressure: {want} ===")
    print(f"{'arm':<16}{'seed':>5}{'start check':>13}{'first signal':>14}{'signal->half':>14}"
          f"{'break step':>12}{'total signal':>14}{'steps run':>11}")
    doses = {}
    for (arm, seed), (_, steps) in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        start = steps[0][1]
        first = next((s for s, _, k in steps if k > 0), None)
        cum, dose, brk = 0, None, None
        for s, c, k in steps:
            cum += k
            if start > 0.2 and c <= start / 2 and brk is None:
                dose, brk = cum, s
        total = sum(k for _, _, k in steps)
        if dose is not None:
            doses.setdefault(arm, {})[seed] = dose
        print(f"{arm:<16}{seed:>5}{start:>13.2f}{(first if first else 'never'):>14}"
              f"{(dose if dose is not None else 'never'):>14}{(brk if brk else 'never'):>12}"
              f"{total:>14}{steps[-1][0]:>11}")
    print("\nReading: 'first signal' late + small dose = it rarely explored leaving, but broke fast once it did.")
    print("Large dose = it kept choosing to stay even while being rewarded for leaving (real resistance).")
    print("'never' first signal = the pressure never reached it (no exploration), not robustness.")
    paired_summary(doses)


def paired_summary(doses, base="exp"):
    """Each arm against `base` on the seeds where both broke: how many seeds go each way,
    the mean ratio base/arm (above 1 = the arm is more fragile), and a paired t statistic."""
    if base not in doses:
        return
    print(f"\n=== Paired against {base} (seeds where both broke) ===")
    print(f"{'arm':<16}{'seeds':>6}{'arm < base':>12}{'mean dose':>11}{'base dose':>11}{'base/arm':>10}{'paired t':>10}")
    for arm in sorted(doses):
        if arm == base:
            continue
        shared = sorted(set(doses[arm]) & set(doses[base]))
        if not shared:
            continue
        a = [doses[arm][s] for s in shared]
        b = [doses[base][s] for s in shared]
        d = [y - x for x, y in zip(a, b)]
        n = len(d)
        mean_d = sum(d) / n
        t = "-"
        if n > 1:
            sd = (sum((x - mean_d) ** 2 for x in d) / (n - 1)) ** 0.5
            t = f"{mean_d / (sd / n ** 0.5):.2f}" if sd > 0 else "inf"
        ratio = (sum(b) / n) / (sum(a) / n) if sum(a) else float("inf")
        print(f"{arm:<16}{n:>6}{sum(x < y for x, y in zip(a, b)):>12}{sum(a) / n:>11.0f}"
              f"{sum(b) / n:>11.0f}{ratio:>10.2f}{t:>10}")
    print("Paired t has n-1 degrees of freedom; with 5 seeds, |t| > 2.78 is p < 0.05 two-sided.")


if __name__ == "__main__":
    main()

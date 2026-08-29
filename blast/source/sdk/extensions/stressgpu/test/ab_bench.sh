#!/usr/bin/env bash
# Paired, interleaved A/B for the GPU stress solve.
#
# The question this answers is "is arm B faster than arm A", reliably, on a box
# we share with other tenants. It is deliberately NOT the question "how fast is
# this in production" -- that needs the live server and is a separate, rarer
# check. Getting the ORDERING right every time is worth more than getting the
# magnitude right once.
#
# Two design choices carry the reliability:
#
#   1. INTERLEAVED, PAIRED. Arms run A,B,A,B,... and the statistic is the
#      per-pair difference. Running all of A then all of B is what makes a
#      co-tenant starting halfway through look like a regression; interleaving
#      makes drift hit both arms roughly equally and paired differencing
#      cancels what is left. This is why a paired design beats simply taking
#      more samples: more samples average the contamination IN.
#
#   2. WORK IS CHECKED BEFORE COST. The bench reports iteration and skip
#      counts. If the arms disagree on those they are not doing the same job,
#      and comparing their times is meaningless -- so that is reported as a
#      warning rather than silently folded into a speedup number.
#
# Calibrate before you trust it: `--a "" --b ""` runs A against itself, so the
# reported difference IS the noise floor. Do not believe an effect smaller than
# that number.
#
# Usage:
#   ab_bench.sh --reps 12 --a "BLAST_GPU_STABLE_GRAPH=1" --b "BLAST_GPU_STABLE_GRAPH=0"
#   ab_bench.sh --reps 12 --a "" --b ""        # null calibration
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

reps=10
arm_a=""
arm_b=""
metric="device_ms"
while [ $# -gt 0 ]; do
  case "$1" in
    --reps)   reps="$2"; shift 2 ;;
    --a)      arm_a="$2"; shift 2 ;;
    --b)      arm_b="$2"; shift 2 ;;
    --metric) metric="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

bash "$here/bench_and_profile.sh" --run-only >/dev/null 2>&1 || {
  echo "bench failed to build/run" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "interleaving $reps pairs; metric=$metric"
echo "  A: ${arm_a:-<baseline>}"
echo "  B: ${arm_b:-<baseline>}"
for i in $(seq 1 "$reps"); do
  for arm in a b; do
    envs="$arm_a"; [ "$arm" = b ] && envs="$arm_b"
    # shellcheck disable=SC2086
    line=$(env $envs bash "$here/bench_and_profile.sh" --run-only 2>/dev/null \
             | grep BENCHRESULT || true)
    [ -z "$line" ] && { echo "run failed (rep $i arm $arm)" >&2; exit 1; }
    echo "$line" >> "$tmp/$arm.txt"
  done
  printf '.'
done
echo

python3 - "$tmp" "$metric" <<'PY'
import sys, re, statistics
tmp, metric = sys.argv[1], sys.argv[2]

def load(path):
    rows = []
    for line in open(path):
        rows.append({k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)})
    return rows

a, b = load(f'{tmp}/a.txt'), load(f'{tmp}/b.txt')
n = min(len(a), len(b))

# Work first. Times are only comparable if the arms did the same job.
for key in ('iterations', 'skipped', 'bonds', 'nodes'):
    av = {r[key] for r in a[:n]}
    bv = {r[key] for r in b[:n]}
    if len(av | bv) > 1:
        spread_a = f"{min(av):.0f}-{max(av):.0f}" if len(av) > 1 else f"{min(av):.0f}"
        spread_b = f"{min(bv):.0f}-{max(bv):.0f}" if len(bv) > 1 else f"{min(bv):.0f}"
        flag = "WORK DIFFERS" if not (av & bv) else "work varies run-to-run"
        print(f"  ! {flag}: {key} A={spread_a} B={spread_b}")

av = [r[metric] for r in a[:n]]
bv = [r[metric] for r in b[:n]]
deltas = [y - x for x, y in zip(av, bv)]          # negative => B faster
pct = [100.0 * d / x for d, x in zip(deltas, av)]
wins = sum(1 for d in deltas if d < 0)

print(f"\n  A median {statistics.median(av):.4f} ms   "
      f"(min {min(av):.4f}, max {max(av):.4f}, spread "
      f"{100*(max(av)-min(av))/statistics.median(av):.1f}%)")
print(f"  B median {statistics.median(bv):.4f} ms   "
      f"(min {min(bv):.4f}, max {max(bv):.4f}, spread "
      f"{100*(max(bv)-min(bv))/statistics.median(bv):.1f}%)")
print(f"\n  paired delta: median {statistics.median(pct):+.2f}%  "
      f"(range {min(pct):+.2f}% .. {max(pct):+.2f}%)")
print(f"  B faster in {wins}/{n} pairs")

# Sign test: with no real effect each pair is a coin flip, so k wins out of n
# has probability C(n,k)/2^n. Reporting the two-sided p keeps "B won 7 of 10"
# from being read as a result when it is a coin.
from math import comb
k = max(wins, n - wins)
p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)
verdict = ("B faster" if wins > n - wins else "A faster") if p < 0.05 else "NO CALL"
print(f"  sign test p={p:.3f} -> {verdict}"
      + ("" if p < 0.05 else "  (consistent with noise)"))
PY

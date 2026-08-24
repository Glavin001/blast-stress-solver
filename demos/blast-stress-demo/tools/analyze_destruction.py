#!/usr/bin/env python3
"""Quantify whether a destruction run was progressive or explosive.

Reads the per-physics-step CSV written by `blast_stress_demo --frame-telemetry`
and answers the question a human answers by watching the video: "did the
structure come apart gradually as it was hit, or did nothing happen and then
the whole thing detonate at once?"

The judgement is deliberately mechanical so it can gate a run without anyone
watching it:

  deadImpactFraction  projectile impacts that landed BEFORE the first split,
                      over all impacts. High => "the balls bounce off and
                      nothing happens" — the visible dead period.
  burstFraction       largest share of all splits falling inside any single
                      0.5 s window. High => "and then it exploded".
  peakFrameShare      share of all splits in the single worst physics step.
                      A structure coming apart under load splits a few bonds
                      per step; one step taking out a large share of the run
                      is a cascade, not an impact.

A run is progressive when damage tracks the impacts that cause it. Both a
long dead period and a single dominant burst are failures, and they are
usually the same failure seen from two ends.
"""
import argparse
import csv
import sys

STEPS_PER_WINDOW = 30  # 0.5 s at the demo's fixed 60 Hz physics step


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(rows):
    splits = [int(r["splits_frame"]) for r in rows]
    impacts = [int(r["projectile_impacts_frame"]) for r in rows]
    seconds = [float(r["simulation_seconds"]) for r in rows]

    total_splits = sum(splits)
    total_impacts = sum(impacts)
    if total_splits == 0:
        return {"verdict": "INERT", "totalSplits": 0, "totalImpacts": total_impacts}

    first_split = next(i for i, s in enumerate(splits) if s > 0)
    dead_impacts = sum(impacts[:first_split])

    # Largest split count inside any contiguous 0.5 s window.
    window = sum(splits[:STEPS_PER_WINDOW])
    burst, burst_at = window, 0
    for i in range(STEPS_PER_WINDOW, len(splits)):
        window += splits[i] - splits[i - STEPS_PER_WINDOW]
        if window > burst:
            burst, burst_at = window, i - STEPS_PER_WINDOW + 1

    peak_frame = max(splits)
    active = [i for i, s in enumerate(splits) if s > 0]
    span = seconds[active[-1]] - seconds[active[0]]

    return {
        "totalSplits": total_splits,
        "totalImpacts": total_impacts,
        "firstSplitSecond": seconds[first_split],
        "deadImpacts": dead_impacts,
        "deadImpactFraction": dead_impacts / total_impacts if total_impacts else 0.0,
        "burstSplits": burst,
        "burstFraction": burst / total_splits,
        "burstAtSecond": seconds[burst_at],
        "peakFrameSplits": peak_frame,
        "peakFrameShare": peak_frame / total_splits,
        "activeSteps": len(active),
        "damageSpanSeconds": span,
    }


# Below this many splits the burst shares are small-sample noise: 7 of 11
# splits landing in one window says nothing about cascading, it just says the
# run barely broke anything. Only the dead-period test is meaningful there.
MIN_SPLITS_FOR_BURST = 30


def verdict(m):
    if m.get("verdict") == "INERT":
        return "INERT", ["nothing broke at all"]
    reasons = []
    if m["deadImpactFraction"] > 0.5:
        reasons.append(
            f"{m['deadImpactFraction']:.0%} of impacts landed before anything broke")
    if m["totalSplits"] >= MIN_SPLITS_FOR_BURST:
        if m["burstFraction"] > 0.5:
            reasons.append(
                f"{m['burstFraction']:.0%} of all damage happened in one 0.5 s window")
        if m["peakFrameShare"] > 0.15:
            reasons.append(
                f"one physics step produced {m['peakFrameShare']:.0%} of all damage")
    if reasons:
        return "EXPLOSIVE", reasons
    if m["totalSplits"] < MIN_SPLITS_FOR_BURST:
        return "LIGHT", [f"only {m['totalSplits']} splits — too little damage to judge cascading"]
    return "PROGRESSIVE", []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="path from --frame-telemetry")
    parser.add_argument("--require-progressive", action="store_true",
                        help="exit non-zero unless the run is progressive")
    args = parser.parse_args()

    metrics = analyze(load(args.csv))
    tag, reasons = verdict(metrics)

    print(f"verdict: {tag}")
    for key, value in metrics.items():
        if key == "verdict":
            continue
        print(f"  {key:22s} {value:.4f}" if isinstance(value, float)
              else f"  {key:22s} {value}")
    for reason in reasons:
        print(f"  ! {reason}")

    if args.require_progressive and tag != "PROGRESSIVE":
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Reading through `head` closes the pipe early; that is not an error.
        sys.exit(0)

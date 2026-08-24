#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${1:-/root/recordings/blast-gpu-varied-12x12-20k-optimized-benchmark.mp4}"
stem="${output%.mp4}"
cargo_bin="${CARGO:-/root/.cargo/bin/cargo}"

cmake --build "${demo_dir}/build" --target blast_stress_demo -j2

cd "${demo_dir}/recorder"
# --resim-passes 0: a resim frame re-runs simulate(), and the 12x12 realtime
# gate has <1 ms of worst-frame headroom, so the recorded benchmark stays on
# the excess-force path. Drop that pin to record the resimulation quality path.
"${cargo_bin}" run --release --locked -- record \
  --sim-bin ../build/blast_stress_demo \
  --grid 12 --duration 30 --settle 1.5 \
  --snapshot-fps 30 --require-gpu --gpu-stress \
  --require-partial-destruction --chase-projectile \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=540 \
  --sim-arg=--projectile-mass-scale --sim-arg=1.8 \
  --sim-arg=--contact-force-scale --sim-arg=3.1 \
  --sim-arg=--excess-force-scale --sim-arg=0.012 \
  --sim-arg=--resim-passes --sim-arg=0 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=20000 \
    --sim-arg=--require-varied-building-heights \
  --metadata "${stem}.metadata.json" \
  --frame-telemetry "${stem}.simulation.frames.csv" \
  --output "${output}"

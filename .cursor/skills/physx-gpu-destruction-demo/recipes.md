# PhysX GPU destruction — recipes

Companion to [SKILL.md](SKILL.md). Copy-paste from repo root unless noted.

## Env

```bash
export DEMO=/root/workspace/blast-stress-solver/demos/blast-stress-demo/build/blast_stress_demo
export REC=/root/workspace/blast-stress-solver/demos/blast-stress-demo/recorder
export SCENE_HR=/root/workspace/blast-stress-solver/blast/blast-stress-demo-rs/assets/scenes/high-rise-10f.json
export SCENE_LOCAL=/root/workspace/blast-stress-solver/blast/blast-stress-demo-rs/assets/scenes/high-rise-10f-local.json
export PATH="/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO=/root/.cargo/bin/cargo
```

Use absolute `$DEMO` paths — shell cwd often resets to `/root`.

## Regenerate high-rise packs

```bash
cd /root/workspace/blast-stress-solver/blast/blast-stress-solver
# optional if dist/scenarios.js missing:
# npm run build:ts
HIGH_RISE_FLOORS=10 node scripts/export-high-rise.mjs
HIGH_RISE_FLOORS=10 HIGH_RISE_LOCAL_DAMAGE=1 node scripts/export-high-rise.mjs
```

## Sim-only tune loop (no MP4)

```bash
mkdir -p /tmp/highrise-tune
name=candidate-a
"$DEMO" --physics gpu --require-gpu --grid 5 --duration 30 --settle 1.5 \
  --gpu-stress --gpu-stress-min-bonds 3000 \
  --min-stress-contact-impulse 15 --stress-workers 20 \
  --tall-building-stride 12 \
  --require-min-authored-chunks 14000 \
  --require-varied-building-heights \
  --require-partial-destruction \
  --scene "$SCENE_LOCAL" \
  --projectile-mass-scale 2.6 \
  --contact-force-scale 1.15 \
  --stress-limit-scale 1.0 \
  --excess-force-scale 0.018 \
  --resim-passes 1 \
  --scoped-resim \
  --quiet-capture-skip \
  --metadata /tmp/highrise-tune/$name.metadata.json \
  --frame-telemetry /tmp/highrise-tune/$name.frames.csv \
  --output-state ''
```

Omit `--max-bodies-per-structure` (default unlimited). Only pass it when someone explicitly asks for a hard body stop.

Omit `--require-realtime` while searching; add it when locking a realtime recipe.

Inspect:

```bash
python3 - <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
t=m['frameTelemetry']; mo=m['destructionMotion']; d=m['destructionDistribution']
print('misses', t['budgetMissFrames'], 'maxMs', round(t['maxFrameHostMilliseconds'],2))
print('mass', m['tuning']['projectileMass'], 'bodies', m['peakBodyCount'], 'splits', m['splits'])
print('moved/fall', mo['structuresWithMovedChunks'], mo['movedChunks'], mo['fallenChunks'])
print('dyn/sup', mo['dynamicChunks'], mo['supportedRemainderChunks'])
print('damage', d['partiallyFracturedStructures'], d['heavilyFracturedStructures'], d['shatteredStructures'])
PY
/tmp/highrise-tune/candidate-a.metadata.json
```

## Record: 25-building local-break (planted walls)

```bash
cd "$REC"
$CARGO run --release --locked -- record \
  --sim-bin ../build/blast_stress_demo \
  --grid 5 --duration 30 --settle 1.5 --snapshot-fps 30 \
  --require-gpu --gpu-stress --require-partial-destruction --chase-projectile \
  --sim-arg=--scene --sim-arg=$SCENE_LOCAL \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=3000 \
  --sim-arg=--projectile-mass-scale --sim-arg=2.6 \
  --sim-arg=--contact-force-scale --sim-arg=3.30 \
  --sim-arg=--min-stress-contact-impulse --sim-arg=5 \
  --sim-arg=--stress-limit-scale --sim-arg=0.50 \
  --sim-arg=--excess-force-scale --sim-arg=0.042 \
  --sim-arg=--stress-workers --sim-arg=20 \
  --sim-arg=--tall-building-stride --sim-arg=12 \
  --sim-arg=--resim-passes --sim-arg=1 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=14000 \
  --sim-arg=--require-varied-building-heights \
  --metadata /root/recordings/blast-gpu-highrise-5x5-14k-10floor-local-break.metadata.json \
  --frame-telemetry /root/recordings/blast-gpu-highrise-5x5-14k-10floor-local-break.frames.csv \
  --output /root/recordings/blast-gpu-highrise-5x5-14k-10floor-local-break.mp4
```

(Adjust scales if gates fail; prefer sim-only first. Do **not** pin `--max-bodies-per-structure` unless explicitly requested — body/bond budgets falsify punch-through.)

## Record: heavier high-rise (non-local pack)

```bash
cd "$REC"
$CARGO run --release --locked -- record \
  --sim-bin ../build/blast_stress_demo \
  --grid 5 --duration 30 --settle 1.5 --snapshot-fps 30 \
  --require-gpu --gpu-stress --require-partial-destruction --chase-projectile \
  --sim-arg=--scene --sim-arg=$SCENE_HR \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=3000 \
  --sim-arg=--projectile-mass-scale --sim-arg=2.6 \
  --sim-arg=--contact-force-scale --sim-arg=1.05 \
  --sim-arg=--min-stress-contact-impulse --sim-arg=15 \
  --sim-arg=--stress-limit-scale --sim-arg=1.17 \
  --sim-arg=--excess-force-scale --sim-arg=0.015 \
  --sim-arg=--stress-workers --sim-arg=20 \
  --sim-arg=--tall-building-stride --sim-arg=12 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=14000 \
  --sim-arg=--require-varied-building-heights \
  --metadata /root/recordings/blast-gpu-highrise-5x5-14k-10floor-heavier.metadata.json \
  --frame-telemetry /root/recordings/blast-gpu-highrise-5x5-14k-10floor-heavier.frames.csv \
  --output /root/recordings/blast-gpu-highrise-5x5-14k-10floor-heavier.mp4
```

## Record: canonical 12×12 realtime (script)

```bash
bash demos/blast-stress-demo/record_stress_benchmark.sh \
  /root/recordings/blast-gpu-varied-12x12-20k-optimized-benchmark.mp4
```

Script pins `--resim-passes 0` for worst-frame headroom. For punch-through quality, re-record with `--sim-arg=--resim-passes --sim-arg=1` and expect more frame time.

## Aggressive break (realtime optional)

When the user relaxes 60 Hz: omit `--require-realtime` / `--sim-arg=--require-realtime`, raise mass/contact/body caps, lower `--stress-limit-scale`, then **report** `frameTelemetry.budgetMissFrames` and max ms. Example output stem: `…-heavy-blast.mp4`.

## Proof frames

```bash
ffmpeg -y -v error -ss 8  -i /root/recordings/OUT.mp4 -frames:v 1 /tmp/proof-8s.png
ffmpeg -y -v error -ss 16 -i /root/recordings/OUT.mp4 -frames:v 1 /tmp/proof-16s.png
```

Read the PNGs in-agent to verify planted vs tipping, wall holes, and whether balls still bounce on fracture frames (`--resim-passes`).

## Impact timeline from frames.csv

```bash
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('/root/recordings/OUT.frames.csv')))
imp=[(float(r['simulation_seconds']), float(r['projectile_impacts_total'])) for r in rows]
for want in [2,5,10,15,20,25,30]:
  t,v=min(imp, key=lambda x: abs(x[0]-want))
  print(f't~{want}s impacts_total={v:.0f}')
PY
```

## Tests worth knowing

```bash
ctest --test-dir demos/blast-stress-demo/build --output-on-failure
```

Includes ScenePack load, GPU convex reduction, stress GPU/CPU checks, and resim snapshot / behavioral penetrate-vs-deflect (`--resim-passes 1` vs `0`). See demo README.

---
name: physx-gpu-destruction-demo
description: >-
  Build, tune, record, and compare PhysX GPU + Blast mini-city destruction
  demos (high-rise skyline, telemetry, MP4 under /root/recordings). Use when
  recording blast-gpu videos, tuning projectile/stress knobs, regenerating
  high-rise ScenePacks, reading *.metadata.json / frames.csv, or reproducing
  prior realtime / local-break / heavy-blast benchmarks.
---

# PhysX GPU destruction demo workflow

Repo root: `/root/workspace/blast-stress-solver`  
Binary: `demos/blast-stress-demo/build/blast_stress_demo`  
Recorder: `demos/blast-stress-demo/recorder`  
Outputs: always under `/root/recordings/`  
Do **not** commit `*.towerstate`.

## Build

```bash
cd /root/workspace/blast-stress-solver
cmake -S demos/blast-stress-demo -B demos/blast-stress-demo/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPHYSX_ROOT=/root/PhysX/physx \
  -DPHYSX_LIB_DIR=/root/PhysX/physx/bin/linux.x86_64/release
cmake --build demos/blast-stress-demo/build --target blast_stress_demo --parallel 16
```

Recorder (once / after Rust changes):

```bash
cd demos/blast-stress-demo/recorder
PATH="/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo build --release --locked
```

## Workflow (tune → gate → record → proof)

1. **Sim-only tune** with absolute binary path (cwd may be `/root`). Write metadata/CSV under `/tmp/...`. Iterate knobs until gates and visuals match the ask.
2. **Record MP4** via recorder `cargo run --release --locked -- record ...` with the same `--sim-arg=` knobs.
3. **Compare** prior vs new `*.metadata.json` (destruction + frameTelemetry).
4. **Proof frames**: `ffmpeg -ss N -i out.mp4 -frames:v 1 /tmp/proof.png` then Read the PNG.

Canonical 12×12 script: `bash demos/blast-stress-demo/record_stress_benchmark.sh`.  
Detailed recipes: [recipes.md](recipes.md).

## High-rise ScenePack

Generator (after `npm run build:ts` in `blast/blast-stress-solver` if needed):

```bash
cd blast/blast-stress-solver
HIGH_RISE_FLOORS=10 node scripts/export-high-rise.mjs
HIGH_RISE_FLOORS=10 HIGH_RISE_LOCAL_DAMAGE=1 node scripts/export-high-rise.mjs
```

Outputs (gitignored):  
`blast/blast-stress-demo-rs/assets/scenes/high-rise-10f.json`  
`blast/blast-stress-demo-rs/assets/scenes/high-rise-10f-local.json`  

Local pack: weak infill bond areas, strong footings, `mass=0` on foundation+columns (kinematic frame). Pass via:

`--sim-arg=--scene --sim-arg=/root/workspace/blast-stress-solver/blast/blast-stress-demo-rs/assets/scenes/high-rise-10f-local.json`

## Important knobs (`--sim-arg`)

| Knob | Role |
|------|------|
| `--projectile-mass-scale` | Ball mass (pack default 2500 kg × scale) |
| `--projectile-speed-scale` | Impact speed |
| `--contact-force-scale` | Contact → stress injection |
| `--stress-limit-scale` | Material strength (<1 breaks easier) |
| `--excess-force-scale` | Post-split kick (used when resim off) |
| `--resim-passes` | `1` = fracture-frame rollback+re-step (default); `0` = excess-force path / realtime pin |
| `--max-bodies-per-structure` | Fracture budget per building |
| `--tall-building-stride` | Full-height every N buildings (skyline) |
| `--gpu-stress-min-bonds` | CUDA stress crossover (3000 for high-rise graphs) |
| `--min-stress-contact-impulse` | Ignore weak non-projectile contacts |
| `--require-realtime` | Fail any >16.67 ms post-warm-up frame |
| `--require-partial-destruction` | Damage quality gate |
| `--require-min-authored-chunks` | Scale gate (e.g. 14000 / 20000) |
| `--require-varied-building-heights` | 1/2/3-floor cohorts present |

Pass demo flags through the recorder as repeated `--sim-arg=FLAG --sim-arg=VALUE`.

## Telemetry to read

**`*.metadata.json`**
- `tuning.*` — effective mass/speed/contact/limits/caps
- `destructionDistribution` — intact/partial/heavy/shattered, bodies/structure
- `destructionMotion` — moved/fallen/far chunks, `supportedRemainderChunks`, max displacement
- `frameTelemetry` — `budgetMissFrames`, mean/p95/max host ms, destruction misses
- `projectileImpactContacts` / `projectileImpactImpulse`
- `resimulation` (if present) — passes totals

**`*.frames.csv`** (60 Hz) — `frame_host_ms`, `physics_step_ms`, `adapter_tick_ms`, `bodies`, `splits`, `projectile_impacts_*`, stress/gpu columns.

**`*.render.frames.csv`** — `spheres` = projectile actor count (pre-spawned; not launch count over time).

Quick compare:

```bash
python3 - <<'PY'
import json
old=json.load(open('/root/recordings/PRIOR.metadata.json'))
new=json.load(open('/root/recordings/NEW.metadata.json'))
# print tuning, frameTelemetry misses/max, destructionMotion, peakBodyCount, splits
PY
```

## Projectiles (behavior)

- Launches are **multi-wave** (4 × building count), spaced across the destruction window (not a 0.15 s burst).
- Lifetime ≈ pack `ttlMs` × `--projectile-ttl-scale` (default scale 0.4 → ~3.2 s).
- Impact *contacts* ≫ ball count; use launch schedule + `spheres` for how many balls exist.

## Physics correctness (do / don’t)

- **Do** use real solved contacts + stress fracture. For punch-through on the fracture frame, use **`--resim-passes ≥ 1`** (rollback + re-step). See `PHYSICS_ENGINE_CONTRACT.md` §2.8 / §4 and demo README “Known limitations”.
- **Don’t** add synthetic momentum, ghost/no-solve projectile contacts, or other fake penetration hacks — rejected as non-physical.
- `--resim-passes 0` keeps the older excess-force continuity path and is what the hard realtime 12×12 benchmark pins.

## Acceptance videos (reference)

| File | Intent |
|------|--------|
| `blast-gpu-varied-12x12-20k-optimized-benchmark.mp4` | Canonical realtime 12×12 (~20k chunks), often `--resim-passes 0` |
| `blast-gpu-highrise-5x5-14k-10floor-realtime.mp4` | 25-building high-rise, realtime |
| `blast-gpu-highrise-5x5-14k-10floor-heavier.mp4` | Heavier ball (~6500 kg), more debris |
| `blast-gpu-highrise-5x5-14k-10floor-local-break.mp4` | Local-damage pack, planted frame, wall peel |
| `blast-gpu-highrise-5x5-14k-10floor-heavy-blast.mp4` | Aggressive destruction; may report 60 Hz misses if gate relaxed |

Sidecars share the stem: `.metadata.json`, `.frames.csv`, `.render.*`, `.simulation.*`.

## Hard realtime vs quality

- User asks for **hard 60 Hz**: keep `--require-realtime`, pin body caps / resim / stress so `budgetMissFrames==0`.
- User asks to **relax realtime** / maximize break: omit `--require-realtime`, report `budgetMissFrames` and max ms from metadata.

## More detail

- [recipes.md](recipes.md) — copy-paste high-rise record commands and tune loops  
- Demo docs: `demos/blast-stress-demo/README.md`, `recorder/README.md`  
- Contract: `blast/blast-stress-solver-rs/PHYSICS_ENGINE_CONTRACT.md`

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

Local pack: `mass=0` on **foundations only** — columns and slabs keep real mass so
gravity actually flows through the stress graph. Joint capacity comes from bond-area
multipliers that stay within an order of magnitude of geometric truth (footing 2.0,
frame 1.5–2.0, facade 0.09–1.5), which yields safety factors of ~2.7 (facade) to ~39
(base anchor) under self-weight. Pass via:

`--sim-arg=--scene --sim-arg=/root/workspace/blast-stress-solver/blast/blast-stress-demo-rs/assets/scenes/high-rise-10f-local.json`

## Important knobs (`--sim-arg`)

| Knob | Role |
|------|------|
| `--projectile-radius-scale` | **Preferred energy dial.** Scale radius and mass together (mass ∝ r³) to keep the ball at a real density |
| `--projectile-mass-scale` | Ball mass (pack default 2500 kg × scale). Raising this alone makes the ball denser — check the density you end up with |
| `--projectile-speed-scale` | Impact speed |
| `--contact-force-scale` | Contact → stress injection. **1.0 is physically correct** (the adapter already divides impulse by dt); treat ≠1 as a sensitivity experiment, not tuning |
| `--stress-limit-scale` | Material strength (<1 breaks easier). **1.0 = the concrete the pack claims to be made of** |
| `--excess-force-scale` | Post-split kick (used when resim off) |
| `--require-min-safety-factor` | Fail if any joint class can't carry self-weight with this margin |
| `--require-max-safety-factor` | Fail if any joint class is so over-authored it carries no load and cannot break |
| `--resim-passes` | `1` = fracture-frame rollback+re-step (default); `0` = excess-force path / realtime pin |
| `--max-bodies-per-structure` | **Default 0 = unlimited.** Opt-in hard stop only; do not use as a perf budget (falsifies fracture/resim) |
| `--max-fractures-per-actor-per-tick` | **Default 0 = unlimited.** Opt-in; artificial caps degrade quality |
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
- `gravityLoadPath` — per joint class: bonds, peak/mean utilisation, safety factor, peak compression/tension/shear (Pa) under self-weight
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

### Get more destruction by adding energy, not gain

`--contact-force-scale` and `--stress-limit-scale` should both stay at **1.0**. They are the two ends of the same equation; moving either one breaks the correspondence between the simulation and the material the pack says it is made of, and the effects cancel so the run still "looks fine".

If a run is not destructive enough, raise the **projectile's physical energy** — bigger radius (with mass ∝ r³ so density stays real), higher speed, or more waves. A 20 t / r = 1.2 m ball at 18 m/s is a 2763 kg/m³ sphere: heavy, and buildable. A 650 t / r = 0.6 m ball is 718,000 kg/m³ — 32× osmium — and only "works" because it is cancelling an authoring error elsewhere.

### Read the gravity load path before tuning anything

Every run prints, before the self-weight gate:

```
gravity load path (utilisation = peak stress / elastic limit):
  infill~slab       bonds=836   peakUtil=0.371    safetyFactor=2.696  ...
  column~slab       bonds=342   peakUtil=0.0418   safetyFactor=23.92  ...
```

and writes the same table to `metadata.gravityLoadPath`. Utilisation is how much of a joint class's capacity the structure's own weight already consumes; safety factor is its reciprocal.

- Safety factor **< 1** → that class cannot hold the building up; it will fail during warmup.
- Safety factor **in the 2–40 band** → a real structure. Facade should be the weakest class, the frame stronger, the base anchor strongest.
- Safety factor **in the thousands or more** → the joint carries no load and **cannot be broken by any impulse the sim can produce**. Destruction results are then vacuous: damage gates pass because the structure is indestructible, not because the load path works.

Pin both ends on any run you trust: `--require-min-safety-factor 2 --require-max-safety-factor 200`.

## Acceptance videos (reference)

| File | Intent |
|------|--------|
| `blast-gpu-varied-12x12-20k-optimized-benchmark.mp4` | Canonical realtime 12×12 (~20k chunks), often `--resim-passes 0` |
| `blast-gpu-highrise-5x5-14k-10floor-realtime.mp4` | 25-building high-rise, realtime |
| `blast-gpu-highrise-5x5-14k-10floor-heavier.mp4` | Heavier ball (~6500 kg), more debris |
| `blast-gpu-highrise-5x5-14k-10floor-local-break.mp4` | Local-damage pack, planted frame, wall peel |
| `blast-gpu-highrise-5x5-14k-10floor-heavy-blast.mp4` | Aggressive destruction; may report 60 Hz misses if gate relaxed |

Sidecars share the stem: `.metadata.json`, `.frames.csv`, `.render.*`, `.simulation.*`.

**These were all recorded before the authoring fix** (bond areas up to 1.5e6 m², contact-force gain 30–99×, projectiles up to 650 t / 718,000 kg m⁻³). Their destruction numbers are not comparable to runs made with unity gain and the safety-factor gates on, and they should be re-recorded rather than used as a target. As of the fix, `record_stress_benchmark.sh` (12×12 on `fractured-tower.json`) fails its own `--require-partial-destruction` and `--require-realtime` gates — this predates the fix and is tracked separately; that pack has no `nodeTypes` and sits at safety factor ≈1.1 under self-weight, i.e. it is marginal on its own.

## Hard realtime vs quality

- User asks for **hard 60 Hz**: keep `--require-realtime`, tune stress/workers/scene size so `budgetMissFrames==0`. Do **not** invent body/bond fracture caps to buy headroom — those falsify destruction/resim.
- User asks to **relax realtime** / maximize break: omit `--require-realtime`, report `budgetMissFrames` and max ms from metadata.

## More detail

- [recipes.md](recipes.md) — copy-paste high-rise record commands and tune loops  
- Demo docs: `demos/blast-stress-demo/README.md`, `recorder/README.md`  
- Contract: `blast/blast-stress-solver-rs/PHYSICS_ENGINE_CONTRACT.md`

# Blast PhysX GPU Mini-City

This demo is an end-to-end C++ integration between the Blast stress solver and
PhysX 5 rigid bodies. It loads the committed fractured-tower ScenePack, clones
it into a deterministic mini-city, injects PhysX contact impulses into the
stress graph, migrates existing shapes when actors split, and can run either the
CPU or CUDA rigid-body pipeline. Large stress graphs can additionally use the
resident CUDA CGNR backend while small graphs stay on the faster CPU path.

The renderer is deliberately separate from simulation. The C++ executable
writes `TWSTATE1` transforms and JSON telemetry; the Rust recorder renders those
transforms headlessly with Vulkan/wgpu and encodes MP4 with NVENC.

## Prerequisites

- Linux x86-64, CMake 3.20+, and a C++17 compiler.
- A release build of PhysX 5 with GPU support. `PHYSX_ROOT` must contain
  `include/PxPhysicsAPI.h`; `PHYSX_LIB_DIR` must contain the static PhysX,
  cooking, CUDA context manager, and foundation libraries used by
  `CMakeLists.txt`.
- CUDA driver/runtime and an NVIDIA GPU for `--physics gpu`.
- For recording: a current Rust toolchain, Vulkan loader/driver, `ffmpeg` with
  `h264_nvenc`, and `ffprobe`. `nvidia-smi` is optional.

The defaults match this workspace:

```sh
PHYSX_ROOT=/root/PhysX/physx
PHYSX_LIB_DIR="$PHYSX_ROOT/bin/linux.x86_64/release"
```

## Configure, build, and test

From the repository root:

```sh
cmake -S demos/blast-stress-demo \
  -B demos/blast-stress-demo/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPHYSX_ROOT="$PHYSX_ROOT" \
  -DPHYSX_LIB_DIR="$PHYSX_LIB_DIR"
cmake --build demos/blast-stress-demo/build --parallel
ctest --test-dir demos/blast-stress-demo/build --output-on-failure
```

The tests cover the CPU split/migration contract, ScenePack loading,
GPU-compatible convex-hull reduction, CUDA/CPU stress equivalence, bond-health
invariants, and compact GPU break-event generation.

## Run the simulation

Fail if CUDA or the GPU rigid-body pipeline is unavailable:

```sh
demos/blast-stress-demo/build/blast_stress_demo \
  --physics gpu --require-gpu --gpu-stress \
  --grid 3 --duration 12 --settle 1.5 \
  --snapshot-fps 30 \
  --state /root/recordings/blast-mini-city.twstate \
  --metadata /root/recordings/blast-mini-city.metadata.json \
  --frame-telemetry /root/recordings/blast-mini-city.frames.csv
```

Use `--physics cpu` for the CPU reference path. Without `--require-gpu`, a GPU
request may fall back to CPU when CUDA context creation fails. Metadata records
both the requested mode and `gpuActive`, so automation should check
`gpuActive: true` rather than infer acceleration from a successful exit.

`--gpu-stress` enables the hybrid CUDA stress backend. Graphs at or above the
measured 4,096-bond crossover use CUDA; smaller graphs stay on CPU. Each
204-chunk tower has only 546 bonds, so the 8×8 acceptance run deliberately
keeps those inexpensive graph solves on CPU while PhysX GPU handles the growing
rigid-body, broadphase, and contact workload. Use
`--gpu-stress-min-bonds 0` only to validate the CUDA stress path on tiny graphs;
it is slower at city scale because it launches one small CUDA solve per tower.

The simulation is fixed at 60 Hz. `--snapshot-fps` must be a divisor of 60 and
only controls exported render frames. Grid size is limited to 1–12. Run
`blast_stress_demo --help` for all paths and camera-size options.

## Record headlessly

Build the recorder:

```sh
cd demos/blast-stress-demo/recorder
PATH="/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo build --release --locked
```

Simulate, render, and encode without X11:

```sh
cargo run --release --locked -- record \
  --sim-bin ../build/blast_stress_demo \
  --grid 8 --duration 18 --settle 1.5 \
  --snapshot-fps 30 --require-gpu --gpu-stress \
  --require-partial-destruction --chase-projectile \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=540 \
  --sim-arg=--contact-force-scale --sim-arg=3.1 \
  --sim-arg=--excess-force-scale --sim-arg=0.012 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=10000 \
  --output /root/recordings/blast-gpu-8x8-13k-gpu-stress-acceptance.mp4
```

All project-generated videos should be written under `/root/recordings/`.
See [`recorder/README.md`](recorder/README.md) for rendering existing state
files, telemetry sidecars, and the `TWSTATE1` layout.

## PhysX GPU configuration and safeguards

GPU mode creates a validated `PxCudaContextManager` and configures:

- `PxSceneFlag::eENABLE_GPU_DYNAMICS`;
- `PxBroadPhaseType::eGPU`;
- TGS, PCM, stabilization, and eight GPU partitions;
- scene-size-derived GPU contact, patch, pair, heap, temporary, and collision
  stack capacities;
- convex cooking with `buildGPUData = true` and at most 64 input points.

`--require-gpu` turns unavailable CUDA into a hard failure. PhysX errors,
capacity/overflow warnings, CUDA abort mode, contact drops, GPU high-water
requirements, body/awake counts, split counts, shape migrations, and split
position/point-velocity drift are exposed in telemetry. The executable exits
nonzero if required GPU execution or continuity/scene health checks fail.

`--require-partial-destruction` (alias `--require-dynamic-destruction`) now
checks visible physical damage rather than graph splits alone. It requires:

- no pre-impact fracture, valid mappings, and split continuity below `1e-3`;
- projectile contacts and splits in at least 60% of structures;
- displaced chunks in at least 60% and falling chunks in at least one-sixth of
  structures, with pieces traveling at least 2 m and falling at least 0.5 m;
- at least half of all authored chunks remaining in supported components;
- zero shattered structures and a duration-scaled upper displacement bound
  that rejects explosive/numerically unstable launches.

The adapter applies Blast `getExcessForces()` as a one-shot `force × dt`
impulse to newly dynamic children. Without this, bonds broke correctly but
fragments inherited the kinematic parent's zero velocity and remained lodged
in the wall. Each tower is capped at 48 bodies so damage remains gradual and
the latent-chunk hierarchy cannot degenerate into a 204-body shatter.

The GPU-stress acceptance tuning is: 1,500 kg, 0.5 m projectiles at 25 m/s;
contact-force scale 93; stress-limit scale 0.8; released-load scale 0.012; a
4 m/s bounded separation impulse; 6 m/s linear and 8 rad/s angular fragment
caps; and a 3.2 s projectile lifetime. Two opposing waves remain inside the
recording window so the city shows citywide tear-through rather than one
sparse pass.

`--require-min-authored-chunks N` makes latent scene scale an executable
contract. `--require-realtime` performs eight unmeasured warm-up steps to
allocate lazy PhysX/solver resources, then fails if any recorded 60 Hz step
exceeds 16.67 ms. Both the all-frame and destruction-only miss counts and the
maximum destruction-frame time are written to metadata.

## Time-series diagnostics

`--frame-telemetry` writes one CSV row per 60 Hz physics step. It separates:

- blocking `simulate()` + `fetchResults()` host time (`physics_step_ms`);
- PhysX contact callback and stress-force injection time;
- gravity loading, stress solve, and fracture/topology mutation time;
- total adapter tick, periodic mapping validation, state export, and complete
  host-frame time;
- current/awake bodies, solver islands/skips, overstressed bonds, contacts,
  projectile impacts/impulse, splits, migrated shapes, sleeping-actor skips,
  contact drops, and continuity drift.
- CUDA stress kernel milliseconds and per-frame H2D/D2H bytes.

`physics_step_ms` is the host-observed completion time for GPU simulation,
including synchronization and callbacks; it is not claimed to be a CUDA
kernel-only timestamp. Recorder runs also sample utilization, memory, power,
temperature, and SM clock with `nvidia-smi`.

The recorder synchronizes the CSV to exported frames and draws those metrics
into the MP4. It additionally writes `*.render.frames.csv` with CPU command
submission, readback, overlay, and NVENC pipe timings plus
`*.render.summary.json` percentiles. The first overview camera orbits slowly,
while the other three remain stable references for spotting discontinuities or
jitter.

`TWSTATE1` frames are delta encoded: unchanged or sleeping visuals are omitted
and retain their previous pose. On the 12×12 capture this reduced a seven-second
state stream from 199 MB to 3.4 MB and cut mean export work from 2.92 ms to
0.68 ms.

## Current benchmark

The current GPU-stress visual acceptance run is 8×8: 64 buildings, 13,056
authored chunks, and 19.5 seconds of recorded time at a fixed 60 Hz:

- simulation: 7.67 s wall time (2.54× real time), 6.55 ms mean, 10.94 ms p95,
  and 16.07 ms maximum host frame;
- the hard real-time gate observed zero frames over 16.67 ms across all 1,170
  measured steps;
- PhysX GPU step: 1.92 ms mean / 3.16 ms p95 / 4.89 ms maximum; CUDA stress:
  0.18 ms mean / 2.41 ms p95 / 6.04 ms maximum (hybrid 540-bond threshold);
- 2,258 peak bodies and 1,068 awake bodies from 13,056 latent chunks, with
  1,047 splits and 3,136 projectile impacts;
- all 64 buildings partially fractured; 64 showed displaced pieces and 63
  showed falling debris; 3,778 chunks moved at least 0.5 m, 2,926 fell, and
  3,113 traveled at least 2 m;
- 7,707 chunks remained in supported components, 5,349 became dynamic, and
  zero buildings were heavily fractured or shattered;
- maximum fragment displacement was 22.93 m, maximum fall was 12.43 m, and
  split continuity drift stayed below `5e-5`;
- 1080p/30 render and NVENC encoding completed in 5.23 s (over 3× real time).

The CUDA stress crossover benchmark uses a 12,769-node/25,312-bond graph with
25 CGNR iterations. On the local RTX 4090 it measured 3.98 ms for the CPU
reference versus 0.65 ms of CUDA work and 0.77 ms host-observed, a 5.20×
speedup. Inputs cost 306,456 B H2D; a quiet compact break readback is 4 B.
High-rise graphs retain that measured crossover. The visual acceptance run
uses a 540-bond hybrid threshold so each intact 546-bond tower starts on CUDA
and falls back to CPU as fracture shrinks its graph. This recorded 3.68 seconds
of CUDA stress work with nonzero H2D/D2H traffic while retaining the latent
hierarchy and bounded active-body count.

The acceptance videos and synchronized sidecars are:

- `/root/recordings/blast-gpu-8x8-13k-mass1050-closeup.mp4`
- `/root/recordings/blast-gpu-8x8-13k-gpu-stress-acceptance.mp4`

GPU rigid-body execution and the high-rise stress crossover are both measured.
Contact routing and PhysX actor/shape topology edits remain CPU-side by API
design.

## Known limitations and deferred work

- PhysX Direct GPU API is not enabled in the production demo. The
  `blast_direct_gpu_p0` executable passed 1,000 actor add/remove cycles, 2,000
  shape migrations, stable survivor indices, device pose/velocity access, and
  `copyContactData` on PhysX 5.10.0. Direct mode also forces sleeping off and
  invalidates normal CPU pose/velocity access, so enabling it requires the
  custom settle/freeze and device snapshot paths first. The adapter uses regular
  `PxRigidDynamic`/`PxShape` APIs between `fetchResults()` and `simulate()` for
  fracture-driven actor creation, shape detach/attach, and mass updates.
- The integrated CUDA backend currently reads full bond impulses back for CPU
  Blast fracture-command generation on active large graphs. The resident GPU
  damage path already emits compact break indices; replacing that validation
  bridge with compact damage/split descriptors is the next PCIe optimization.
- Full snapshot/restore and fracture-frame resimulation are not implemented.
  Splits preserve motion by fitting child state and reconciling PhysX COM/inertia.
- Support-containing actors are kinematic `PxRigidDynamic` objects, allowing
  support status to change in place; they are not immutable static actors.
- Convex input is reduced to 64 points. ScenePack geometry that cannot produce a
  valid four-point hull is rejected.
- Debris TTL/collision tiers, sibling contact grace, and scoped resimulation
  remain future scaling work.

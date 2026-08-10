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
204-chunk tower has only 546 bonds, so the production default keeps those
inexpensive graph solves on CPU while PhysX GPU handles the growing rigid-body,
broadphase, and contact workload. The canonical benchmark deliberately sets
`--gpu-stress-min-bonds 540` so intact three-floor towers exercise CUDA before
fracture shrinks them onto the CPU path.

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
  --grid 12 --duration 30 --settle 1.5 \
  --snapshot-fps 30 --require-gpu --gpu-stress \
  --require-partial-destruction --chase-projectile \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=540 \
  --sim-arg=--projectile-mass-scale --sim-arg=1.8 \
  --sim-arg=--contact-force-scale --sim-arg=3.1 \
  --sim-arg=--excess-force-scale --sim-arg=0.012 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=20000 \
  --sim-arg=--require-varied-building-heights \
  --output /root/recordings/blast-gpu-varied-12x12-20k-optimized-benchmark.mp4
```

All project-generated videos should be written under `/root/recordings/`.
The same canonical run is available as
`bash demos/blast-stress-demo/record_stress_benchmark.sh [output.mp4]`.
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

The GPU-stress benchmark tuning is: 1,800 kg, 0.5 m projectiles at 25 m/s;
contact-force scale 93; stress-limit scale 0.8; released-load scale 0.012; a
4 m/s bounded separation impulse; 6 m/s linear and 8 rad/s angular fragment
caps; and a 3.2 s projectile lifetime. Every building receives one oblique
impact and every eighth building receives a lower opposing impact. Launches
are distributed over the available destruction window to avoid artificial
same-frame impact bursts while retaining citywide damage.

By default the city deterministically alternates actual one-, two-, and
three-floor support graphs. The shorter variants remove upper chunks and
remap their bonds rather than merely scaling the same rigid body. Use
`--uniform-building-heights` for the legacy all-three-floor layout.

`--require-min-authored-chunks N` makes latent scene scale an executable
contract. `--require-realtime` performs eight unmeasured warm-up steps to
allocate lazy PhysX/solver resources, then fails if any recorded 60 Hz step
exceeds 16.67 ms. Both the all-frame and destruction-only miss counts and the
maximum destruction-frame time are written to metadata.
`--require-varied-building-heights` additionally requires populated one-,
two-, and three-floor cohorts, preventing a uniform skyline from satisfying
the canonical benchmark by chunk count alone.

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
and retain their previous pose. The adapter exposes an active-body delta query
that evaluates one PhysX pose per emitting body rather than rescanning all
20,880 shapes. Mean state-export work on the current 12×12 run is 0.28 ms.

## Current benchmark

The benchmark baseline is now 12×12: 144 buildings, 20,880 authored chunks,
and 31.5 seconds of 1080p/30 video from a fixed 60 Hz simulation. Its skyline
contains 48 each of one-floor/83-chunk, two-floor/148-chunk, and
three-floor/204-chunk graphs:

- simulation: 13.01 s wall time (2.42× real time), 6.87 ms mean, 10.21 ms p95,
  and 16.09 ms maximum host frame;
- the hard real-time gate observed zero frames over 16.67 ms across all 1,890
  measured steps, including state export, with 0.57 ms worst-case headroom;
- PhysX GPU step: 2.34 ms mean / 3.46 ms p95 / 6.80 ms maximum; CUDA stress:
  0.06 ms mean / 0.49 ms p95 / 3.22 ms maximum (hybrid 540-bond threshold);
- 4,590 peak bodies and 964 awake bodies from 20,880 latent chunks, with
  1,760 splits and 7,001 projectile impacts;
- all 144 buildings fractured without shattering; 116 showed displaced pieces
  and 114 showed falling debris; 5,284 chunks moved at least 0.5 m, 4,617
  fell, and 4,394 traveled at least 2 m;
- 12,748 chunks remained in supported components, 8,132 became dynamic,
  maximum displacement was 22.44 m, maximum fall was 12.03 m, and split
  continuity drift stayed below `6.4e-5`;
- 1080p/30 Vulkan rendering and NVENC encoding completed in 8.34 s, 3.78×
  faster than video duration.

The CUDA stress crossover benchmark uses a 12,769-node/25,312-bond graph with
25 CGNR iterations. On the local RTX 4090 it measured 4.52 ms for the CPU
reference versus 0.68 ms of CUDA work and 0.75 ms host-observed, a 6.06×
speedup. Inputs cost 306,456 B H2D; a quiet compact break readback is 4 B.
High-rise graphs retain that measured crossover. The visual benchmark uses a
540-bond hybrid threshold, so each intact 546-bond three-floor tower starts on
CUDA and falls back to CPU as fracture shrinks its graph; shorter graphs remain
on CPU. The run recorded nonzero CUDA stress work and H2D/D2H traffic while
retaining the latent hierarchy and bounded active-body count.

CUDA inputs, status, and impulse output use persistent pinned staging buffers
and asynchronous copies, collapsing upload/solve/status waits into one
completion point. Island-aware solves use a size-tiered warm-start budget
(32 iterations below 1,024 bonds, 44 above), and unchanged converged CPU inputs
are skipped. Contact forces cross the bridge in batches, and body poses are
cached once per snapshot/contact phase.

The acceptance videos and synchronized sidecars are:

- `/root/recordings/blast-gpu-8x8-13k-mass1050-closeup.mp4`
- `/root/recordings/blast-gpu-8x8-13k-gpu-stress-acceptance.mp4`
- `/root/recordings/blast-gpu-varied-10x10-14k-stress-benchmark.mp4`
- `/root/recordings/blast-gpu-varied-12x12-20k-optimized-benchmark.mp4`

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

# Session Recorder — capture & replay destruction bugs

A drop-in **⏺ Session recorder** overlay sits on every destruction demo
(wall-demolition, tower-collapse, fractured-wall / -tower / -bridge,
fracture-policy, high-rise, mini-city). Press **● Record**, do the thing that
misbehaves (shoot a projectile, detonate the skyline, watch pieces jitter), press
**■ Stop**, then **⬇ Save**. You get a single gzipped file
(`<demo>-recording-<timestamp>.sim.json.gz`) you can attach to a bug report.

It is built to answer the hard-to-describe, hard-to-reproduce questions — *"when
I destroy this building, that piece lags / its velocity isn't what I expect / it
teleports a frame"* — by recording **enough** that the scenario can be analysed
(and largely reproduced) offline.

## What it captures

| Captured | Detail |
| --- | --- |
| **Per-frame body kinematics** | Every dynamic rigid body, every frame: position, orientation (quaternion), **linear & angular velocity**, keyed by Rapier handle. |
| **Inputs** | Every projectile spawn, external force, and gravity change — with the frame index and simulation time they happened at — captured automatically (clicks, meteor storm, detonate, …). |
| **Fracture / topology** | A baseline node→body map plus a delta event stream: `migrate` (a chunk moved to a new body / split), `detach`, `destroy`, `bodyRemoved`. Reconstructs the **current hierarchy** at any frame. |
| **Scalars/frame** | Active bond count, rigid-body count, projectile count, dt, sim time. |
| **Initial structure** | The full `ScenarioDesc` (nodes, bonds, fractured fragment geometry) + the core config + UI config — the ground truth to rebuild the scene. |
| **Per-frame timing (full session)** | Every frame's per-phase cost — physics step, stress solve, contacts, fracture (generate/apply), split planning, body/collider edits, snapshots, damage (7 sub-timers), spawn/cleanup — captured for the *whole* recording (not just a rolling window) and aligned 1:1 with the kinematic frames. The leaf phases sum to ~`totalMs`, so you can account for **every millisecond**. Captured by multiplexing onto `core.setProfiler`, so it coexists with the live frame-profiler overlay. |

## Performance

Recording is **allocation-free on the hot path**: each frame writes straight into
a pre-grown `Float32Array` (14 floats/body) and a handful of scalar columns — no
objects, no JSON, no `stringify` while recording, so there are **no GC pauses** to
distort the very frame times you are trying to capture. When *not* recording the
cost is a single boolean check. Serialisation + gzip happen only on **⬇ Save**.

A long mini-city / tall-tower session can be tens of MB raw, but the trace gzips
extremely well (motion is smooth). A soft 10-minute frame cap auto-stops a
forgotten recording.

## File format (`blast-sim-recording/v1`)

Plain JSON (gzipped). Heavy arrays are little-endian base64:

- `columns.{simTime,dt,bodyCount,activeBonds,rigidBodies,projectiles}` — parallel
  per-frame typed arrays.
- `bodies` — flat `Float32Array` of `Σ bodyCount` rows × `bodyStride` (14) floats;
  frame *i*'s rows start at `(Σ_{j<i} bodyCount[j]) * 14`. Layout in `bodyLayout`.
- `initialBodyByNode` / `nodeIndices` — the topology baseline.
- `events` — the input + topology timeline.
- `scenario`, `coreConfig`, `meta`, `profiler` — context.

In code, `decodeSimRecording(data)` (from `blast-stress-solver/rapier`) rehydrates
everything into typed arrays with `frame(i)` / `bodyInFrame(i, handle)` helpers.

## Inspecting a recording

A zero-dependency CLI prints a summary and answers example questions (it flags
position **jumps** larger than velocity predicts — the jitter/teleport signal):

```bash
cd blast/blast-stress-solver

# Summary (frames, body counts, bond Δ, event histogram, scenario size)
npm run inspect:recording -- ~/Downloads/mini-city-recording-….sim.json.gz

# Performance: where did every millisecond go (per-phase Σ/%, slowest frames)
npm run inspect:recording -- rec.sim.json.gz --perf

# One body's trajectory + speed, flagging suspicious jumps, over a frame range
npm run inspect:recording -- rec.sim.json.gz --body 42 --range 90 140

# The timeline (optionally filtered: projectile|force|gravity|migrate|detach|destroy|bodyRemoved)
npm run inspect:recording -- rec.sim.json.gz --events migrate

# Every body's row for a single frame
npm run inspect:recording -- rec.sim.json.gz --frame 120
```

## Wiring it into another page

Same shape as the frame-profiler overlay:

```ts
import { createRecordingOverlay } from 'blast-stress-solver/rapier';

const recorder = createRecordingOverlay({
  exportName: 'my-demo-recording',
  getProfilerExport: () => profiler.exportData(), // optional: embed timings
});

// after building / rebuilding the core (scenario + config are in scope here):
recorder.attach(core, { scenario, meta: { demo: 'my-demo', config: CONFIG } });

// optional, once per frame — only refreshes the live readout:
recorder.render();
```

Capture is automatic once you press ● Record — the recorder wraps the core's
`step` / `enqueueProjectile` / `applyExternalForce` / `setGravity`, so nothing in
the render loop needs to change.

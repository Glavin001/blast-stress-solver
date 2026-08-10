# Blast PhysX GPU Mini-City Recorder

Self-contained headless recording pipeline for the Blast PhysX GPU Mini-City:

```text
C++ simulation -> TWSTATE1 transforms -> offscreen wgpu/Vulkan -> h264_nvenc -> MP4
```

No X server, Xvfb, window, fake monitor, or swapchain is used. The renderer
draws instanced boxes and spheres into an offscreen Vulkan texture, reads back
three frames in flight, and streams ordered RGBA frames to FFmpeg/NVENC.

## Build

```sh
cd demos/blast-stress-demo/recorder
cargo build --release --locked
```

Runtime requirements are a Vulkan-capable discrete GPU, Vulkan loader and
driver, FFmpeg with `h264_nvenc`, `ffprobe`, and (optionally) `nvidia-smi`.

## Render an existing state

```sh
cargo run --release --locked -- render \
  --state /path/to/run.twstate \
  --frame-telemetry /path/to/run.simulation.frames.csv \
  --output /root/recordings/run.mp4 \
  --chase-projectile
```

The first overview pane orbits slowly. `--chase-projectile` additionally
replaces the fourth static pane with a close impact observer locked onto the
front-center buildings. The fixed world-space reference makes fragment motion
easy to distinguish from camera motion and avoids jumps between projectiles.

## Simulate and record

```sh
cargo run --release --locked -- record \
  --sim-bin ../build/blast_stress_demo \
  --grid 8 \
  --duration 18 \
  --settle 1.5 \
  --snapshot-fps 30 \
  --require-gpu --gpu-stress \
  --require-partial-destruction --chase-projectile \
  --sim-arg=--gpu-stress-min-bonds --sim-arg=540 \
  --sim-arg=--contact-force-scale --sim-arg=3.1 \
  --sim-arg=--excess-force-scale --sim-arg=0.012 \
  --sim-arg=--require-realtime \
  --sim-arg=--require-min-authored-chunks --sim-arg=10000 \
  --output /root/recordings/blast-gpu-8x8-13k-gpu-stress-acceptance.mp4
```

The default video is a timestamped
`/root/recordings/blast-physx-gpu-mini-city-NxN-*.mp4`. With no `--state`, the
recorder uses a temporary `*.tmp.twstate` and removes it after successful
encoding unless `--keep-state` is set. An explicit `--state` is always
persistent. `--output`, `--metadata`, `--sim-telemetry`, and
`--frame-telemetry` override their default sidecar paths. Repeat
`--sim-arg VALUE` for additional scene arguments. Keep explicit video outputs
under `/root/recordings/` as well.

The `record` command launches the simulation without `DISPLAY` and passes:

```text
--state PATH
--metadata PATH
--telemetry PATH
--frame-telemetry PATH
--grid N
--duration SECONDS
--settle SECONDS
--snapshot-fps FPS
--pane-width PIXELS
--pane-height PIXELS
[--require-gpu]
[--gpu-stress]
[--require-partial-destruction]
```

The simulation executable is responsible for writing the state and its
machine-readable metadata/telemetry sidecars. Useful metadata includes PhysX
mode, confirmed GPU activation, scene/body/chunk counts, peak awake bodies,
split timings, simulation throughput, GPU buffer capacities/high-water marks,
overflow warnings, and CUDA abort status.

The per-step `*.simulation.frames.csv` is synchronized to the state stream and
drawn into the output video. It includes physics wait, stress/contact/fracture,
export, frame budget, body/island, split, migration, contact, and continuity
metrics, including direct projectile-impact count and impulse. The recorder
writes the CUDA stress kernel time and H2D/D2H byte counts into the same
diagnostic overlay when those optional CSV columns are present. The recorder
writes `*.render.frames.csv` and
`*.render.summary.json` for CPU submission, Vulkan readback, overlay, NVENC pipe,
and end-to-end rendering throughput. It independently samples
`*.simulation.gpu.csv`, `*.simulation.gpu-summary.txt`, `*.render.gpu.csv`, and
`*.render.gpu-summary.txt` with `nvidia-smi`; missing `nvidia-smi` is non-fatal.

## TWSTATE1 binary layout

Every integer and float is little-endian. Floats are IEEE-754 binary32. Fields
are packed exactly as listed with no C/C++ struct padding.

Header (156 bytes):

```text
offset  size  field
0       8     ASCII magic "TWSTATE1"
8       4     u32 version (= 1)
12      4     u32 snapshot fps
16      4     u32 frame count
20      4     u32 pane width
24      4     u32 pane height
28      4     u32 scene instance/building count (informational)
32      4     u32 camera count (= 4)
36      4     f32 captured duration seconds
40      4     f32 settle seconds
44      112   four cameras, 28 bytes each:
                  f32 eye[3]
                  f32 direction[3]
                  f32 vertical field-of-view degrees
```

The header is followed by a record stream:

```text
Actor definition (record type 1):
    u8  record_type (= 1)
    u32 actor_id
    u8  part_tag
    u32 shape_count
    repeated shape_count times:
        u8  shape_kind       (1 = box, 2 = sphere)
        f32 parameters[3]    (box half-extents xyz; sphere radius in x)
        f32 local_position[3]
        f32 local_rotation[4] (quaternion x, y, z, w)

Frame (record type 2):
    u8  record_type (= 2)
    u32 frame_index
    u32 update_count
    repeated update_count times:
        u32 actor_id
        f32 world_position[3]
        f32 world_rotation[4] (quaternion x, y, z, w)
        u8  sleeping          (0 = false, nonzero = true)

End:
    u8 record_type (= 255)
```

Actor IDs must be zero-based and contiguous in definition order. An actor must
be defined before a frame references it. Frames must be contiguous from zero,
and their total must equal the header frame count. Frame updates are deltas: the
exporter omits unchanged poses and sleep state, and omitted actors retain their
preceding values. The first frame updates every initially visible actor. Actors
become visible on their first update and remain visible; actors that may appear
later can be defined up front and first updated when spawned.

The stream intentionally has no actor count or byte offsets, preserving the
existing proven `TWSTATE1` format byte-for-byte.

## Source provenance

This recorder is ported from the proven self-contained Rust implementation in
`/root/workspace/physx-tower/tower-demo`. The TWSTATE1 reader, camera
choreography, instanced primitive renderer, triple-buffered Vulkan readback,
NVENC integration, and NVIDIA telemetry retain that implementation's structure.
It has no build-time or runtime dependency on the source workspace. Existing
source license/copyright headers were retained; the upstream Rust/WGSL files
did not contain additional per-file headers.

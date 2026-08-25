---
name: blast-core-development
description: Build and test the engine-neutral destruction core and its Rapier/PhysX adapters, and dogfood a work-in-progress library against a consuming app. Use when changing anything under blast-stress-solver-rs/src/{backend,backends,pipeline,scene_pack,ids} or the C ABI in blast/physx_backend.
---

# Developing the blast-stress-solver core

The library is an engine-neutral pipeline plus thin per-engine adapters. Two
engines exist so that engine assumptions cannot leak into the core: if a change
only passes on one, the core is wrong.

## Running the suite

```bash
cd blast/blast-stress-solver-rs
cargo test --features rapier,scenarios,physx
```

Everything must pass on **both** Rapier and PhysX. `--features physx` builds the
C++ adapter in `blast/physx_backend/`; PhysX dlopens `libPhysXGpu_64.so`, so the
build script emits an rpath. `cargo:rustc-link-arg` does **not** propagate from a
dependency's build script to a dependent crate's test binaries -- a consuming
crate needs its own `build.rs` emitting the same rpath, or GPU silently reports
unavailable on a machine that has a GPU.

## The C ABI is the sharp edge

`physx_backend.h` and the `#[repr(C)]` mirror in `physx_backend.rs` must agree
**field for field, in order**. The struct is passed by pointer, so a field out of
place is silently reinterpreted memory, not a compile error. A mismatch here once
caused a 36-byte out-of-bounds read per material.

`tests/ffi_abi_test.rs` pins this, and includes a case proving the *old* layout
would now be rejected -- when you extend a struct, extend that test the same way,
or it stops discriminating.

## Rules the design depends on

- **`BackendHandle` is not `Ord`.** Rapier handles are not ordered; `sort_key()`
  is the only ordering. Do not add an `Ord` bound.
- **Writes must be elided when unchanged.** Re-stamping a body or shape wakes a
  sleeping PhysX actor; doing it every tick once held ~600 of ~735 bodies
  permanently awake. Every command path compares before writing and counts
  `writes_elided`.
- **Poses in events are COM-frame, never the actor frame.** Exactly one child per
  split reuses its parent's body and keeps the parent's origin, so raw actor
  poses draw that island's chunks one COM-height off.
- **Bounds on id packing are hard asserts, not `debug_assert`.** As
  `debug_assert` they vanish in release and corrupt island membership silently.

## Things that are not what they look like

**`generateFractureCommands` is a damage stream, not a break stream.** A command
is issued *every tick* a bond is overstressed while its health is still positive,
and its `health` field is the damage applied, not what remains. Counting commands
as breaks reported 1067 on a 546-bond tower. A bond breaks when its health
crosses zero -- read it via `ext_stress_solver_get_bond_healths`.

**A bond's `userdata` is always 0**, so it cannot serve as a bond index. Resolve
the index from the node pair.

**Authored packs are multi-material.** `ExtStressSolver::new` puts every bond on
material 0; use `new_with_materials` for a real pack. Flattening a table is not a
rescale -- foundations are authored strongest *because* they carry the most load,
so flattening leaves them weaker than their own load and the structure collapses.

## Dogfooding against a consumer

Point a consuming app at a WIP library with `BLAST_ROOT`:

```bash
BLAST_ROOT=/root/workspace/blast-stress-solver-2/blast \
  cargo test -p vibe-land-destruction --features physx,blast-core -- --test-threads=1
```

Use `--test-threads=1` when several tests each construct a physics world;
the process-wide PhysX runtime is refcounted and shared, and concurrent
construction is exactly what `tests/physx_runtime_thread_safety_test.rs` guards.

The consumer is where integration bugs surface that the library's own tests
cannot see -- collision filter data, host raycasts, and node lookup by live world
position rather than authored centroid were all found this way.

## Adjudicating a failing golden

Several checked-in expectations were recorded against older assets or an older
port. Before changing an assertion, **prove** which side is wrong and record the
proof in the test, not the commit message. Two worked examples live in
`examples/`: `actor_count_invariant.rs` recomputes actor count by independent
union-find, and `strong_wall_adjudication.rs` measures peak bond stress against
the material's own fatal limit. A golden changed without that kind of evidence is
indistinguishable from making the test pass.

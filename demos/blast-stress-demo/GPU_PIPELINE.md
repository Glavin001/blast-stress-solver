# Device contact and stress integration — 2026-09-05

This implementation establishes an executable PhysX → contact decoder → load producer → stress solver chain in one CUDA context. Contact payloads and node loads remain on the GPU between those stages. It is a validated integration primitive, not a replacement for the city adapter, a complete fracture transaction, or a city-scale performance result.

The tested engine is **PhysX 5.10.0**, from `/root/PhysX/physx`. This repository's bundled PhysX source is 5.6.1; do not mistake it for the engine linked by the current Vast build. CUDA is 12.8.93; the test GPU is an RTX 4090 with driver 595.71.05.

## Interfaces and ownership

- `ExtStressPhysXDirectGpuMotionBuffer` captures poses and linear/angular velocities into persistent device allocations. Restore uses those allocations directly. Only GPU indices cross from host to device. Three ordered PhysX copies share a completion event and one host wait. Invalid captures invalidate the previous checkpoint; removed actors are rejected before restoration; all three transfers must succeed. The CPU fallback is used only for ordinary scenes. Keep supplied actors alive through the next capture or release. This is a **motion checkpoint**, without topology, force-journal, sleeping, contact-cache or complete shared-scene rollback semantics.
- `ExtStressPhysXDirectGpuContactDrain::copyContactsDevice` returns borrowed device records and a completion event. Wait on that event before consuming the records or starting another simulation. Finish consuming the records before another drain or release. Each nonzero normal impulse and friction-anchor impulse retains its world application point, actor identity, and shape transform-cache identifiers. Record order is unspecified. A nonzero device overflow flag invalidates the result; consumers must not apply a partial load set. `copyContacts` is the synchronous host reference/debug interface, and `lastCopyComplete` distinguishes empty success from failure.
- `ExtStressGpuSolver::solveDevice` accepts a device load array in solver node order, its exact element count, and an optional producer event. The array/event must remain valid through return. Without an event the producer must already be complete. A device-to-device copy keeps captured graph addresses and the previous input baseline stable. Solving is synchronous at the public boundary, like the host API. There is no load download/re-upload and no implicit bond-impulse readback. The existing GPU bond-stress walk can consume the solver's resident impulses.

The device input path preserves converged-only settled-island skipping. A GPU comparison against the previous device inputs produces one dirty flag per island; the current host scheduler reads those compact flags. Host input after device input conservatively invalidates the obsolete host comparison baseline. The implementation retains the existing CG/CGLS kernels, tolerance, damage formulas, materials and iteration policy.

Two particularly consequential defects were removed. The old motion prototype overwrote its device checkpoint from stale CPU motion and treated any successful component transfer as a successful restoration. The old contact drain returned actor pairs with zero positions and impulses. Neither was a valid destruction input path.

There is also a local PhysX 5.10 workaround: `PxgGpuNarrowphaseCore::copyContactData` returns early when `mTotalNumPairs == 0`, leaving both the output count and completion event unchanged. The wrapper resets its count and seeds an event on its own stream before the call. The nonempty engine path waits for that event and re-records it after copying; the empty path already has a valid zero-count completion. This also orders reuse after the previous decode, without a steady-state host synchronization. The installed engine was not modified.

The CMake link places the CUDA runtime before PhysX's static library group. Otherwise PhysX's private `CudaKernelWrangler` can capture application CUDA registration symbols and produce unresolved engine-only `PxGpuCudaRegister*` references. The decoder builds when `BLAST_ENABLE_CUDA_STRESS` is enabled; other builds report the contact drain unavailable instead of fabricating contacts.

## Validation

Configure against the actual 5.10 installation, then build:

```sh
cmake -S demos/blast-stress-demo -B demos/blast-stress-demo/build \
  -DPHYSX_ROOT=/root/PhysX/physx \
  -DPHYSX_LIB_DIR=/root/PhysX/physx/bin/linux.x86_64/release \
  -DBLAST_ENABLE_CUDA_STRESS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build demos/blast-stress-demo/build \
  --target direct_gpu_resim_test direct_gpu_contact_test gpu_device_input_test gpu_stress_test -j8
ctest --test-dir demos/blast-stress-demo/build \
  -R 'blast_stress_(physx_direct_gpu|gpu_)' --output-on-failure
```

Run GPU tests exclusively; the campaign paused only the idle, checkout-owned city deployment and restored the same binary/environment afterward.

All four selected tests passed. Compute Sanitizer memcheck reported **zero errors** for both the contact pipeline and device-input regressions.

- Motion tests compare device-observed position, orientation, linear velocity and angular velocity after replay. They cover growing allocations, repeated restore, recapture, invalid capture and actor removal. Isolated runs at 32 and 5,000 bodies passed. At 5,000 bodies, 100 capture/restore repetitions averaged 0.0402/0.0422 ms respectively. These numbers measure motion copying only, not fracture remapping, a whole city tick, or a matched comparison with the production checkpoint.
- The contact test uses an actual sliding impact. Eighteen GPU contact-to-stress submissions passed; the largest difference between decoded impulse and measured momentum change was **9.53674e-7 kg·m/s**. It separately checks a zero-resultant contact couple with nonzero torque, output overflow, and the transition from live contacts to an empty scene. The integration fixture's load kernel is deliberately small; production still needs the real compound-shape/node mapping, local-frame load construction and full fracture semantics.
- The device-input test uses a deliberately delayed producer in a separate nonblocking stream. Cold/warm solves, unchanged islands, a changed middle island, host/device transitions, bond removal and cold restart all matched the host-input solver with maximum observed relative difference **zero**. It also mutates producer storage immediately after return to test snapshot ownership. Device load submission reports zero load bytes through the host bus.
- The existing independent CPU/GPU stress equivalence test passed.

Before enabling this path for the city, complete shared-world rollback and force replay; preserve gameplay queries and sleep semantics; implement production shape-to-node load mapping and fracture provenance; remove eager impulse readback only after auditing every consumer; and validate numerical load paths and multi-impact destruction at equal material settings. The small fixtures here establish interface correctness, not that the large-structure solver has converged or that an iteration budget can be reduced.

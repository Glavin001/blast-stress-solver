# Settled-bond stress-recompute skip — bit-exactness A/B + timing

`ExtStressSolverImpl::updateBondStress()` recomputes every bond group's stress every frame. On the
**island-aware + skip-settled** path (what the mini-city/WASM demo runs, and what other work is making
default-on), the solver already leaves *settled* islands' impulses untouched — so their stresses are
unchanged too. This change skips recomputing them, when it is provably safe to do so.

## What's measured

Per the native profiling that motivated this, on a settled city `updateBondStress` is **~76% of the
"CGNR solve (WASM)" timer** (≈0.62 of 0.82 ms) — bigger than the CGNR iterations themselves on settled
frames. (The TS `idleSkip` already short-circuits *fully* idle frames, so the recoverable window is
"solver runs + some islands settled + nothing overstressed" — settling tails and gentle localized
activity, common after an impact.)

## The two gates (why it's output-preserving)

A bond group's stored stress is still exact iff **nothing it depends on changed**. Its inputs are the
bond *impulses* and the bond *health* (area). So we skip group `i` only when:

1. **Its island was skipped as settled** (`StressProcessor::wasBondSettledSkipped(i)`) — impulses are
   byte-identical to last solve (velocities bit-identical + the island had converged). Skip-settled
   already guarantees this; we just expose it per bond.
2. **Nothing was overstressed last frame** (`m_prevOverstressedBondCount == 0`). Bond health only
   changes via stress damage, which only fires on overstressed bonds. So no overstress last frame ⇒ no
   damage applied ⇒ health frozen this frame. This is the subtle case: a bond can be *partially
   damaged while its node velocities stay frozen* (damage doesn't move the Rapier body), which gate 1
   alone would miss — gate 2 forces a full recompute on the next frame after any overstress.

A skipped group can never be newly overstressed (frozen impulses + frozen health ⇒ frozen stress), so
it contributes 0 to the overstressed count exactly as before; no broken bond is missed (breaking needs
overstress, which disables the skip). Topology/health changes already reset the settled baseline.

## Run it

```bash
cd blast/source/sdk/extensions/stress/bench
./run.sh
```

Builds the Ext solver in the web build's scalar config, **island-aware + skip-settled ON**, twice
(default vs `-DSTRESS_NO_BONDSTRESS_SKIP`), and asserts a byte-identical hash of the stress output.

## Results (clang 18, scalar, 28-building city, 7020 nodes / 17712 bonds)

| harness | what it exercises | result |
|---|---|---|
| `ext_bondstress_settle_ab` | settle + 3 buildings driven by time-varying forces (25 settle) | **bit-exact**; steady-state solve **0.65 → 0.22 ms (≈2.9×)** |
| `ext_bondstress_fracture_ab` | weak material, full generate→apply fracture cycle (2775 bond-fractures, partial damage, splits, 111 islands) | **bit-exact** through the partial-damage transition |

The hash covers per-frame debug-render stress (per-bond), per-actor excess forces / residuals,
overstressed / bond / actor / island counts, and (fracture harness) every fracture command + split.

This only affects the island-aware path: with island-aware off, `getIslandsSkipped()==0` ⇒ the skip
is never taken ⇒ the whole-graph path is unchanged (the Rust `bond_stress`/`kinematic_invariants`
suites, which run island-aware off, pass identically).

## Scaling (`./run-scaling.sh`)

The saving is proportional to the **settled bonds it skips**, so it grows with city size and with how
much of the scene is at rest.

**City-size sweep** — 3 buildings driven (localized action), the rest settled:

| scene | bonds | settled solve OFF | ON | saved/frame | ratio |
|---|--:|--:|--:|--:|--:|
| tiny | 8.4k | 0.58 ms | 0.37 ms | 0.2 ms | 1.6× |
| small | 34k | 1.63 ms | 0.84 ms | 0.8 ms | 1.9× |
| medium | 90k | 3.73 ms | 1.55 ms | 2.2 ms | 2.4× |
| large | 219k | 8.73 ms | 3.37 ms | **5.4 ms** | 2.6× |
| xlarge | 438k | 17.43 ms | 5.75 ms | **11.7 ms** | 3.0× |

**Activity sweep** — fixed 219k-bond city, varying how many of 100 buildings are active:

| active | settled | solve OFF | ON | saved/frame |
|--:|--:|--:|--:|--:|
| 0 | 100/100 | 8.22 ms | 2.50 ms | 5.7 ms |
| 5 | 95/100 | 9.16 ms | 3.71 ms | 5.5 ms |
| 25 | 75/100 | 13.77 ms | 8.64 ms | 5.1 ms |
| 50 | 50/100 | 18.79 ms | 15.39 ms | 3.4 ms |
| 100 | 0/100 | 30.05 ms | 29.20 ms | 0.9 ms |

The absolute saving holds at ~5 ms while ≥75% of the city is settled (the steady state and the whole
post-destruction settling tail); it fades to ~0 only when the *entire* city is in motion at once
(those frames are physics-bound anyway, and the skip then correctly costs nothing). Numbers are scalar
clang 18, `-O3`; the SIMD CGNR path speeds up iterations but not this trig/area math, so the skip is an
even larger share of the solve there.

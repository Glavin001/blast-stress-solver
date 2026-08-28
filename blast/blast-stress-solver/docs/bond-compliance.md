# Why thin joints are overloaded, and what would actually fix it

## The symptom

Bonds with a small contact area report enormous stress under ordinary load.
Measured on a 47,631-chunk masonry structure, mean bond utilisation by contact
area:

| contact area | mean utilisation |
|---|---:|
| 1e-3..1e-2 m² | 0.26 |
| 1e-2..1e-1 m² | 0.17 |
| 1e-1..1e0 m² | 0.03 |
| 1e0..1e1 m² | 0.01 |

Twenty-six times more load per unit strength on the smallest joints than on
the largest. Those joints crack, hand their share to their neighbours, and the
building comes apart under its own weight.

The current mitigation (`lib/sliver.mjs`) deletes contacts below 2% of the
smaller chunk's face. It works — 424 overloaded joints became 1 — but it is a
mesh filter standing in for a missing physical property, and it is the thing
this document exists to eventually delete.

## The cause

The solve is **underdetermined**, and its tie-break ignores area.

Every authored pack has far more bonds than dynamic nodes:

| pack | dynamic nodes | bonds | bonds per node |
|---|---:|---:|---:|
| parking-garage | 3,350 | 9,054 | 2.70 |
| algedra-tower | 7,836 | 19,705 | 2.51 |
| park-432 | 19,701 | 48,763 | 2.48 |
| minas-tirith | 44,885 | 165,897 | 3.70 |

The system solved is `B J = -v`: find bond impulses `J` (one per bond, N of
them) whose effect on the nodes cancels the applied velocity `v` (M of them).
With N ≈ 3M there are many exact solutions, and CGNR converges to the
**minimum-norm** one — the `J` minimising `Σ J²`.

That objective treats a 5 mm seam and a two-square-metre bearing face as
equally willing to carry load. Nothing in it knows about area. Area only
appears afterwards, when stress is computed as `force / area` — so the sliver
is handed a full-sized share and then divided by almost nothing.

## The fix

Minimise the right thing. A joint's stiffness scales with its area
(`k = EA/L`), and the energy stored carrying impulse `J` through stiffness `k`
goes as `J²/k`. So the physically correct tie-break is the minimum-ENERGY
solution:

    minimise  Σ J²/A     rather than     Σ J²

A thin joint then attracts force in proportion to its area and carries almost
none, which is what a compliant joint does in reality — and the sliver filter
becomes unnecessary, because a sliver stops being handed load in the first
place.

### How, concretely

Weighted minimum norm is ordinary column scaling. Substitute `J = S J'` with
`S = diag(sqrt(A_b / A_ref))`. Then

    min ||J'||  subject to  (B S) J' = -v

gives `J = S J'`, which is the minimiser of `Σ J²/A`. Solve for `J'`, then
scale back.

Worth being precise about why this works here and would not in general: column
scaling is normally just a preconditioner and leaves the answer alone. It
changes the answer for an UNDERDETERMINED system, because it selects a
different member of the solution space — which is exactly the degree of
freedom being mis-spent.

Implementation shape:

1. `BondMatrix` (`source/shared/stress_solver/bond.h`) gains a per-column scale
   pointer and an N-sized scratch.
2. `BondMatrixOps::rmul` scales `x` by `s` before applying `C`;
   `lmul` scales the result by `s` after applying `C^T`.
3. `StressProcessor` fills the scale vector from bond area, in both the
   whole-graph and per-island paths (`stress.cpp`).
4. **The CUDA kernel must do the same** (`NvBlastExtStressGpu.cu`). This is not
   optional: the server runs the GPU path, and a CPU-only change would mean the
   two backends disagree about the physics — the kind of split that is very
   hard to notice and very expensive to debug.

An equivalent formulation is Tikhonov regularisation of the normal equations,
`(BᵀB + diag(α)) J = Bᵀ(-v)` with `α ∝ 1/A` — the same thing other engines call
constraint force mixing. That needs the CG recurrence itself modified rather
than just the operators, so column scaling is the cheaper route to the same
place.

## Why it is not done yet

It has to land in both backends at once, and a subtle error in either produces
physics that is wrong everywhere without being obviously wrong anywhere. That
wants a session that starts with it, and a rig that can hand-check the answer:
`rig-column` is the right one, because its single column under a known cap has
a closed-form utilisation (0.13 MPa against a 48 MPa limit, 0.003) that can be
checked by arithmetic rather than by comparison.

## What it would let us delete

- `SLIVER_FRACTION` in `lib/sliver.mjs`, and the whole cull with it.
- Probably some of `BLAST_BEND_MAX_GAIN`'s conservatism: the cap is there
  because the section-modulus term amplifies noise on small joints, and small
  joints would no longer be carrying the load that makes that noise matter.

Two of the three non-physical knobs in the system, removed by modelling the
thing they stand in for.

## Confirmed against the buildings

The structural audit (`destruction/tests/structural_audit.rs`) was run over the
authored set at 32 iterations. Two buildings never reach equilibrium, and both
fail the same way:

| building | settles | peak | over yield | worst joint |
|---|---|---:|---:|---|
| house-1story | 0.15 s | 0.45 | 0 | — |
| house-2story | 0.17 s | 0.55 | 0 | — |
| villa-savoye | 0.30 s | 0.98 | 0 | — |
| parking-garage | 1.17 s | 1.16 | 1 | column→slab, 0.040 m² |
| algedra-tower | never | 2.48 | 27 | slab→beam, 0.120 m² |
| park-432 | never | 2.52 | 53 | slab→slab, 0.067 m² |

Every overloaded joint named is a SMALL one — 0.015 to 0.14 m² — in buildings
whose hottest joint class averages 0.02 to 0.10. The load is not high; it is
landing in the wrong places, which is this document's subject.

### The authoring lever was tried, and is not enough

If small joints are the problem, coarser fracture makes bigger ones. Tested on
the tower by raising concrete cell area 2.5x and the chunk cap 2.3x:

    peak 2.48 -> 2.36,  joints over yield 27 -> 15

An improvement, and nowhere near sufficient — it still never settles, and the
pack now fails its own monolith check, because chunks that big are no longer a
fractured building. Reverted.

That is the useful negative result: this cannot be authored around. The two
failing buildings need the solve to stop handing small joints a full share of
the load, which is the fix above.

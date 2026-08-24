# Chunk crushing

Destruction in this library was bond-only: the stress solver damaged the joints
between chunks, Blast split the actor when the graph disconnected, and every
chunk that existed before an impact still existed after it. A wall could come
apart; it could never lose material. Hit it hard enough and intact rigid bodies
were pushed back, which is why bond-only destruction reads as explosive rather
than as damage.

Chunk crushing adds the other half: a chunk whose own material is overwhelmed is
**comminuted** and leaves the simulation as dust. Both mechanisms run from the
same solve, and a single impact typically produces both -- most of a wall
separates along its joints while the small region under the hit is ground up.

Opt-in per material. A pack that authors no `crush` block behaves exactly as it
did before this existed, verified by running the crush-enabled reference
building with `--no-crush` against the plain v2 pack and comparing physics.

## What NvBlast already provided, and what it did not

Worth stating because it explains the shape of the implementation:

| Exists in NvBlast | Missing |
|---|---|
| Per-chunk health storage (`FamilyHeader::getLowerSupportChunkHealths`) | The stress solver never generated chunk fractures -- `fillFractureCommands` hard-nulled them |
| `NvBlastChunkFractureData` + hierarchical apply | Chunk health cannot hold PARTIAL damage: any positive damage on a support chunk detaches it immediately, regardless of health remaining |
| A chunk fracture zeroes every incident bond and drops the node from the island graph -- exactly the right graph surgery | A leaf chunk at zero health is a dead end. `partitionSingleLowerSupportChunk` returns before `release()`, so the actor persists forever, inert and still visible. There is no native pulverize |
| `NvBlastActorDeactivate` | ...which the damage pipeline never calls |

So: the graph surgery is reused, progressive damage lives in the solver (it
cannot live in Blast's health array), and the physics-side removal is ours.

## The model

Each solve, for every chunk on a crush-enabled material, build a Cauchy stress
tensor by the **Love-Weber (virial) sum** over the forces acting on it:

```
sigma = (1/V) * sum over contacts of sym( branch (outer) force )
```

Bond forces are attributed to member bonds by area share; external contacts
enter through the position-aware `addForceAt` (the ordinary `addForce` discards
the application point, which the bond solve does not need but a stress tensor is
built from). Gravity and centrifugal loads are excluded -- they are body forces,
not surface tractions. The bond's angular impulse is excluded too: a
self-equilibrated couple is antisymmetric and vanishes under symmetrization, so
a Cauchy mean stress genuinely does not carry it. Bending-driven edge spall is
therefore not represented.

Reduce to invariants and yield against a **Drucker-Prager cone with a pressure
cap**, with tension excluded:

```
p = -trace(sigma)/3              positive in compression
q = sqrt(1.5 * (s:s))            s = sigma + p*I

crushing requires p > 0
excess = max( q - (cohesion + frictionSlope*p),  p - capPressure )
```

Flow is **Perzyna overstress viscoplasticity**; damage is that plastic work per
unit volume normalized by the specific comminution energy:

```
epsdot_p = excess / crushViscosity
D += excess^2 * dt / (crushViscosity * crushEnergy),   pulverized at D >= 1
```

Three properties of that law are what make it behave:

- **Quadratic in overstress.** A chunk barely past yield takes a very long time
  to comminute; one hit hard enough to sit far outside the surface goes almost
  at once. One material covers both "survives ordinary abuse" and "pulverizes
  under a real hit" with no second threshold to author.
- **No strain measurement needed.** A chunk buried in a collapse and loaded only
  through its bonds comminutes exactly as a struck one does. An earlier revision
  drove damage from a contact-closing-rate surrogate and never fired on that
  case at all -- which is the case that matters for collapsing structures.
- **Nothing accumulates below yield**, so a structure standing under its own
  weight never grinds itself to dust however long it stands.

Tension is excluded because comminution is compressive: a chunk in net tension
cracks, which the bond model already represents. Without the cutoff the cone's
own limit falls with pressure, so unconfined chunks crushed MORE readily -- the
opposite of the intent, and the failure mode the model exists to avoid.

### The energy bill

`crushEnergy` is charged, not merely referenced. With `applyCrushResistance`
(default on) each damage increment extracts `dD * crushEnergy * volume` from the
crushing body's kinetic energy:

```
J = M * (v - sqrt(v^2 - 2*dE/M)),  capped at M*v
```

so resistance can stop a crusher but never bounce it. Charged per payer against
a running velocity estimate -- several chunks billed to one ball share ONE
kinetic-energy budget, and summing individually clamped impulses reverses the
payer.

Charges are **deferred to the resimulation pass**. Contact relative velocities
are sampled after `fetchResults`, when PhysX has already done its rigid momentum
exchange, so an immediate charge extracts from an already-stopped ball. The
energy the payer wrongly keeps only comes into existence on the resim pass:
motion is rewound to full speed, the crushed chunks are gone, and the contact
that stopped it is never re-issued. **Resistance therefore needs
`--resim-passes >= 1` to bite on hard impacts**; at `resim-passes 0` crushing
still removes material but the crusher is not slowed.

Bond-borne crushes deep inside a structure charge nobody: their load path is the
structure itself and the reaction is already inside the solve. A structure's own
debris never pays either -- internal contact impulses dwarf the impactor's, and
that energy exchange is likewise already in the solve.

## API

Authoring is ScenePack v3 (`materials[].crush`, `nodes[].m`) -- see
`SCENE_PACK_FORMAT.md` for the schema, units, and how to derive the cone from a
material's own compressive strength rather than dialling it.

Runtime, on `ExtStressPhysXDestructible`:

```cpp
// Input: one new optional field per contact.
contact.worldRelativeVelocity = otherVelocity - myVelocity;  // rate hardening
contact.otherActor            = theOtherBody;                // who pays

// Readback, authored-node-indexed, valid on an intact motionless structure.
getNodeCrushUtilisation(float*, uint32_t);   // 1 = at yield; 1/u = safety factor
getNodeCrushDamage(float*, uint32_t);        // [0,1], 1 = pulverized
getNodeStressInvariants(float* p, float* q, uint32_t);
isCrushEnabled();

// Events: exactly-once drain, or override the frame hook.
drainChunkDestroyedEvents(ExtStressPhysXChunkDestroyed*, uint32_t);
ExtStressPhysXFrameHooks::onChunkDestroyed(destructible, events, count);
```

`ExtStressPhysXChunkDestroyed` carries `mass`, `volume`, `worldPose`,
`linearVelocity` (at the chunk, not the body's centre of mass),
`angularVelocity`, and the `peakPressure`/`peakDeviator` that destroyed it --
everything a consumer needs to spawn a momentum-matched dust cloud instead of
guessing. `mini_city_main.cpp`'s `DustPool` is the reference consumer.

`getNodeCrushUtilisation` is the number to author against: the crush analogue of
a joint's safety factor, readable before anything breaks.

## Measured behaviour

Reference building (`assets/reference/reference-building-crush.json`), CPU,
`--resim-passes 1`:

| Load | Peak crush utilisation | Chunks comminuted |
|---|---|---|
| Gravity only | 0.12 (safety factor 8.4) | 0 of 64 |
| Ordinary impact | 3.3 | 0 |
| 2x projectile mass | 4.7 | 3 (4.7%) |
| 8x | 6.2 | 4 (6.3%) |
| 16x | 8.0 | 7 (10.9%) |

The punch-through wall (`assets/reference/crush-wall.json`), the scenario where
crushing is the behaviour rather than a detail:

| Load | Crush OFF | Crush ON |
|---|---|---|
| 1.0-1.2x | ball stopped, wall intact-looking | ball stopped, 5-31 chunks bitten out |
| **1.4x** | **plug blown through, ball exits at 36.5 m/s** | **wall absorbs 5.2 MJ of the ball's 6.3, ball stopped at 2.7 m/s** |

The mid-energy row is the one that cannot be reproduced any other way: bond
fracture plus resimulation severs 55 joints and the wall is internally rubble,
yet friction and jamming hold every block in place and nothing visible happens.
Removing material does not require dislodging it.

City scale, GPU, 400 calibrated buildings / 16,556 chunks:

| | Crush OFF | Crush ON |
|---|---|---|
| Shattered structures | 204 / 400 | **167 / 400** |
| Heavily damaged but standing | 196 | **233** |
| Peak bodies | 12,112 | 11,870 |

37 more buildings survive in a damaged-but-recognizable state: the local bite
absorbs energy that would otherwise propagate into total collapse.

### Cost

Isolated on CPU, the physics itself is ~20-40 ns per chunk per frame -- O(nodes
+ bonds) against a CGNR solve that is O(iterations * bonds). The larger cost is
bookkeeping: three per-tick O(n) FFI damage readbacks, roughly 0.3-0.5 ms/frame
at 16k chunks, consolidatable to one if it ever matters. At 6k chunks the
whole-frame difference is inside run-to-run noise. Crushing is not the real-time
constraint at any scale measured; the resimulation pass and PhysX contact volume
are.

## Reproducing

```sh
# every published A/B comparison, or name scenarios individually
demos/blast-stress-demo/record_crush_videos.sh /tmp/out
demos/blast-stress-demo/record_crush_videos.sh /tmp/out wall-punch

# the behavioural evidence
ctest --test-dir demos/blast-stress-demo/build -R crush --output-on-failure
```

`chunk_crush_test` isolates one claim per test, half of them pinning things
crushing must NOT do. Four end-to-end ctest gates run the reference building and
the wall at the production resim setting.

## Known limits

- **Requires `graphReductionLevel 0`.** Reduction merges chunks into aggregate
  solver nodes, so a per-chunk stress tensor would describe the aggregate.
  Creation fails rather than reporting a plausible wrong number.
- **Resistance requires `resim-passes >= 1`** to slow a fast penetrator, for the
  reason given above. Removal itself works at any setting.
- **Bending is not represented** in the mean stress.
- **A chunk with no bonds and no contacts never crushes**, which follows from
  excluding body forces from the virial.
- **Pancaking floors do not decelerate from grinding what is below them.**
  Bond-borne crushes charge nobody; that would need payer attribution through
  the bond graph.
- **TS/Rapier and Rust/Rapier do not implement crushing.** The shared
  conformance digest deliberately stays on the v2 fixture, and the TS/WASM
  bridge is separately ABI-skewed against the current material table (it passes
  5 arguments to a 7-argument `create`) -- repair that before wiring Rapier.
- **Dust rendering is pool-based.** TWSTATE1 declares every actor before frame 0
  and cannot spawn or despawn, so crushed chunks are parked out of frame and
  dust is a fixed pool of pre-declared motes. Real emission needs a format that
  can carry events.

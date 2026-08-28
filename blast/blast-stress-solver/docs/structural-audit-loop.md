# Fixing a building with the structural audit

The loop, and what is already known so nobody rediscovers it.

## The loop

    cd /root/workspace/vibe-land-2
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/root/PhysX/physx/install/linux-clang/PhysX/bin/linux.x86_64/release"
    AUDIT_SECS=45 AUDIT_PACKS=<name> cargo test -p vibe-land-destruction \
      --features cuda-stress --test structural_audit --release -- --ignored --nocapture

Read the card, change the geometry in
`blast-stress-solver/blast/blast-stress-solver/structures/<name>.mjs`, rebuild:

    cd /root/workspace/blast-stress-solver/blast/blast-stress-solver/structures
    node build.mjs <name> --emit-vibe-land /root/workspace/vibe-land-2

then re-audit. The build has its own gates -- interpenetration, overhang,
grounding, monolith -- and they must stay green; they catch the geometry
mistakes before the GPU does.

## Reading the card

    settles 5.5s | peak 1.16 -> 0.99 (-0.17) | 0.0 joints past yield
    broken bonds: 0 in the first half, 0 in the second

  - **broken bonds is the ground truth.** A joint can sit above its elastic
    limit forever without accumulating damage if it only does so in peaks.
    Petronas rides at 1.4x with 2-3 joints past yield and breaks NOTHING in
    45 s. Do not "fix" a building that is not breaking.
  - **peak trend beats peak.** Falling means it is resolving itself; rising
    means it is not. A single peak number is nearly meaningless on a structure
    in motion -- two consecutive runs of the same pack once read 1.23 and 2.15.
  - **persistence beats magnitude.** The list is ranked by the share of the run
    a joint spent overloaded. A joint hot in 8% of the run is a settling
    transient and is not worth touching; the garage's scariest number (2.6x)
    is one of those.
  - **the class line points at the fix.** Joint classes ranked by time
    overloaded is the most actionable line on the card, because it names a
    detail rather than a bond.

**Time window matters.** 432 Park breaks 5 bonds in the first 45 s and 14,561
in the next 45. Anything that looks marginal deserves `AUDIT_SECS=90`.

## What is already settled -- do not redo these

  - **The solve is fixed.** Compliance weighting (minimum energy, not minimum
    norm) landed in both backends. See `bond-compliance.md`. Load now
    distributes by stiffness, which is why hot joints finally land in
    physically real places.
  - **Iterations are 32**, which is where the answer converges. Not a dial.
  - **Bending gain stays 3.** Re-derived at 32 iterations; higher values
    destabilise the one building that is sound on its own terms.
  - **The sliver cull stays at 2%.** Tested again AFTER compliance weighting,
    which was predicted to make it redundant. It does not: without it 432 Park
    fails later rather than not at all (4,223 bonds by 120 s) and Algedra gets
    worse (1,000 -> 1,420). Tightening it to 5% is also worse.
  - **Coarser fracture is not the answer.** Raising concrete cell area 2.5x
    moved a tower from 2.48 to 2.36 peak and broke the monolith gate.

## Fixes that have worked, and why

Every one was a real construction detail, not a number:

  - **Strip footings** under a perimeter pier line, replacing isolated pads.
    The podium plate had been spanning pad to pad and reading 2.5x mid-bay.
  - **Structural spandrels** -- a reinforced band directly under each plate,
    butted to the piers. Emit them for EVERY storey; the first version lived
    inside the facade-infill loop, which mechanical floors skip, and those
    floors stayed the hottest thing in the tower.
  - **A real core.** 432 Park's shaft was built around the staircase footprint
    (a 4 m box in the corner of a 9.4 m core), so plates spanned 28 m over a
    stub. Bending goes as span squared: giving them the full core box cut the
    stress ninefold and second-half breakage from 14,561 to 3,140.
  - **Timber wall plates** under masonry-borne roofs. A bond takes the WEAKER
    of its two materials, so roof-on-stone resolved to white-stone at 0.8 MPa
    tension -- and bonds resist moment, so a sagging roof bearing put that
    joint in tension it could not carry. Landing the roof on timber makes the
    joint wood-to-wood at 14.4 MPa. Joints past yield 49 -> 6.8.

**The material-of-the-weaker-member rule is worth internalising.** Any joint
between a strong member and a weak one is governed by the weak one, so a
detail that interposes the right material is often worth more than making
anything stronger.

## The open shared problem: slab-to-slab

Both buildings that still fail are dominated by `slab<->slab` -- the floor
plate's own fracture seams carrying its span moment. 432 Park: 3,220
bond-samples over limit, four times the next class. Algedra: 865.

A real reinforced plate carries that with steel crossing every seam, or with
post-tensioning; this model has neither, so the seams carry it alone. The
levers that follow are geometric:

  - **shorten the span** -- beams under the plate, a bigger core, a tighter
    column grid. Moment goes as span squared, so this is the strongest lever
    available and worth reaching for first.
  - **deepen the plate** -- section modulus goes as depth squared, but the
    extra depth is also extra dead load, so the net is much weaker than it
    looks. Algedra 0.35 -> 0.45 m bought 1,311 -> 1,000 broken bonds.

A caution learned the expensive way: a beam grid in BOTH directions puts
1,434 pairs of beams inside each other where they cross, and beam lines that
do not touch the core have nothing to sit on in the middle and span 26 m. If
you add beams, either make them one-way, or make one direction continuous and
segment the other between it -- and check every line has something to bear on
at both ends.

## Rules

  - Change geometry and structure, not the material table or the global
    solver knobs. Those are shared by every building and are calibrated.
  - Keep the build gates green. A pack that fails interpenetration or
    grounding is not a fix.
  - Report what you MEASURED, including things that did not work. A negative
    result stops the next person repeating it, and several of the entries
    above are negative results.

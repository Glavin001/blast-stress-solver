# Designing the parking garage, at this world's gravity

Worked through against the measurements, including the parts that failed.

## What should fail first

**Punching shear at the slab-column connection.** Not the overhang tearing and
not the column crushing.

1. The surviving columns inherit the tributary area of the ones removed.
2. The slab fails AROUND the remaining column head -- it punches through, like
   a cookie cutter. Brittle, almost no warning, and a flat plate has no beams
   to redistribute through.
3. That column's load moves to its neighbours, which punch in turn: the
   pancake, laterally across the floor.
4. Slab flexural failure follows. Column crushing or buckling is usually LAST
   -- columns are the strongest element, heavily reinforced, working in
   compression where concrete is good.

Pipers Row car park, Wolverhampton, 1997, is the reference case.

A stair core is a reinforced shear-wall box and genuinely survives. It carries
only the floor that can span to it, so the correct behaviour is the slab
tearing away at the core's edge, not the deck staying attached to it.

## The checks, at 20 m/s^2 rather than 9.81

    rho = 2400 kg/m^3, t = 0.30 m, bay 7.6 x 8.0 m, column 0.55 m, f'c = 35 MPa

    self weight        w = rho * t * g = 14.4 kPa
    shear at a column  V = w * bayX * bayZ = 876 kN per level
    punching perimeter u = 4*(0.55 + 0.27) = 3.28 m, d = 0.27 m
    demand             V / (u*d) = 0.99 MPa
    capacity           0.35*sqrt(f'c) = 2.07 MPa
    utilisation        0.48 intact

Flexure, per unit width of a plate spanning L:

    sigma = (w L^2 / 8) / (t^2 / 6) = 0.75 * rho * g * L^2 / t = 7.68 MPa

against plain concrete's ~3.8 MPa. **The deck cracks under its own weight at
this gravity**, which is the root of everything below.

## Parameterising by gravity

Solving the flexural check for the two things a designer can change:

    thickness   t = 0.75 * rho * g * L^2 / sigma
    span        L = sqrt(sigma * t / (0.75 * rho * g))

Both return the AUTHORED values at Earth gravity -- 0.30 m and 8.0 m -- so the
garage was correctly designed for 9.81 and simply never recorded which gravity
it assumed. At 20 m/s^2 they give 0.65 m or 5.6 m respectively.

**Thickening is the wrong lever, and this was measured rather than reasoned.**
A plate is its own load, so `t` appears on both sides: going to 0.65 m
multiplied the deck's weight by 2.2, and every column and punching connection
underneath carried it. Intact breakage went from 344 bonds to 998. Flexure was
satisfied; the two checks coupled to it were not.

**Shortening the span is the right lever and still not sufficient.** A 5.6 m
grid puts a drop panel through the ramp bay (14 shared-volume pairs) and left
the intact garage worse again. What a real engineer reaches for at this load is
a beam-and-slab deck or post-tensioning, neither of which this model expresses.

## What is in the shipped garage

**Drop panels** at every column head: a 2.4 m pad, 200 mm below the soffit,
with the column stopping at the pad rather than running to the deck. This is
the detail a real flat plate uses against punching, and it is what lets the
garage stand at 20 m/s^2 -- it passes its stability gate, where before it
settled at 16 s while shedding 146 bonds.

## What is NOT in it, and why

The material limits, which are the reason nothing punches:

    reinforced concrete     ours        real
    compression           48.0 MPa    30-50      in range
    tension               14.4 MPa    3-4        assumes rebar across seams
    shear                 19.2 MPa    1-2        (4 with stirrups)

A fracture seam has no steel crossing it, so those tension and shear values
describe a continuity that is not there. Corrected -- tension 3.8 MPa, shear
6.1 MPa, the latter derived through this model's bond geometry (the bond is the
0.30 m^2 column face against a 0.89 m^2 real punching perimeter, concentrating
2.9x) -- the garage finally fails the way it should:

    columns cut    bonds broken    deck
    25%                 40         standing
    50%                 74         standing
    75%                154         standing
    90%              2,898         COLLAPSING, 2.31 m

against 0-5 bonds and nothing until 100% before. Progressive damage that
accumulates and then goes.

It is not applied because the intact garage cannot then stand: the deck cracks
along its own hogging lines at 344 bonds, which is the flexural check above
failing honestly. Applying it needs the deck redesigned for this gravity at the
same time -- beams, or a grid the ramp can live with -- and the other six
buildings recalibrated with it.

## The model limit worth fixing first

Capacity here is bond AREA. A 2.4 m drop panel therefore earns 19x the bare
column's capacity, where the real detail earns about 3x, because real punching
capacity grows with the PERIMETER at the panel edge and not with the pad's
area. That over-credit is why sizing the panel produced non-monotonic results
across 1.0, 1.6 and 2.4 m, and it is a solver change rather than an authoring
one.

## Third lever tried: a beam-and-slab deck. Also not sufficient.

Beams are the textbook answer at this load, and they are the only lever that
leaves the column grid where parking needs it -- 7.5 m minimum for an aisle --
while shortening what the SLAB has to span. Parameterised the same way:

    intermediate beams per bay = ceil(bay / maxSlabSpan) - 1

which returns **0 at Earth gravity** (a flat plate, the original design) and
**1 at 20 m/s^2**, putting the slab on a 4.0 m span where it develops 1.9 MPa
against concrete's 3.8. Two details fell out of it that are worth keeping:

  - drop panels belong ONLY to the flat-plate design. Once a deck has beams
    they frame into the column head and deliver load in bearing, so there is no
    punching perimeter left to thicken -- and a pad there occupies the same
    space as the beam crossing it (745 shared-volume pairs).
  - the intermediate beams must be segmented at each primary, or four beams
    meet at a point and put themselves inside each other. Same crossing
    mistake as 432 Park's first beam grid, 1,434 pairs.

It still does not stand: 4,693 bonds, and slab-to-slab is STILL the dominant
class at 3,400. So the deck seams crack whether the slab spans 8.0 m or 4.0 m,
which means the flexural span is not what is driving them -- something else in
the deck is generating that tension, and finding it is where the next attempt
should start rather than reaching for a fourth lever.

## Where this leaves the decision

Three levers tried against realistic concrete, all measured, none sufficient:

    thicker plate    344 -> 998 bonds     worse; the plate is its own load
    tighter grid     drop panel through the ramp bay; garage unparkable
    beam-and-slab    4,693 bonds; slab-to-slab still dominant

The shipped garage keeps drop panels and the current material table, and passes
its gate. The realistic table is a one-line change away and makes the garage
fail correctly (40 / 74 / 154 bonds at 25 / 50 / 75% of columns cut, then
collapse at 90%) at the cost of not standing intact.

The alternative worth weighing: every building in this set is correctly
designed for 9.81, and 20 m/s^2 was chosen for how the PLAYER moves, not for
the structures. VIBE_WORLD_GRAVITY=9.81 makes the whole problem disappear
without a line of structural work.

## The number that resolves it

Realistic concrete at the DESIGN gravity, 9.81, still breaks 194 bonds. So it
is not simply that this world is heavy. The flexural check says why:

    g = 9.81, L = 8.0 m, t = 0.30 m   ->   sigma = 3.77 MPa
    plain concrete in tension                     3.80 MPa

The deck is designed to 99% of its limit, with no safety factor at all. Any
dynamic transient takes it over, which is exactly what 194 bonds of settling
damage is. Real design carries 1.5 or more.

Put the safety factor in and solve for the bay:

    g       safety   bay
    9.81      1.0    8.0 m      <- what is authored
    9.81      1.5    6.6 m
    20        1.0    5.6 m
    20        1.5    4.6 m

**A parking bay needs 7.5 m** for an aisle plus spaces. So with plain concrete
there is no bay that is both parkable and safe, at either gravity. That is not
a bug in the garage; it is why real garages are not built from plain concrete.
They are reinforced, and the rebar carries the tension across every crack.

Which is the actual finding. This model has no reinforcement: a fracture seam
is plain concrete, so either it gets concrete's 3.8 MPa and no realistic garage
can span a parking bay, or it keeps 14.4 MPa -- rebar's continuity, smeared
into the concrete -- and nothing ever cracks or punches. Both are wrong in
opposite directions, and no material number fixes it because the missing thing
is not a number.

What would: a seam that carries tension past cracking at a REDUCED but non-zero
strength, which is what a reinforced crack does -- concrete lets go at 3.8 MPa
and the steel across it keeps carrying to yield. That is the elastic-to-fatal
band this solver already has, used properly: elastic at concrete's cracking
stress, fatal at the reinforcement's capacity, with the band between them wide
rather than the 3x it is now. It is a materials change, it is expressible
today, and it is the first thing to try next.

## And the wide band does not do it either -- which locates the real gap

Tried it: tension cracking at 3.8 MPa with ultimate at 30.7 (band 8), so the
seam lets go early and keeps carrying to a rebar-like capacity. Still 222 bonds
in an intact garage, against 194 with a narrow band.

Widening the band only slows damage; it does not stop it. Anything above the
elastic limit accrues damage for as long as it stays there, so a deck that
sits at 1.1x cracking forever eventually sheds its seams whatever the fatal
limit is.

**That is the gap, and it is in the damage model rather than in any material.**
A reinforced crack does not behave that way. It opens to a width the steel can
hold and then STOPS -- it is a stable state, not a stage of failure. Concrete
past its tensile strength with reinforcement across it reaches equilibrium and
stays there for decades.

So what is needed is damage that ARRESTS: above the elastic limit a joint loses
section until the reduced section carries the load, and then stops, rather than
continuing to accrue while any overload remains. The runaway path stays for
joints that genuinely have no reserve -- plain masonry, glass, the seam between
two unreinforced blocks -- which is where it belongs.

Concretely: `generateStressDamage` currently takes health down by
health * multiplier * dt * rate for as long as stress exceeds elastic. It wants
a floor, per material, below which damage stops -- the residual capacity the
reinforcement represents. A joint would then crack, weaken to that floor, and
hold, which is both what reinforced concrete does and what would let this
garage stand while still punching when a column goes.

That is one function in NvBlastExtStressSolver.cpp, one new material field, and
it is the thing to do before touching the garage again. Everything above --
three geometric levers and two material tables -- was an attempt to work around
its absence.

## The last thing tried, and the constraint that stopped it

Deck joints in a stripped garage sit at 2.9x their elastic limit. Elastic is
3.8 MPa, so that is 11.0 MPa -- and the fatal limit is 3 x 3.8 = 11.4. They are
held up by four tenths of a megapascal, and by a fatal limit that is not
physical: the band from cracking to ultimate is what the STEEL adds, and 1%
reinforcement at 500 MPa is 5 MPa on the gross section. That is a band of about
1.35, not 3, and a band of 3 implies an ultimate tension no rebar ratio
delivers.

Setting it to 1.35 is rejected by the build's own gate, correctly:

    FAIL  cladding is only 4.4x weaker than the frame -- a hit that takes
          panels off will take structure with it

The tier gap between frame and cladding must stay at 8x or more, and lowering
concrete's ultimate closes it from the top. So the band cannot move on its own;
facade-panel and its clips have to come down with it, re-derived rather than
scaled, and the four authored-structure gates re-run against the result.

That is the next piece of work, and it is bounded: derive cladding's cracking
and ultimate the same way concrete's were derived here, check the tier gap
holds, and the deck joints that are currently surviving on 0.4 MPa of
unphysical margin will let go.

## Summary of where the garage stands

Working, committed, verified:

  - stands on its own at 9.81 with realistic concrete, 3.8 MPa tension and
    6.1 MPa shear
  - carries 1200 Pa of live load, about half the code minimum, which is what
    this slab-to-column connection takes
  - real dimensions, verified by the flexural check rather than asserted:
    7.6 x 8.0 m bays, 3.1 m storeys, 0.30 m slab, 0.55 m columns
  - drop panels and mushroom capitals at every column head
  - parameterised by gravity: maxSlabSpan, bayForGravity, liveLoadDensityScale
  - convex fractures, 3,965 hulls against 145 cuboids

Not working:

  - it still does not collapse when the columns are stripped. At 90% removed:
    0.00 m sag, 0 broken bonds.

Two things stand between here and that, both identified and neither guessed:
punching capacity should follow the PERIMETER rather than the bearing area, and
the elastic-to-fatal band should come from the reinforcement rather than a flat
3. The second is blocked on re-deriving cladding to keep the tier gap.

## The band was tried properly, with cladding re-derived, and it regressed

The tier-gap objection above is arithmetic, so it was satisfied rather than
worked around: reinforced concrete's band to 1.35 (the reinforcement figure),
and facade-panel from 12 to 6.0 MPa so the frame-to-cladding ratio comes back to
9x. Weaker cladding is also the more honest number -- these are non-structural
panels whose job is to come off, and 12 MPa was closer to a structural precast
unit.

All 27 packs build. The garage still stands. And the building tier went from
6 of 7 passing to 3 of 7, so it was reverted.

Two things worth keeping from it:

  - The tier gate is not the obstacle. It can be satisfied by moving both sides,
    and doing so is defensible on its own terms.
  - Something in the utilisation path does not match the model of it that this
    document has been reasoning with. At band 1.35 a joint reported at 2.93x
    its elastic limit is far past fatal -- stressMultiplier would be 5.5 -- and
    should break in one tick. It does not. Either the reported utilisation is
    not stress/elastic, or the damage path is not reading the band the material
    table thinks it set.

**That question should be answered before any further material tuning.** Six
levers have now been tried against this garage and the last three all failed in
ways that only make sense if the number being read is not the number being set.
Chasing a seventh without resolving that is how the next day gets spent.

The cheap way to settle it: take one bond in a stripped garage, print its
stress, its material's elastic and fatal limits, and the stressMultiplier the
damage path computes for it, and check they agree with each other.

### Narrowed: where to look for the missing damage

Traced the two paths that should agree.

REPORTED UTILISATION (`getBondUtilisations`, solver ~4221) divides each fibre
stress by that bond's OWN material elastic limit and takes the max. So a
reported 2.93 means stress is 2.93x elastic -- not some other reference.

DAMAGE (`generateStressDamage`, ~4939) computes
`stressMultiplier = (stress - elastic) / (fatal - elastic)` and breaks the bond
outright at >= 1. At band 1.35 a joint at 2.93x elastic gives 5.5, which is an
immediate break. It does not break.

Both read the same stresses -- the overstress test at ~2466 uses
`calcSolverBondStresses` + `fibreStresses`, exactly what the report uses -- so
the formulas are not the discrepancy. What sits between them is the GATE:

    const bool anyBondWork = graphNodeCount > 1
                          && m_graphProcessor->getOverstressedBondCount() > 0;
    ...
    if (anyBondWork && (!nodeSkip || m_graphProcessor->isNodeOverstressed(node0)))

Damage only runs for nodes that path marks overstressed, and those marks come
from the strip walk, which is itself skippable: a group that is "not dirty and
not overstressed" is skipped, and `m_overstressedBondCount` is rebuilt from
what the walk visits (~2341), with a cached value restored at ~2745 and ~2832.

**So the first thing to check is whether a bond can be overstressed while the
group holding it is marked clean.** That is exactly the shape of the two bugs
already found in this area -- the settled-island skip freezing stresses at tick
2, and the sampler running every 30 ticks -- and it would produce precisely
this symptom: a joint the report can see at 2.93x that the damage path never
visits.

Concretely, in a stripped garage: pick a bond the report shows above 1.0, and
check `isNodeOverstressed` for both its nodes and whether its group was skipped
that tick. If the group is clean while the bond is hot, that is the bug.

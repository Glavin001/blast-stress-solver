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

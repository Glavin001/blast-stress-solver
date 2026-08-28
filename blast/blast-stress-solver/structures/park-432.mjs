/**
 * 432 Park Avenue — Rafael Viñoly, New York, 2015.
 *
 * A square concrete tube with a completely uniform grid of square windows, and
 * nothing else. That plainness is why it earns a place here: the geometry is
 * almost free to generate, and what it buys is the one thing the rest of the
 * set lacks — SLENDERNESS. Everything else here is wider than it is tall and
 * pancakes. This topples.
 *
 * The open mechanical floors are modelled because they are real and they are
 * structural: every twelfth floor is left open on all four sides so wind passes
 * through instead of pushing. In a destructible building they are also the
 * obvious thing to aim at, being the only levels with no facade to chew
 * through.
 *
 * Not to scale in height. The real tower is 426 m on a 28.5 m square — 1:15 —
 * which at this fracture density would be forty thousand chunks. Modelled at
 * 1:5.5, which still topples rather than pancaking.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { staircase, stairShaft, slabWithOpening, shaftVoid, unionStairFootprints, DRY_RUN } from './lib/elements.mjs';

export const PARK_432 = {
  side: 28.0,
  floors: 32,
  floorHeight: 3.9,
  slabThickness: 0.34,
  baysPerSide: 6,          // windows per facade per floor
  pierWidth: 1.35,         // the deep white piers between windows
  spandrelDepth: 1.1,      // the band between one floor's windows and the next
  // Structural spandrel: the reinforced band directly UNDER each plate,
  // spanning pier to pier. This is what makes the facade a TUBE rather than a
  // row of columns -- and without it the stress card said what the plate edge
  // thought of spanning 4.67 m bays alone: slab seams at 2.5x mid-bay, every
  // storey. The real tower's grid is exactly this: pier, spandrel, window.
  spandrelBeamDepth: 0.8,
  glassThickness: 0.07,
  columnSize: 1.1,         // perimeter columns are the piers themselves
  coreSide: 9.4,
  // A tower's core walls are metres thick, not centimetres. At 0.35 m this
  // core carried 85% of the building through 1.1 m^2 at three times yield;
  // the core of a real tube-and-core tower is sized for exactly that load.
  coreThickness: 0.9,
  mechanicalEvery: 12,     // floors left open on all four sides
  footingDepth: 2.4,
};

export function buildPark432(cfg = {}) {
  const C = { ...PARK_432, ...cfg };
  const b = new ScenePackBuilder({
    key: 'park_432',
    title: '432 Park Avenue — square tube, uniform window grid',
    seed: 0x432a,
  });

  const h = C.side / 2;
  const podiumTop = 0.6;
  const slabBase = (k) => (k === 0 ? 0 : podiumTop + k * C.floorHeight);
  const slabTop = (k) => (k === 0 ? podiumTop : slabBase(k) + C.slabThickness);
  const isMechanical = (k) => k > 0 && k < C.floors && k % C.mechanicalEvery === 0;

  // The facade grid. Piers land on the bay boundaries, so a pier IS a column —
  // which is what a tube structure means: the perimeter carries the building
  // and the core carries the rest.
  const bayW = C.side / C.baysPerSide;
  const cs = C.columnSize / 2;
  // Perimeter piers are pulled inside the building line rather than straddling
  // it. Straddling halves how much of each pier the plate above actually bears
  // on, which pushes the load it does not take onto the core instead.
  const clamp = (v) => Math.max(-h + cs, Math.min(h - cs, v));
  const pierAt = Array.from({ length: C.baysPerSide + 1 }, (_, i) => clamp(-h + i * bayW));

  // ── footings: a continuous strip under the perimeter pier line ───────────
  //
  // These were isolated pads, one under each pier, and the stress card said
  // exactly what that costs: the podium plate's edge strip has to SPAN from
  // pad to pad carrying the pier line and the ground-storey facade, and its
  // bottom face mid-bay read 2.5x its tension limit standing still -- the
  // hottest joints in the whole tower, at (-7, 0, -11) and its mirrors,
  // midway between pads at -4.67 and -9.33. Fifteen of the tower's thirty
  // over-limit joints were this one detail.
  //
  // A strip footing is how real perimeter foundations solve it: continuous
  // bearing under the line, so the plate edge stops being a beam. Four
  // strips, the two in Z shortened so the corners are not doubly owned.
  {
    const ring = h - cs;                 // the pier line the strips run under
    const sw = cs + 0.25;                // half-width: pier face plus a margin
    for (const sign of [-1, 1]) {
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [-h, -C.footingDepth, sign * ring - sw],
        max: [h, 0, sign * ring + sw],
        fixed: true, fracture: false });
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [sign * ring - sw, -C.footingDepth, -ring + sw],
        max: [sign * ring + sw, 0, ring - sw],
        fixed: true, fracture: false });
    }
  }
  const ch = C.coreSide / 2;

  // ── the stair core, and the void it needs through every plate ────────────
  // Flush into the core's corner, so the flights bear on the core walls
  // themselves. Held off by 0.3 they hung in mid-air once the shaft grew to
  // the full core (31 overhanging landings); wrapped in their own enclosure
  // instead, that enclosure's walls shared volume with the core's (595 pairs).
  // Sitting them against the wall is both the simplest and what a stair in a
  // core corner actually does.
  const stairAt = [-ch, -ch];
  // Per storey: the first flight climbs from the podium and is taller than the
  // rest, so a shaft sized from a typical floor is too short for it.
  const dryFor = (k) => staircase(DRY_RUN,
    { at: stairAt, y0: slabTop(k - 1), y1: slabTop(k), axis: 'x' });
  const dry = unionStairFootprints(
    Array.from({ length: C.floors }, (_, i) => dryFor(i + 1)),
  );
  const CORE_T = C.coreThickness;
  const void_ = dry.void;

  // The core the plates actually need.
  //
  // coreSide has always been 9.4 m, but the shaft was built around the
  // STAIRCASE footprint -- a ~4 m box parked at (-4.4, -4.4), because stairAt
  // puts the flight in the core's corner. So every floor plate was a 28 m
  // square spanning to its perimeter piers over one small off-centre stub.
  //
  // Bending in a plate goes as span squared, and the arithmetic lands exactly
  // where the audit did: w = 8.2 kN/m^2 on a 28 m span is M = wL^2/8 = 807
  // kN.m/m against a section modulus of 0.0193 m^3, so 42 MPa on a 14.4 MPa
  // tension limit -- 2.8x, and slab-to-slab seams were the hottest joint class
  // in the tower by a factor of two over anything else. Over ninety seconds
  // that is not a rounding error: 5 bonds broke in the first half and 14,561
  // in the second.
  //
  // A real tower of this proportion has a real central core, and 432 Park's is
  // most of its middle. Giving the plates the full 9.4 m box to bear on cuts
  // the clear span to 9.3 m, and span squared does the rest: 4.6 MPa, 0.32 of
  // the limit. The staircase still lives inside it.
  const core = { x0: -ch, x1: ch, z0: -ch, z1: ch };

  // The core lands on its OWN raft, not on the podium plate. It carries most of
  // the building, and routing that through a floor-plate material on the way to
  // the ground is the same mistake the tower's stilts made.
  //
  // The raft covers the WHOLE shaft, walls included — not just the stair void
  // the upper floors leave open. Sized to the void it caught only part of each
  // core wall and the rest came down on the podium plate, which then had to
  // pass 16,500 t into the raft through 1.5 m^2 of its own edge. There is no
  // stair arriving from below at the podium, so no arrival strip is needed
  // here and the raft can fill the shaft outright.
  const shaftOuter = {
    x0: core.x0 - CORE_T, x1: core.x1 + CORE_T,
    z0: core.z0 - CORE_T, z1: core.z1 + CORE_T,
  };
  b.box({ type: 'foundation', material: 'footing-anchor',
    min: [shaftOuter.x0, -C.footingDepth, shaftOuter.z0],
    max: [shaftOuter.x1, podiumTop, shaftOuter.z1],
    fixed: true, fracture: false });
  slabWithOpening(b, {
    material: 'reinforced-concrete',
    min: [-h, 0, -h], max: [h, podiumTop, h], opening: shaftOuter,
  });

  // ── floors ───────────────────────────────────────────────────────────────
  for (let k = 1; k <= C.floors; k += 1) {
    // Plate. Reinforced throughout: the perimeter piers land on it, and a floor
    // plate's strength is not what carries a column.
    slabWithOpening(b, {
      material: 'reinforced-concrete',
      min: [-h, slabBase(k), -h], max: [h, slabTop(k), h], opening: void_,
    });

    // Core walls, which brace the tube. Open on one side to walk in.
    // One shaft size for every storey, taken from the tallest flight, so the
    // walls line up floor to floor and the plate opening matches all of them.
    // Spans THROUGH the plate above, not up to its underside. A shaft that
    // stops at slab level sits entirely inside the floor void and touches
    // nothing at all — 1,364 stair pieces and the whole core came back as
    // floating. Running it through puts the slab edge against the wall face,
    // which is the joint a core actually has.
      // Stops UNDER the plate, which then passes over it. The alternative —
      // running the wall through the plate — needs the plate cut away wherever
      // the wall is, and that cut-out cannot coexist with the arrival strip:
      // the side walls run the full length of the shaft, including alongside
      // the strip. Stopping the wall here gives the core a far better joint
      // anyway: its whole top face bears on the slab, rather than the slab's
      // edge meeting its side.
    stairShaft(b, {
      footprint: core, y0: slabTop(k - 1), y1: slabBase(k),
      material: 'reinforced-concrete', openSide: '-x', thickness: CORE_T,
    });
    staircase(b, {
      at: stairAt, y0: slabTop(k - 1), y1: slabTop(k), axis: 'x',
      material: 'reinforced-concrete', newelPost: false,
    });

    // Perimeter piers, from the plate BELOW this one up to it — the same storey
    // the stair and shaft span. Indexed the other way round they start at floor
    // one and the podium carries nothing, which floats the entire tower.
    for (const x of pierAt) {
      for (const z of pierAt) {
        if (Math.abs(x) < h - 1e-6 && Math.abs(z) < h - 1e-6) continue;
        b.box({ type: 'column', material: 'reinforced-concrete',
          min: [x - cs, slabTop(k - 1), z - cs], max: [x + cs, slabBase(k), z + cs] });
      }
    }
    // The structural spandrel ring: reinforced, full pier depth, butted
    // against the piers either side, carrying each plate edge continuously.
    // Emitted for EVERY storey, before the mechanical-floor skip below: the
    // skip is about infill, and when this beam lived inside the infill loop
    // the mechanical floors kept spanning bare and stayed the hottest joints
    // in the tower.
    {
      const bTop = slabBase(k);
      const bBot = bTop - C.spandrelBeamDepth;
      for (const [axis, sign] of [['z', 1], ['z', -1], ['x', 1], ['x', -1]]) {
        for (let i = 0; i < C.baysPerSide; i += 1) {
          const limit = h - C.columnSize;
          const a0 = Math.max(pierAt[i] + cs, -limit);
          const a1 = Math.min(pierAt[i + 1] - cs, limit);
          if (!(a1 - a0 > 0.2)) continue;
          const sLo = Math.min(sign * (h - C.columnSize), sign * h);
          const sHi = Math.max(sign * (h - C.columnSize), sign * h);
          b.piece({
            type: 'beam', material: 'reinforced-concrete', axis,
            lo: sLo, hi: sHi,
            poly: axis === 'z'
              ? [[a0, bBot], [a1, bBot], [a1, bTop], [a0, bTop]]
              : [[bBot, a0], [bBot, a1], [bTop, a1], [bTop, a0]],
          });
        }
      }
    }

    if (isMechanical(k)) continue;   // open floor: piers only, no infill

    // Facade infill for THIS storey — the same band the piers of storey k
    // occupy, below plate k. Indexed to the storey above instead, the top floor
    // grew a facade over the roof and drove it through the parapet.
    const y0 = slabTop(k - 1), y1 = slabBase(k);
    const sill = y0 + C.spandrelDepth;
    // The pane stops short of the plate above. Fill the opening and the floor
    // rests ON the glass, and the static walk duly routes a storey's weight
    // through a glazing clip. Real curtain wall hangs off one slab and carries
    // only itself.
    const head = y1 - C.spandrelBeamDepth - 0.02;
    for (const [axis, sign] of [['z', 1], ['z', -1], ['x', 1], ['x', -1]]) {
      const at = sign * (h - C.glassThickness);
      for (let i = 0; i < C.baysPerSide; i += 1) {
        // Clamped short of where the PERPENDICULAR facade's plane band starts.
        // Without that the +Z spandrel runs past x = h - columnSize and straight
        // through the +X spandrel at every corner of every floor.
        const limit = h - C.columnSize;
        const a0 = Math.max(pierAt[i] + cs, -limit);
        const a1 = Math.min(pierAt[i + 1] - cs, limit);
        if (!(a1 - a0 > 0.2)) continue;
        // Spandrel, in the same plane as the piers so the facade reads flat.
        // min/max rather than the raw signed pair: on the -X and -Z faces the
        // signed values arrive in descending order and an extent must not.
        const sLo = Math.min(sign * (h - C.columnSize), sign * h);
        const sHi = Math.max(sign * (h - C.columnSize), sign * h);
        // Infill between the piers — cladding. The piers themselves are the
        // tube and stay reinforced.
        b.piece({
          type: 'wall', material: 'facade-panel', axis,
          lo: sLo, hi: sHi,
          poly: axis === 'z' ? [[a0, y0], [a1, y0], [a1, sill], [a0, sill]]
                             : [[y0, a0], [y0, a1], [sill, a1], [sill, a0]],
        });
        b.piece({
          type: 'glazing', material: 'glass', axis,
          lo: at - C.glassThickness / 2, hi: at + C.glassThickness / 2,
          poly: axis === 'z' ? [[a0, sill], [a1, sill], [a1, head], [a0, head]]
                             : [[sill, a0], [sill, a1], [head, a1], [head, a0]],
        });
      }
    }
  }

  // ── parapet ──────────────────────────────────────────────────────────────
  // The topmost floor plate IS the roof; the loop above already built it. An
  // extra plate above it had no columns underneath and came back floating.
  const pt = slabTop(C.floors);
  for (const [min, max] of [
    [[-h, pt, -h], [h, pt + 1.4, -h + 0.4]], [[-h, pt, h - 0.4], [h, pt + 1.4, h]],
    [[-h, pt, -h + 0.4], [-h + 0.4, pt + 1.4, h - 0.4]], [[h - 0.4, pt, -h + 0.4], [h, pt + 1.4, h - 0.4]],
  ]) b.box({ type: 'parapet', material: 'facade-panel', min, max });

  b.build();
  return { pack: b.emit({ cameraTarget: [0, pt * 0.45, 0], cameraDistance: 240 }), builder: b };
}

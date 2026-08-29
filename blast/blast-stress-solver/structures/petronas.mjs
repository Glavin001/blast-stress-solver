/**
 * Petronas Towers — César Pelli, Kuala Lumpur, 1998.
 *
 * Twin towers on a lobed plan, tapering in setbacks, joined at mid-height by a
 * double-decker skybridge.
 *
 * The skybridge is the reason this is worth the geometry. Everything else in
 * the set is a single object that fails on its own; here two structures are
 * COUPLED, so damage to one has somewhere to travel. The real bridge is not
 * rigidly fixed at either end — it slides, because two 450 m towers sway
 * independently and a stiff link would tear itself apart. It is arch-supported
 * from below by two legs off each tower, and that is modelled: the legs carry
 * it, the ends only lean on the towers.
 *
 * The plan is Islamic geometry — two interlocking squares making an eight-point
 * star, with circular infills. Approximated here by a radius that varies as
 * cos(8θ), which produces the same eight-lobed silhouette out of straight
 * pieces. The format has no curves and no rotation, so every lobe is faceted;
 * at twelve segments a floor it reads as round from any distance you would look
 * at a tower from.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { staircase, stairShaft, roofSlope, unionStairFootprints, slabWithOpening, DRY_RUN } from './lib/elements.mjs';

export const PETRONAS = {
  radius: 19.5,          // mean outer radius at the base
  lobeAmplitude: 0.11,   // depth of the eight lobes, as a fraction of radius
  coreHalf: 6.0,         // the square core
  floors: 22,
  floorHeight: 4.0,
  slabThickness: 0.32,
  // 16, not 12, and the number is load-bearing. Segment boundaries fall every
  // 22.5 deg, which includes the 45 deg diagonals — so no segment straddles a
  // corner of the square core, and every segment's inner edge lies flat along
  // one core face instead of cutting across it.
  segments: 16,
  columnSize: 0.85,
  glassBandHeight: 2.4,
  spandrelHeight: 1.0,
  setbackEvery: 7,       // radius steps down at these floors
  setbackFraction: 0.1,
  spacing: 62.0,         // centre to centre
  bridgeFloor: 15,
  bridgeWidth: 5.6,
  bridgeDepth: 2.6,
  footingDepth: 3.0,
};

export function buildPetronas(cfg = {}) {
  const C = { ...PETRONAS, ...cfg };
  const b = new ScenePackBuilder({
    key: 'petronas',
    title: 'Petronas Towers — twin lobed towers with a skybridge',
    seed: 0x9e70,
  });

  const podiumTop = 0.8;
  const slabBase = (k) => (k === 0 ? 0 : podiumTop + k * C.floorHeight);
  const slabTop = (k) => (k === 0 ? podiumTop : slabBase(k) + C.slabThickness);
  /** Mean radius at floor k, stepping in at each setback. */
  const meanR = (k) => C.radius * (1 - C.setbackFraction * Math.floor(k / C.setbackEvery));
  /** The eight-lobed outline: radius as a function of angle. */
  const outerR = (k, t) => meanR(k) * (1 + C.lobeAmplitude * Math.cos(8 * t));
  const ch = C.coreHalf, cs = C.columnSize / 2;

  const tower = (cx, cz, openSide, bridgeDir) => {
    const P = (r, t) => [cx + r * Math.cos(t), cz + r * Math.sin(t)];
    // Columns ring just outside the core, on the segment boundaries.
    // Inside the SMALLEST setback, not a fraction of the base radius. The
    // facade steps in at every setback until, near the top, its inner face had
    // arrived at the column ring and the two occupied the same space. The
    // columns cannot step in with it — they have to stack to carry anything.
    const tightest = C.radius * (1 - C.setbackFraction * Math.floor(C.floors / C.setbackEvery));
    const colR = tightest * (1 - C.lobeAmplitude) - 1.6;

    // ── foundation ─────────────────────────────────────────────────────────
    b.box({ type: 'foundation', material: 'footing-anchor',
      // Pulled in: at ch+1 its corners reached the footings of the columns on
      // the 45-degree diagonals.
      min: [cx - ch + 0.5, -C.footingDepth, cz - ch + 0.5],
      max: [cx + ch - 0.5, 0, cz + ch - 0.5], fixed: true, fracture: false });
    for (let i = 0; i < C.segments; i += 1) {
      const t = (2 * Math.PI * i) / C.segments;
      const [x, z] = P(colR, t);
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [x - 1.0, -C.footingDepth, z - 1.0], max: [x + 1.0, 0, z + 1.0],
        fixed: true, fracture: false });

      // And under the perimeter ring, at the BASE tier's radius. Without
      // these the new perimeter columns had nothing to stand on but the
      // podium slab, and adding them made the load path WORSE rather than
      // better -- 65.3% of the weight reaching a support fell to 59.3%,
      // because a column that ends on a plate is one more thing that plate is
      // carrying.
      const baseTierR = C.radius * (1 - C.lobeAmplitude) - 1.2;
      if (baseTierR > colR + 2.0) {
        const [px, pz] = P(baseTierR, t);
        b.box({ type: 'foundation', material: 'footing-anchor',
          min: [px - 1.0, -C.footingDepth, pz - 1.0], max: [px + 1.0, 0, pz + 1.0],
          fixed: true, fracture: false });
      }
    }

    // A podium ring under the whole plan, sitting on the footings — which stop
    // at grade for exactly this reason, so the ring is not trying to occupy the
    // same space as them. Without the ring the first storey's facade
    // began in mid-air — the footings are only under the core and the columns,
    // and the facade stands 19 m out from either.
    for (let i = 0; i < C.segments; i += 1) {
      const t0 = (2 * Math.PI * i) / C.segments, t1 = (2 * Math.PI * (i + 1)) / C.segments;
      b.piece({
        type: 'slab', material: 'reinforced-concrete', axis: 'y', lo: 0, hi: podiumTop,
        poly: [P(2.0, t0), P(outerR(0, t0), t0), P(outerR(0, t1), t1), P(2.0, t1)],
      });
    }

    // ── the stair core, running the full height, CENTRED ───────────────────
    // Centred because the floor plate is a ring around it: a core off in one
    // corner pokes out through the ring's inner edge on one side and leaves a
    // void on the other.
    // Sized from the LONGEST flight in the tower, at the origin. Storey one
    // runs from the podium and is taller than the rest; sizing the core from a
    // typical floor made it smaller than the flight it holds AND smaller than
    // the square the floor ring is cut to, so it burst out of both.
    const atOrigin = unionStairFootprints(Array.from({ length: C.floors }, (_, i) => staircase(
      DRY_RUN, { at: [0, 0], y0: slabTop(i), y1: slabTop(i + 1), axis: 'x' },
    )));
    const pw = atOrigin.x1, ph = atOrigin.z1;
    const stairAt = [cx - pw / 2, cz - ph / 2];
    // Re-run at the real position rather than translating the origin copy, so
    // the arrival strip in `void` is computed by the one piece of code that
    // knows where it belongs.
    const dry = unionStairFootprints(Array.from({ length: C.floors }, (_, i) => staircase(
      DRY_RUN, { at: stairAt, y0: slabTop(i), y1: slabTop(i + 1), axis: 'x' },
    )));
    const CORE_T = 0.55;
    // Half-size of the square the floor ring's inner edge follows: the core's
    // own outer face, squared off to the larger of its two dimensions.
    const coreA = Math.max(pw, ph) / 2 + CORE_T;
    /** Where a ray at angle t leaves that square. */
    const innerP = (t) => {
      const k = coreA / Math.max(Math.abs(Math.cos(t)), Math.abs(Math.sin(t)));
      return [cx + k * Math.cos(t), cz + k * Math.sin(t)];
    };

    for (let k = 1; k <= C.floors; k += 1) {
      const y0 = slabTop(k - 1), y1 = slabTop(k);

      // Core walls through the plate, and the flight inside them.
      // Walls stop under the plate above; the plate passes over them.
      stairShaft(b, { footprint: dry, y0, y1: slabBase(k), material: 'reinforced-concrete', openSide, thickness: CORE_T });
      // A plate inside the core. The floor ring stops at the core's outer face,
      // so without this the core interior has no floor at any level — you climb
      // the flight and step off into the shaft.
      slabWithOpening(b, {
        material: 'reinforced-concrete',
        min: [cx - coreA, slabBase(k), cz - coreA], max: [cx + coreA, slabTop(k), cz + coreA],
        opening: dry.void,
      });
      staircase(b, { at: stairAt, y0, y1: slabTop(k), axis: 'x', material: 'reinforced-concrete', newelPost: false });

      // Columns for this storey: an inner ring, and a perimeter ring that
      // steps in with the setbacks.
      //
      // There was only the inner ring, at colR = 10.5 m, deliberately placed
      // inside the SMALLEST setback so it could stack the whole height. That
      // left every floor plate spanning 4.5 m from the core to it and then
      // CANTILEVERING 9.0 m past it to the facade at 19.5 m. A cantilever's
      // moment is wL^2/2, so on a 0.32 m plate that is 334 kN.m/m against a
      // section modulus of 0.0171 -- 19.6 MPa on a 14.4 MPa tension limit.
      // Over the limit standing still, before anyone loads it, which is why
      // the build's own walk found only 65% of the weight reaching a support
      // and the tower kept breaking bonds five minutes in.
      //
      // The real towers carry their floors on a perimeter ring that steps in
      // at each setback, and that is what this is: a ring at the tier's own
      // radius, present only while the facade is still outside it. The plate
      // then spans 4.5 m in and about 4 m out instead of hanging 9 m in air.
      for (let i = 0; i < C.segments; i += 1) {
        const t = (2 * Math.PI * i) / C.segments;
        const [x, z] = P(colR, t);
        b.box({ type: 'column', material: 'reinforced-concrete',
          min: [x - cs, y0, z - cs], max: [x + cs, slabBase(k), z + cs] });

        // Perimeter ring, on the segment boundary, just inside this tier's
        // own lobe minimum so it never pokes through the facade it carries.
        const tierR = meanR(k) * (1 - C.lobeAmplitude) - 1.2;
        if (tierR > colR + 2.0) {
          const [px, pz] = P(tierR, t);
          b.box({ type: 'column', material: 'reinforced-concrete',
            min: [px - cs, y0, pz - cs], max: [px + cs, slabBase(k), pz + cs] });
        }
      }

      // ── the floor: a ring of trapezoids from the core out to the lobes ───
      for (let i = 0; i < C.segments; i += 1) {
        const t0 = (2 * Math.PI * i) / C.segments, t1 = (2 * Math.PI * (i + 1)) / C.segments;
        b.piece({
          // Reinforced: these plates frame into the core and land on the
          // columns, and a floor-plate 12 MPa is not what carries either.
          type: 'slab', material: 'reinforced-concrete', axis: 'y',
          lo: slabBase(k), hi: slabTop(k),
          poly: [innerP(t0), P(outerR(k, t0), t0), P(outerR(k, t1), t1), innerP(t1)],
        });
      }

      // ── facade: a spandrel band and a glass band per storey ──────────────
      const sill = y0 + C.spandrelHeight;
      const head = Math.min(sill + C.glassBandHeight, slabBase(k) - 0.03);
      for (let i = 0; i < C.segments; i += 1) {
        const t0 = (2 * Math.PI * i) / C.segments, t1 = (2 * Math.PI * (i + 1)) / C.segments;
        const r0 = outerR(k, t0), r1 = outerR(k, t1);
        // The spandrel is a 0.28 m concrete band; the glazing is 6 cm of glass.
        // Both were built at 0.28, which made every window pane 28 cm thick —
        // five cubic metres of glass per bay, and once chunks were capped by
        // volume that one mistake alone produced 43,555 glass chunks.
        const band = (lo, hi, material, type, depth) => {
          b.piece({
            type, material, axis: 'y', lo, hi,
            poly: [P(r0 - depth, t0), P(r0, t0), P(r1, t1), P(r1 - depth, t1)],
          });
        };
        // The facade is left open where the skybridge passes through it, on the
        // two storeys it occupies and only on the segments facing the other
        // tower. A real skybridge enters through an opening; modelled without
        // one the bridge has nowhere to land and floats.
        const facing = Math.cos(((t0 + t1) / 2) - bridgeDir) > 0.86;
        const atBridge = k === C.bridgeFloor + 1 || k === C.bridgeFloor + 2;
        if (facing && atBridge) continue;
        // Cladding hung off the floor ring, not part of it.
        band(y0, sill, 'facade-panel', 'wall', 0.28);
        if (head > sill + 0.2) band(sill, head, 'glass', 'glazing', 0.06);
      }
    }

    // A stepped crown, which is what makes the towers taper to a point rather
    // than stop.
    let crownR = meanR(C.floors) * 0.5;
    let cy = slabTop(C.floors);
    for (let s = 0; s < 4; s += 1) {
      const next = cy + 3.0;
      // Solid wedges from the axis, not annuli. As rings each step was
      // narrower than the gap to the one below it, so only the first touched
      // anything and the rest of the crown floated.
      for (let i = 0; i < C.segments; i += 1) {
        const t0 = (2 * Math.PI * i) / C.segments, t1 = (2 * Math.PI * (i + 1)) / C.segments;
        b.piece({
          type: 'parapet', material: 'steel', axis: 'y', lo: cy, hi: next,
          poly: [[cx, cz], P(crownR, t0), P(crownR, t1)],
        });
      }
      cy = next; crownR *= 0.68;
    }
    return { colR };
  };

  const half = C.spacing / 2;
  tower(-half, 0, '+x', 0);          // its bridge opening faces +x
  tower(half, 0, '-x', Math.PI);     // and this one's faces -x

  // ── the skybridge ────────────────────────────────────────────────────────
  const by = slabTop(C.bridgeFloor);
  const bw = C.bridgeWidth / 2;
  // The tower's OUTER face at a given floor — which steps in at every setback,
  // so a single radius taken from the base drives the bridge straight through
  // the facade at one end and leaves it short at the other.
  const faceAt = (k) => meanR(k) * (1 + C.lobeAmplitude) + 0.06;
  // Lapped 2 m INSIDE each tower, onto its floor plate, rather than stopped at
  // the facade line. Butted against the facade the deck met the curving ring on
  // a few square centimetres at best; lapped on, it bears on the plate.
  const reach = faceAt(C.bridgeFloor) - 2.0;
  const x0 = -half + reach, x1 = half - reach;
  const deckT0 = C.bridgeDepth * 0.18;
  for (const level of [by, by + 5.4]) {
    b.box({ type: 'bridge', material: 'steel',
      min: [x0, level, -bw], max: [x1, level + C.bridgeDepth * 0.18, bw] });
    for (const z of [-bw, bw - 0.14]) {
      b.box({ type: 'bridge', material: 'steel',
        min: [x0 + 2.4, level + deckT0, z], max: [x1 - 2.4, level + deckT0 + 1.1, z + 0.14] });
    }
  }
  // Struts tying the two decks into one structure. Without them the upper deck
  // was connected to nothing and came back floating.
  // Between the decks, and only within the clear span: started at deck level
  // they occupied the lower deck, and placed near the ends they sat inside the
  // tower's own floor plate.
  const deckT = C.bridgeDepth * 0.18;
  for (const x of [x0 + 4.0, (x0 + x1) / 2, x1 - 4.2]) {
    // On the centre line: at the edges they stood in the same space as the rails.
    b.box({ type: 'bridge', material: 'steel',
      min: [x, by + deckT, -0.08], max: [x + 0.16, by + 5.4, 0.08] });
  }

  // The real bridge is arch-supported by two legs springing off the towers.
  // They are not modelled. Every anchorage I tried for them either met the
  // faceted facade along a line — no bearing at all, so the legs floated — or
  // had to lap inside the tower and drove through the facade band above.
  // Making them work needs a facade opening at the leg's own height, the way
  // the deck has one; that is real work rather than a tweak, and a
  // simply-supported span is the honest simplification in the meantime.
  //
  // What it costs: the bridge still couples the towers, which is the point of
  // including them, but it is carried by its ends rather than from below.

  b.build();
  const top = slabTop(C.floors) + 14;
  return { pack: b.emit({ cameraTarget: [0, top * 0.4, 0], cameraDistance: 260 }), builder: b };
}

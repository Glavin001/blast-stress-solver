/**
 * Composite building elements: things made of several pieces.
 *
 * A wall with a window in it is the motivating case. It cannot be one piece —
 * a wall-with-a-hole is concave, and colliders here are cuboids or convex
 * hulls, never trimeshes — so it has to be authored as the convex rectangles
 * around the opening plus a pane inside it.
 */

/** Plane axes for a wall whose normal is `axis`. */
const PLANE = { x: [1, 2], y: [0, 2], z: [0, 1] };

/**
 * A wall panel with rectangular openings, plus glazing filling them.
 *
 * The wall is decomposed into piers (full-height strips between openings),
 * sills (below an opening) and heads (above it). That decomposition covers the
 * wall exactly with non-overlapping rectangles, which is what keeps the
 * cross-piece interpenetration check clean.
 *
 * @param axis      wall normal: 'x' or 'z'
 * @param at        [lo, hi] wall thickness extent along `axis`
 * @param u0,u1     horizontal extent, in the wall's plane
 * @param v0,v1     vertical extent (world Y)
 * @param openings  [{ u0, u1, v0, v1 }] — window and door holes
 * @param glaze     material for panes, or null to leave openings empty (doors)
 */
export function wallWithOpenings(builder, {
  type = 'wall', material, axis, at, u0, u1, v0, v1, openings = [], glaze = 'glass',
  glazeInset = 0, headGap = 0.015, type_glaze = 'glazing',
}) {
  const [lo, hi] = at;
  const solid = (a0, b0, a1, b1) => {
    if (!(a1 - a0 > 1e-4) || !(b1 - b0 > 1e-4)) return;
    builder.piece({
      type, material, axis, lo, hi,
      poly: axis === 'z'
        ? [[a0, b0], [a1, b0], [a1, b1], [a0, b1]]          // (x, y)
        : [[b0, a0], [b0, a1], [b1, a1], [b1, a0]],          // (y, z) for axis 'x'
    });
  };

  const sorted = [...openings].sort((a, b) => a.u0 - b.u0);
  let cursor = u0;
  for (const o of sorted) {
    solid(cursor, v0, o.u0, v1);                 // pier left of the opening
    solid(o.u0, v0, o.u1, o.v0);                 // sill below it
    solid(o.u0, o.v1, o.u1, v1);                 // head above it
    cursor = o.u1;
  }
  solid(cursor, v0, u1, v1);                     // pier right of the last opening

  if (!glaze) return;
  // The pane fills its opening on three sides and stops `headGap` short of the
  // top. Both details are load-bearing decisions, not cosmetics:
  //
  //   - No inset on the sill and reveals. A pane held off its reveal touches
  //     nothing, bonds to nothing, and falls out of the wall on frame one.
  //     Edge-to-reveal contact is exactly the small, weak, brittle
  //     glazing-clip joint we want.
  //   - Clearance at the head. Without it the masonry above the opening rests
  //     on the glass, and the static walk duly routes the wall's weight through
  //     a 25 cm^2 clip at 79% of yield. Real glazing bears on its sill and a
  //     lintel carries the wall; the gap models that without a lintel piece.
  const mid = (lo + hi) / 2, half = Math.min(0.012, (hi - lo) / 4);
  for (const o of sorted) {
    builder.piece({
      type: type_glaze, material: glaze, axis, lo: mid - half, hi: mid + half,
      poly: axis === 'z'
        ? [[o.u0 + glazeInset, o.v0 + glazeInset], [o.u1 - glazeInset, o.v0 + glazeInset],
           [o.u1 - glazeInset, o.v1 - headGap], [o.u0 + glazeInset, o.v1 - headGap]]
        : [[o.v0 + glazeInset, o.u0 + glazeInset], [o.v0 + glazeInset, o.u1 - glazeInset],
           [o.v1 - headGap, o.u1 - glazeInset], [o.v1 - headGap, o.u0 + glazeInset]],
    });
  }
}

/**
 * One slope of a pitched roof: a parallelogram cross-section extruded along the
 * ridge.
 *
 * The format has no rotation, so a sloped plane cannot be a rotated box — it is
 * a prism whose CROSS-SECTION carries the slope. Extruding along the ridge axis
 * means one piece covers the whole slope and the pitch is baked into the hull.
 */
export function roofSlope(builder, {
  type = 'roof', material, ridgeAxis = 'z', lo, hi,
  eaveU, eaveV, ridgeU, ridgeV, thickness,
}) {
  // Thickness is added STRAIGHT UP, not perpendicular to the slope, so the two
  // ends are vertical faces. A perpendicular offset also shifts the ends along
  // the slope — on a garage ramp that pushed each end 48 mm back into the
  // landing it was supposed to meet, which reads as an authored overlap. It is
  // also how a sloping slab is actually cut.
  const poly = [
    [eaveU, eaveV], [ridgeU, ridgeV],
    [ridgeU, ridgeV + thickness], [eaveU, eaveV + thickness],
  ];
  const inPlane = ridgeAxis === 'z' ? poly : poly.map(([a, b]) => [b, a]);
  return builder.piece({ type, material, axis: ridgeAxis, lo, hi, poly: inPlane });
}

/**
 * One facade's balcony band: straight segments along a face whose OUTWARD depth
 * varies, so the silhouette reads as a curve without any piece being curved.
 *
 * Built per facade rather than as a ring around the building, and that is a
 * geometric necessity rather than a style choice. A ring following a rounded
 * rectangle cuts a diagonal chord across each corner of a rectangular slab, and
 * that chord lies inside the slab — an authored overlap the contact finder
 * rightly refuses to turn into a bond. Per-facade bands stop at the building
 * line, so the +Z band never reaches past x = hw and the +X band never reaches
 * past z = hd; they meet at the corner instead of interpenetrating, and a
 * corner block fills what is left.
 *
 * @param face    '+x' | '-x' | '+z' | '-z'
 * @param half    [hx, hz] building half-extents; the band starts at this line
 * @param lo,hi   vertical extent of the deck
 * @param depth   t in [0,1] -> outward projection in metres
 * @returns the deck polygons in XZ, so a parapet can be built on their outer edge
 */
export function facadeBand(builder, {
  type = 'balcony', material, face, half, lo, hi, segments, depth,
}) {
  const [hx, hz] = half;
  const alongX = face === '+z' || face === '-z';
  const sign = face === '+x' || face === '+z' ? 1 : -1;
  const a0 = alongX ? -hx : -hz, a1 = alongX ? hx : hz;
  const base = alongX ? hz : hx;
  const decks = [];

  for (let k = 0; k < segments; k++) {
    const t0 = k / segments, t1 = (k + 1) / segments;
    const s0 = a0 + (a1 - a0) * t0, s1 = a0 + (a1 - a0) * t1;
    const d0 = depth(t0), d1 = depth(t1);
    // Trapezoid: flat on the building line, stepped out to the varying depth.
    const poly = alongX
      ? [[s0, sign * base], [s1, sign * base], [s1, sign * (base + d1)], [s0, sign * (base + d0)]]
      : [[sign * base, s0], [sign * base, s1], [sign * (base + d1), s1], [sign * (base + d0), s0]];
    builder.piece({ type, material, axis: 'y', poly, lo, hi });
    decks.push({ poly, alongX, sign, base, s0, s1, d0, d1 });
  }
  return decks;
}

/** The corner block that fills the notch where two facade bands meet. */
export function bandCorner(builder, { type = 'balcony', material, half, lo, hi, sx, sz, depth }) {
  const [hx, hz] = half;
  builder.piece({
    type, material, axis: 'y', lo, hi,
    poly: [
      [sx * hx, sz * hz], [sx * (hx + depth), sz * hz],
      [sx * (hx + depth), sz * (hz + depth)], [sx * hx, sz * (hz + depth)],
    ],
  });
}

/**
 * A parapet standing on the outer lip of a balcony deck.
 *
 * Takes the deck polygon it sits on rather than recomputing the geometry, so
 * the two cannot drift apart and leave a rail hanging in the air.
 */
export function parapetOn(builder, deck, { material, lo, hi, thickness = 0.16, type = 'parapet' }) {
  const { alongX, sign, base, s0, s1, d0, d1 } = deck;
  const o0 = base + d0, o1 = base + d1;
  const i0 = o0 - thickness, i1 = o1 - thickness;
  const poly = alongX
    ? [[s0, sign * i0], [s1, sign * i1], [s1, sign * o1], [s0, sign * o0]]
    : [[sign * i0, s0], [sign * i1, s1], [sign * o1, s1], [sign * o0, s0]];
  builder.piece({ type, material, axis: 'y', poly, lo, hi });
}

/**
 * A dog-leg staircase: two flights and a half-landing, floor to floor.
 *
 * The steps are the interesting part. The obvious construction — a box per
 * tread, offset by one run and one riser — produces steps that meet only along
 * an EDGE, which is zero contact area and therefore no bond at all; the flight
 * would arrive as a column of unconnected blocks and fall over. Each step here
 * is instead long enough to overlap the one below by `bearing`, so consecutive
 * steps share a real horizontal face and the flight carries itself down to the
 * slab the way a stack of cantilevered treads actually does.
 *
 * @param at      [x, z] of the flight's bottom outer corner
 * @param y0,y1   floor top to floor top; the stair spans exactly this
 * @param axis    'x' or 'z' — the direction the FIRST flight climbs
 * @returns { ...footprint, arrivalDepth, void: {...} }
 *
 * `void` is the part of the footprint a floor plate must LEAVE OPEN, and it is
 * deliberately smaller than the footprint. The second flight tops out short of
 * the footprint's near edge — it has to, because it climbs back the way the
 * first one came — so the strip between them is where you actually step off
 * the stair. Cut the opening to the whole footprint and that strip is a hole:
 * you climb to the top tread, and the nearest floor is on the far side of a
 * 0.9 m drop back down the well. `void` keeps it solid.
 */
/**
 * A builder that emits nothing, for measuring a stair's footprint before
 * placing it. Exported so the stubbed surface stays in one place: it grew a
 * second method when flights gained a waist, and every call site that had
 * hand-rolled `{ box() {} }` broke at once.
 */
export const DRY_RUN = { box() {}, piece() {} };

export function staircase(builder, {
  material = 'concrete-slab', type = 'stair', at, y0, y1, axis = 'x',
  width = 1.25, run = 0.29, landingLen = 1.35, waist = 0.14, clearance = 0.06,
  newelPost = true,
}) {
  const rise = y1 - y0;
  const perFlight = Math.max(2, Math.round(rise / 2 / 0.18));
  const riser = rise / (perFlight * 2);
  const [ax, az] = at;
  const along = axis === 'x' ? 0 : 2;
  const cross = along === 0 ? 2 : 0;
  // The flights stand clear of the shaft walls rather than filling it exactly.
  // Touching them, every wedge in every flight becomes a second path to ground
  // alongside the core — and because a wedge's side face is a few hundred square
  // centimetres, the static walk duly pushed 104 t through one of them at nearly
  // twice yield. A stair spans between floors; it is not a column.
  const lane0 = clearance;

  /** A box, given distance along the flight, lane across it, and height band. */
  const box = (d0, d1, lane, laneW, yLo, yHi) => {
    const lo = [0, yLo, 0], hi = [0, yHi, 0];
    lo[along] = Math.min(d0, d1); hi[along] = Math.max(d0, d1);
    lo[cross] = lane; hi[cross] = lane + laneW;
    builder.box({
      type, material,
      min: [ax + lo[0], lo[1], az + lo[2]],
      max: [ax + hi[0], hi[1], az + hi[2]],
    });
  };

  /**
   * A prism whose CROSS-SECTION is in the (along, vertical) plane, extruded
   * across the lane — the shape a flight actually is, seen from the side.
   */
  const profile = (poly, lane, laneW) => {
    if (along === 0) {
      // Extruded across Z, so the section lives in (x, y).
      builder.piece({
        type, material, axis: 'z', lo: az + lane, hi: az + lane + laneW,
        poly: poly.map(([d, y]) => [ax + d, y]),
      });
    } else {
      // Extruded across X, so the section lives in (y, z).
      builder.piece({
        type, material, axis: 'x', lo: ax + lane, hi: ax + lane + laneW,
        poly: poly.map(([d, y]) => [y, az + d]),
      });
    }
  };

  /**
   * One flight: a solid bottom step, an inclined WAIST, and a wedge per tread
   * sitting on it.
   *
   * The waist is the whole point. Without it a flight is a stack of thin
   * treads each overhanging the one below by a full tread depth, cantilevered
   * over a void — which is what a naive "box per step, offset by a run and a
   * riser" produces, and it reads in game as floating plates with open risers.
   * A cast flight is an inclined slab with the steps formed on top of it, and
   * that is also what carries the load: every wedge bears on the waist over its
   * full sloping underside, and the waist spans floor to landing in one piece.
   *
   * @param dir +1 climbs in the +along direction, -1 climbs back
   */
  const flight = (startDist, startY, steps, lane, dir) => {
    const d = (k) => startDist + dir * k * run;      // distance at step k's back edge
    // Step one is solid down to whatever it stands on. It also holds the low end
    // of the waist clear of that floor: the waist's underside at the springing
    // would otherwise sit `waist` BELOW floor level, inside the slab.
    box(d(0), d(1), lane, width, startY, startY + riser);

    // Waist, from the back of step one to the head of the flight. Its top face
    // passes through every nosing; its underside is that line dropped `waist`.
    const dTop = d(steps);
    const yLow = startY + riser, yTop = startY + steps * riser;
    profile([
      [d(1), yLow - waist], [dTop, yTop - waist], [dTop, yTop], [d(1), yLow],
    ], lane, width);

    // A wedge per remaining tread, resting on the waist along its hypotenuse.
    //
    // Each stops 2 mm short of the next. Consecutive wedges meet where one's
    // tread ends and the next one's riser begins — a LINE, with no area — but
    // the contact test still finds a sliver there and turns it into a bond, and
    // a sliver is the worst kind: stress is force over area, so 46 t through
    // 90 cm^2 reported a flight at 110% of yield. The gap is invisible and each
    // wedge still bears on the waist over its whole underside.
    const gap = 0.002;
    for (let k = 1; k < steps; k += 1) {
      const back = d(k), front = d(k + 1) - dir * gap;
      const yBack = startY + k * riser, yFront = startY + (k + 1) * riser - gap * (riser / run);
      profile([[back, yBack], [front, yFront], [back, startY + (k + 1) * riser]], lane, width);
    }
  };

  // ── flight one, climbing away from `at` in the near lane ─────────────────
  const landingStart = (perFlight - 1) * run;
  const midY = y0 + perFlight * riser;
  flight(0, y0, perFlight - 1, lane0, 1);

  // ── the half-landing ─────────────────────────────────────────────────────
  // Thicker than a tread by exactly the waist, so its back face meets the end
  // of the incoming waist over a real area rather than along an edge.
  //
  // It spans the FULL width of the well, not just the two flights, so that in a
  // shafted stair it bears on both walls — which is how a half-landing is
  // actually carried. Stopped at the flights it was held only along its lane-A
  // edge, by the waist arriving underneath, and the whole lane-B half of it
  // cantilevered off that.
  const landingEnd = landingStart + landingLen;
  const wellWidth = 2 * width + 2 * clearance;
  const landingTop = midY, landingBottom = midY - riser - waist;
  box(landingStart, landingEnd, 0, wellWidth, landingBottom, landingTop);

  // A newel post under its outer corner — for an OPEN stair, where nothing else
  // holds that corner up and a real feature stair puts one there.
  //
  // Off inside a shaft, and not merely as an optimisation: the post stands at
  // the far end of the well, which on every floor but the lowest is over the
  // VOID of the storey below. It would land on nothing. Inside a shaft the
  // landing reaches both walls and they carry it.
  if (newelPost) {
    const post = Math.min(0.22, landingLen / 3);
    box(landingEnd - post, landingEnd, wellWidth - post, post, y0, landingBottom);
  }

  // ── flight two, climbing back the other way in the far lane ──────────────
  flight(landingEnd, midY, perFlight, lane0 + width, -1);

  // The footprint, and the part of it a floor must leave open. The arrival
  // strip runs from the near edge to the head of the second flight — the floor
  // you actually step onto. Cut the opening to the whole footprint instead and
  // the head of every flight faces a hole back down the well.
  const arrivalDepth = landingLen - run;
  if (!(arrivalDepth > 0.4)) {
    throw new Error(
      `staircase: landing ${landingLen} m leaves only ${arrivalDepth.toFixed(2)} m to step off ` +
      `onto. Needs landingLen > run + 0.4 (= ${(run + 0.4).toFixed(2)} m here).`,
    );
  }

  const fp = { x0: ax, x1: ax, z0: az, z1: az, arrivalDepth };
  if (along === 0) {
    fp.x1 = ax + landingEnd;
    fp.z1 = az + 2 * width + 2 * clearance;
    fp.void = { x0: ax + arrivalDepth, x1: fp.x1, z0: az, z1: fp.z1 };
  } else {
    fp.z1 = az + landingEnd;
    fp.x1 = ax + 2 * width + 2 * clearance;
    fp.void = { x0: ax, x1: fp.x1, z0: az + arrivalDepth, z1: fp.z1 };
  }
  return fp;
}

/**
 * Shaft walls around a stair, leaving one side open to walk in through.
 *
 * A real tower's stair core also BRACES it, which is why the towers here get
 * one and the houses get an open feature stair instead.
 */
export function stairShaft(builder, {
  material = 'reinforced-concrete', type = 'core', footprint, y0, y1, thickness = 0.25, openSide = '-x',
}) {
  const { x0, x1, z0, z1 } = footprint;
  const sides = {
    '-x': [[x0 - thickness, y0, z0 - thickness], [x0, y1, z1 + thickness]],
    '+x': [[x1, y0, z0 - thickness], [x1 + thickness, y1, z1 + thickness]],
    '-z': [[x0, y0, z0 - thickness], [x1, y1, z0]],
    '+z': [[x0, y0, z1], [x1, y1, z1 + thickness]],
  };
  for (const [side, [min, max]] of Object.entries(sides)) {
    if (side === openSide) continue;
    builder.box({ type, material, min, max });
  }
}

/**
 * One footprint covering every storey's flight, with its arrival strip intact.
 *
 * Storey heights differ — a taller ground floor needs a longer flight — so a
 * shaft has to be cut for the longest of them. Taking the union edge-by-edge
 * loses the `void`, which is what keeps the head of the flight standing on
 * something, so it is rebuilt here from the arrival depth (constant across
 * storeys: it depends only on the landing length and the tread).
 */
export function unionStairFootprints(fps) {
  const base = fps[0];
  const out = {
    x0: Math.min(...fps.map((f) => f.x0)), x1: Math.max(...fps.map((f) => f.x1)),
    z0: Math.min(...fps.map((f) => f.z0)), z1: Math.max(...fps.map((f) => f.z1)),
    arrivalDepth: Math.min(...fps.map((f) => f.arrivalDepth)),
  };
  // Which axis the flights run along, read off where the base footprint put its
  // own arrival strip rather than passed in again and able to disagree.
  const alongX = base.void.x0 !== base.x0;
  out.void = alongX
    ? { x0: out.x0 + out.arrivalDepth, x1: out.x1, z0: out.z0, z1: out.z1 }
    : { x0: out.x0, x1: out.x1, z0: out.z0 + out.arrivalDepth, z1: out.z1 };
  return out;
}

/**
 * The outer face of a stair shaft — the void a floor plate must leave for it.
 *
 * The plate's opening has to match the shaft's OUTER face, not the stair's own
 * footprint. Cut it to the stair and the core walls stand inside the hole,
 * touching each floor only along a sliver of edge: on the 432 Park tower that
 * left the core tied to its plates over 0.19 m^2 and carrying 4,500 t. Matching
 * the outer face puts the slab edge flat against the wall, which is how a slab
 * frames into a core.
 */
export function shaftVoid(footprint, thickness = 0.25) {
  const v = footprint.void ?? footprint;
  // Expanded on every edge EXCEPT the arrival one, which is not an edge of the
  // opening at all — it is where the floor continues under the head of the
  // flight. Expanding it would reopen the hole this exists to close.
  const keep = {
    x0: v.x0 !== footprint.x0, x1: v.x1 !== footprint.x1,
    z0: v.z0 !== footprint.z0, z1: v.z1 !== footprint.z1,
  };
  return {
    x0: v.x0 - (keep.x0 ? 0 : thickness), x1: v.x1 + (keep.x1 ? 0 : thickness),
    z0: v.z0 - (keep.z0 ? 0 : thickness), z1: v.z1 + (keep.z1 ? 0 : thickness),
  };
}

/**
 * A floor plate with a rectangular hole in it.
 *
 * A plate with a hole is concave, and colliders here are cuboids or convex
 * hulls, so it has to be four panels around the void rather than one piece.
 * The decomposition covers the plate exactly and never overlaps, which is what
 * keeps the interpenetration check clean.
 */
export function slabWithOpening(builder, {
  type = 'slab', material = 'concrete-slab', min, max, opening, inset = 0,
}) {
  const [x0, y0, z0] = min, [x1, y1, z1] = max;
  const o = opening;
  const panel = (a0, b0, a1, b1) => {
    if (!(a1 - a0 > 0.05) || !(b1 - b0 > 0.05)) return;
    builder.box({ type, material, min: [a0, y0, b0], max: [a1, y1, b1] });
  };
  panel(x0 + inset, z0 + inset, o.x0, z1 - inset);           // strip left of the hole
  panel(o.x1, z0 + inset, x1 - inset, z1 - inset);           // strip right of it
  panel(o.x0, z0 + inset, o.x1, o.z0);                       // in front of it
  panel(o.x0, o.z1, o.x1, z1 - inset);                       // behind it
}

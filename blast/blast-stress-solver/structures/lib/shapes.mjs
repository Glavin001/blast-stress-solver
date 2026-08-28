/**
 * Cross-section polygons for the pieces a building is made of.
 *
 * Everything here returns a convex polygon in a piece's plane coordinates, to
 * be extruded by ScenePackBuilder.piece(). Convexity is the hard requirement:
 * colliders are cuboids or convex hulls, never trimeshes, so a concave outline
 * has to be authored as several pieces rather than one.
 */

/** Axis-aligned rectangle. */
export const rectPoly = (u0, w0, u1, w1) => [[u0, w0], [u1, w0], [u1, w1], [u0, w1]];

/**
 * Point and outward normal at normalised perimeter position `t` on a rounded
 * rectangle of half-extents (hu, hw) with corner radius `r`.
 *
 * The rounded rectangle is the spine of every balcony band: the Algedra
 * building's slabs are a soft-cornered rectangle, not a circle, and reading the
 * outline off a real perimeter parameterisation keeps segment lengths even
 * around the corners instead of bunching them.
 */
export function roundedRectAt(hu, hw, r, t) {
  r = Math.min(r, hu, hw);
  const su = 2 * (hu - r), sw = 2 * (hw - r); // straight run lengths
  const arc = (Math.PI / 2) * r;
  const total = 2 * su + 2 * sw + 4 * arc;
  let d = ((t % 1) + 1) % 1 * total;

  // Walk the perimeter counter-clockwise from the middle of the +u edge.
  const segs = [
    { kind: 'line', len: sw / 2, from: [hu, 0], dir: [0, 1], n: [1, 0] },
    { kind: 'arc', len: arc, c: [hu - r, hw - r], a0: 0 },
    { kind: 'line', len: su, from: [hu - r, hw], dir: [-1, 0], n: [0, 1] },
    { kind: 'arc', len: arc, c: [-(hu - r), hw - r], a0: Math.PI / 2 },
    { kind: 'line', len: sw, from: [-hu, hw - r], dir: [0, -1], n: [-1, 0] },
    { kind: 'arc', len: arc, c: [-(hu - r), -(hw - r)], a0: Math.PI },
    { kind: 'line', len: su, from: [-(hu - r), -hw], dir: [1, 0], n: [0, -1] },
    { kind: 'arc', len: arc, c: [hu - r, -(hw - r)], a0: (3 * Math.PI) / 2 },
    { kind: 'line', len: sw / 2, from: [hu, -(hw - r)], dir: [0, 1], n: [1, 0] },
  ];
  for (const s of segs) {
    if (d > s.len && s !== segs[segs.length - 1]) { d -= s.len; continue; }
    if (s.kind === 'line') {
      return { p: [s.from[0] + s.dir[0] * d, s.from[1] + s.dir[1] * d], n: s.n };
    }
    const a = s.a0 + d / r;
    const n = [Math.cos(a), Math.sin(a)];
    return { p: [s.c[0] + r * n[0], s.c[1] + r * n[1]], n };
  }
  throw new Error('unreachable');
}

/**
 * A ring of convex quads following a rounded rectangle — one floor's balcony band.
 *
 * `depth(t)` gives the band's outward depth at perimeter position t, so a
 * sinusoidal depth produces the wave that makes the Algedra facade read as
 * organic rather than as a racetrack. Each segment is a 4-vertex quad, which
 * costs 24 hull points — well inside the 64-point PhysX GPU cook limit.
 *
 * Segments are emitted as separate pieces so the band can fracture and fall
 * apart in sections; a single ring would be one indestructible loop.
 */
export function bandSegments({ hu, hw, radius, segments, inset = 0, depth }) {
  const out = [];
  for (let k = 0; k < segments; k++) {
    const t0 = k / segments, t1 = (k + 1) / segments;
    const a = roundedRectAt(hu, hw, radius, t0);
    const b = roundedRectAt(hu, hw, radius, t1);
    const da = typeof depth === 'function' ? depth(t0) : depth;
    const db = typeof depth === 'function' ? depth(t1) : depth;
    const inner = [
      [a.p[0] + a.n[0] * inset, a.p[1] + a.n[1] * inset],
      [b.p[0] + b.n[0] * inset, b.p[1] + b.n[1] * inset],
    ];
    const outer = [
      [b.p[0] + b.n[0] * (inset + db), b.p[1] + b.n[1] * (inset + db)],
      [a.p[0] + a.n[0] * (inset + da), a.p[1] + a.n[1] * (inset + da)],
    ];
    const quad = [inner[0], inner[1], outer[0], outer[1]];
    if (Math.abs(signedArea(quad)) < 1e-6) continue;
    out.push(quad);
  }
  return out;
}

function signedArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

/**
 * A gable cross-section: the triangle-topped end wall of a pitched roof, as a
 * convex polygon in the (u, w) = (horizontal, vertical) plane.
 */
export const gablePoly = (halfWidth, wallTop, ridge) => [
  [-halfWidth, 0], [halfWidth, 0], [halfWidth, wallTop], [0, ridge], [-halfWidth, wallTop],
];

// ── radial: concentric rings ────────────────────────────────────────────────
//
// A ring wall is the one shape the format is least able to express. There is no
// rotation, so curvature has to be baked into vertices; and a cross-section is
// capped at MAX_POLY_VERTS = 10, so an arc can never be one polygon. The
// established answer, from Petronas, is one convex quad per angular segment,
// extruded vertically -- these helpers just stop every caller rewriting it.

/** A point on a circle, in the (x, z) plane an `axis: 'y'` piece uses. */
export const ringPoint = (cx, cz, r, t) => [cx + r * Math.cos(t), cz + r * Math.sin(t)];

/**
 * An annulus, as one convex quad per segment.
 *
 * Angles run counter-clockwise from `from` to `to` (default: a full turn).
 * Each quad is [inner(t0), outer(t0), outer(t1), inner(t1)], so consecutive
 * segments share their radial face EXACTLY -- same two vertices, same floats --
 * which is what makes them bond to each other instead of merely touching.
 *
 * `skip(i, t0, t1)` omits a segment, for a gateway or a road passing through.
 *
 * Segment count is a real decision, not a detail. Petronas notes that twelve
 * "reads as round from any distance you would look at a tower from", and picks
 * boundaries that land on the features below so no segment straddles one. Hold
 * the chord near a constant length as the radius shrinks rather than keeping
 * the count fixed, or the inner rings cost segments they do not need.
 */
export function ringSegments({ cx = 0, cz = 0, rInner, rOuter, segments,
                               from = 0, to = Math.PI * 2, skip = null }) {
  if (!(rOuter > rInner)) throw new Error(`ringSegments: rOuter ${rOuter} <= rInner ${rInner}`);
  const out = [];
  for (let i = 0; i < segments; i += 1) {
    const t0 = from + ((to - from) * i) / segments;
    const t1 = from + ((to - from) * (i + 1)) / segments;
    if (skip && skip(i, t0, t1)) continue;
    out.push({
      i,
      t0,
      t1,
      tMid: (t0 + t1) / 2,
      poly: [
        ringPoint(cx, cz, rInner, t0), ringPoint(cx, cz, rOuter, t0),
        ringPoint(cx, cz, rOuter, t1), ringPoint(cx, cz, rInner, t1),
      ],
    });
  }
  return out;
}

/**
 * Segment count that holds a chord near `chord` metres at this radius.
 *
 * Rounded to a multiple of 4 so the cardinal directions always fall on a
 * segment boundary -- gates, the spur and the roads are all placed on axes, and
 * a segment straddling one cannot be cleanly skipped for an opening.
 */
export function segmentsForChord(radius, chord = 15, min = 8) {
  const raw = Math.ceil((Math.PI * 2 * radius) / chord);
  return Math.max(min, Math.round(raw / 4) * 4);
}

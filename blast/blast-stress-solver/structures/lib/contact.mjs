/**
 * Exact face contact between two convex prisms.
 *
 * Bounding boxes cannot do this job. A pitched roof sheet's AABB spans from its
 * eave to its ridge, so it "overlaps" the ridge beam beneath it by 22 cm on
 * paper while the real surfaces only touch — and a contact finder working on
 * boxes throws that bond away as a modelling error, leaving the roof
 * unsupported and floating.
 *
 * The prisms here have known geometry, so their face normals are exact: the two
 * extrude-axis caps, plus one per cross-section edge. That is the complete
 * candidate set for a separating-axis test between convex polytopes whose
 * contact is face-to-face, which is what a building is made of.
 *
 * The contact patch is then the overlap of the two pieces' SHADOWS on the plane
 * perpendicular to the contact normal. For a face contact that shadow overlap
 * is exactly the bearing area — the same quantity `convexIntersectArea` gives
 * for two stacked slabs, generalised to any orientation.
 */
import { polygonArea, convexIntersectArea } from '../../scripts/export-fractured-city.mjs';

const TOUCH_EPS = 1e-3;
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Plane axes (u, w) and the world mapping, per extrude axis. Mirrors pack.mjs. */
const FRAME = {
  x: { toWorld: (u, w, t) => [t, u, w], normal: [1, 0, 0], eu: [0, 1, 0], ew: [0, 0, 1] },
  y: { toWorld: (u, w, t) => [u, t, w], normal: [0, 1, 0], eu: [1, 0, 0], ew: [0, 0, 1] },
  z: { toWorld: (u, w, t) => [u, w, t], normal: [0, 0, 1], eu: [1, 0, 0], ew: [0, 1, 0] },
};

/** World vertices of a prism: its cross-section at both ends of the extent. */
export function prismVertices({ axis, poly, lo, hi }) {
  const f = FRAME[axis];
  const out = [];
  for (const t of [lo, hi]) for (const [u, w] of poly) out.push(f.toWorld(u, w, t));
  return out;
}

/**
 * Unit face normals of a prism: the two caps plus one per cross-section edge.
 * Signs do not matter — the SAT below projects onto the line, not the ray.
 */
export function faceNormals({ axis, poly }) {
  const f = FRAME[axis];
  const out = [f.normal];
  const ccw = polygonArea(poly, true) >= 0 ? 1 : -1;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    // Outward normal of edge p->q for CCW winding is (dy, -dx).
    const nu = (q[1] - p[1]) * ccw, nw = -(q[0] - p[0]) * ccw;
    const len = Math.hypot(nu, nw);
    if (len < 1e-9) continue;
    const wu = nu / len, ww = nw / len;
    out.push([
      f.eu[0] * wu + f.ew[0] * ww,
      f.eu[1] * wu + f.ew[1] * ww,
      f.eu[2] * wu + f.ew[2] * ww,
    ]);
  }
  return out;
}

function projectExtent(verts, axis) {
  let lo = Infinity, hi = -Infinity;
  for (const v of verts) {
    const d = dot3(v, axis);
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  return [lo, hi];
}

/** Andrew's monotone chain — the 2D convex hull of a projected point set. */
function hull2D(points) {
  const pts = [...points].sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (src) => {
    const h = [];
    for (const p of src) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    h.pop();
    return h;
  };
  return [...half(pts), ...half([...pts].reverse())];
}

/** Two unit vectors spanning the plane perpendicular to `n`. */
function basisFor(n) {
  const up = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const e0 = [
    n[1] * up[2] - n[2] * up[1],
    n[2] * up[0] - n[0] * up[2],
    n[0] * up[1] - n[1] * up[0],
  ];
  const l0 = Math.hypot(...e0) || 1;
  const a = [e0[0] / l0, e0[1] / l0, e0[2] / l0];
  const b = [
    n[1] * a[2] - n[2] * a[1],
    n[2] * a[0] - n[0] * a[2],
    n[0] * a[1] - n[1] * a[0],
  ];
  return [a, b];
}

/**
 * The bearing between two prisms, or null when they do not touch.
 *
 * `maxPenetration` rejects a pair that shares real volume: that is an authoring
 * error, not a joint, and turning it into a bond would hide it.
 *
 * @returns { area, normal } — area in m^2, normal the true contact-surface
 *          normal. The solver splits load into normal (compression/tension) and
 *          tangential (shear) components about this vector, so a centroid-to-
 *          centroid direction here would book compression as shear.
 */
export function prismContact(A, B, { maxPenetration = 0.02 } = {}) {
  const va = prismVertices(A), vb = prismVertices(B);
  const axes = [...faceNormals(A), ...faceNormals(B)];

  let bestAxis = null, bestOverlap = Infinity;
  for (const ax of axes) {
    const [a0, a1] = projectExtent(va, ax);
    const [b0, b1] = projectExtent(vb, ax);
    const overlap = Math.min(a1, b1) - Math.max(a0, b0);
    if (overlap < -TOUCH_EPS) return null;          // a separating axis: no contact
    if (overlap < bestOverlap) { bestOverlap = overlap; bestAxis = ax; }
  }
  if (!bestAxis) return null;
  if (bestOverlap > maxPenetration) return null;    // shared volume, not a bearing

  // Shadow overlap on the plane perpendicular to the contact normal.
  const [e0, e1] = basisFor(bestAxis);
  const flat = (verts) => hull2D(verts.map((v) => [dot3(v, e0), dot3(v, e1)]));
  const pa = flat(va), pb = flat(vb);
  if (pa.length < 3 || pb.length < 3) return null;
  const area = convexIntersectArea(pa, pb);
  if (!(area > 0)) return null;
  // Reject a contact that is really an EDGE rather than a face.
  //
  // Two stair wedges meet along the line where one's tread ends and the next
  // one's riser begins — geometrically a line, zero area. The shadow overlap
  // returns a sliver there anyway, and a sliver is the worst possible bond:
  // stress is force over area, so a 50 cm^2 phantom contact took 26 t and
  // reported the flight at 112% of yield. A real bearing is a decent fraction
  // of the smaller face; anything far below that is numerical.
  const smallest = Math.min(polygonArea(pa), polygonArea(pb));
  if (area < 0.02 * smallest) return null;
  return { area, normal: bestAxis, penetration: Math.max(0, bestOverlap) };
}

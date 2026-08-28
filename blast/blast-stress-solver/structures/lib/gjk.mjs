import { colliderOf } from './colliders.mjs';
/**
 * Do two convex point clouds overlap? GJK, boolean form.
 *
 * Bounding boxes cannot answer this for the shapes a building is made of. Two
 * roof slopes meeting at a ridge have heavily overlapping AABBs and share only
 * a face; two Voronoi shards of one wall tile exactly and still have
 * overlapping AABBs. Testing those with boxes reports hundreds of collisions
 * that are not there.
 *
 * GJK needs no face or edge topology, only a support function — the farthest
 * point of a set along a direction — which a point cloud gives directly. That
 * makes it the right fit here, where colliders arrive as bare vertex lists.
 */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const neg = (a) => [-a[0], -a[1], -a[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
/** (a x b) x c — the "perpendicular to ab, pointing towards c" idiom GJK runs on. */
const tripleCross = (a, b, c) => cross(cross(a, b), c);

function support(points, d) {
  let best = 0, bestDot = -Infinity;
  for (let i = 0; i < points.length; i++) {
    const v = dot(points[i], d);
    if (v > bestDot) { bestDot = v; best = i; }
  }
  return points[best];
}

const minkowski = (A, B, d) => sub(support(A, d), support(B, neg(d)));

/**
 * True when the convex hulls of `A` and `B` share interior volume.
 *
 * Pieces that merely touch (a column seated on a slab) must NOT count, so each
 * cloud is shrunk towards its own centroid by `margin` first. Shrinking rather
 * than testing distance keeps this a pure boolean test.
 */
export function hullsOverlap(A, B, margin = 0.01) {
  A = shrink(A, margin);
  B = shrink(B, margin);

  let d = sub(centroid(B), centroid(A));
  if (dot(d, d) < 1e-18) d = [1, 0, 0];
  let a = minkowski(A, B, d);
  if (dot(a, d) < 0) return false;              // no overlap along the very first direction
  const simplex = [a];
  d = neg(a);

  for (let iter = 0; iter < 64; iter++) {
    a = minkowski(A, B, d);
    if (dot(a, d) <= 0) return false;           // farthest point falls short of the origin
    simplex.push(a);
    const step = nextSimplex(simplex, d);
    if (step === true) return true;
    d = step;
    if (dot(d, d) < 1e-20) return true;         // origin lies exactly on the simplex
  }
  // Non-convergence on a degenerate cloud: report no overlap rather than
  // inventing one, and let the caller's other checks speak.
  return false;
}

function centroid(points) {
  const c = [0, 0, 0];
  for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  return [c[0] / points.length, c[1] / points.length, c[2] / points.length];
}

function shrink(points, margin) {
  const c = centroid(points);
  return points.map((p) => {
    const r = sub(p, c);
    const len = Math.hypot(...r);
    const k = len > margin ? (len - margin) / len : 0;
    return [c[0] + r[0] * k, c[1] + r[1] * k, c[2] + r[2] * k];
  });
}

/** Evolve the simplex towards the origin. Returns `true` if it encloses it, else a new direction. */
function nextSimplex(s, d) {
  if (s.length === 2) {
    const [b, a] = s;
    const ab = sub(b, a), ao = neg(a);
    return dot(ab, ao) > 0 ? tripleCross(ab, ao, ab) : (s.splice(0, 1), ao);
  }
  if (s.length === 3) {
    const [c, b, a] = s;
    const ab = sub(b, a), ac = sub(c, a), ao = neg(a);
    const abc = cross(ab, ac);
    if (dot(cross(abc, ac), ao) > 0) {
      if (dot(ac, ao) > 0) { s.splice(1, 1); return tripleCross(ac, ao, ac); }
      s.splice(0, 1); return nextSimplex(s, d);
    }
    if (dot(cross(ab, abc), ao) > 0) { s.splice(0, 1); return nextSimplex(s, d); }
    return dot(abc, ao) > 0 ? abc : (s.splice(0, 2, c, b), neg(abc));
  }
  const [dd, c, b, a] = s;
  const ao = neg(a);
  const faces = [[b, c], [c, dd], [dd, b]];
  for (const [p, q] of faces) {
    const n = cross(sub(p, a), sub(q, a));
    if (dot(n, ao) > 0) {
      s.length = 0; s.push(q, p, a);
      return nextSimplex(s, d);
    }
  }
  return true;   // origin is inside the tetrahedron
}

/** World-space vertices of a node's collider. */
export function nodePoints(pack, i) {
  const c = pack.scenario.nodes[i].centroid;
  const col = colliderOf(pack.scenario, i);
  if (col.kind === 'convex_hull') {
    const seen = new Map();
    for (let k = 0; k < col.points.length; k += 3) {
      const key = `${col.points[k]},${col.points[k + 1]},${col.points[k + 2]}`;
      if (!seen.has(key)) seen.set(key, [c.x + col.points[k], c.y + col.points[k + 1], c.z + col.points[k + 2]]);
    }
    return [...seen.values()];
  }
  const h = col.halfExtents, out = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    out.push([c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z]);
  }
  return out;
}

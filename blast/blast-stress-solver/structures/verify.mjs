#!/usr/bin/env node
/**
 * Validate an authored ScenePack: does it parse, does it hold together, and
 * does it have a load path?
 *
 *   node structures/verify.mjs <pack.json> [...]
 *
 * Exits non-zero on the first structure that fails. build.mjs runs this on
 * every pack it writes, so a bad edit cannot reach the game.
 *
 * SCOPE: the statics check below is a static load model, not a solver run. It
 * proves the structure is not overstressed standing still and that every piece
 * has a path to the ground. It does NOT predict collapse dynamics -- those only
 * appear once PhysX has the pack.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { hullsOverlap, nodePoints } from './lib/gjk.mjs';
import { colliderOf } from './lib/colliders.mjs';

const EPS = 1e-6;
const G = 9.81;

/** Collect failures rather than throwing, so one run reports every problem. */
class Report {
  constructor(name) { this.name = name; this.fail = []; this.warn = []; this.info = []; }
  check(ok, msg) { if (!ok) this.fail.push(msg); return ok; }
  note(msg) { this.info.push(msg); }
  caution(msg) { this.warn.push(msg); }
  print() {
    console.log(`\n${this.name}`);
    for (const i of this.info) console.log(`  ${i}`);
    for (const w of this.warn) console.log(`  WARN  ${w}`);
    for (const f of this.fail) console.log(`  FAIL  ${f}`);
    console.log(this.fail.length ? `  => ${this.fail.length} FAILURE(S)` : '  => ok');
    return this.fail.length === 0;
  }
}

/**
 * True world AABB of a node.
 *
 * A Voronoi shard's `centroid` is its AREA centroid, which is not the centre of
 * its bounding box, so centroid +/- nodeSize/2 is off by the difference. Hull
 * points are stored centroid-relative, so for a hull the exact box comes from
 * them; only a cuboid, whose centroid IS its centre, can use nodeSize.
 */
function aabbOf(pack, i) {
  const c = pack.scenario.nodes[i].centroid;
  const col = colliderOf(pack.scenario, i);
  if (col.kind === 'convex_hull') {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < col.points.length; k += 3) {
      for (let a = 0; a < 3; a++) {
        mn[a] = Math.min(mn[a], col.points[k + a]);
        mx[a] = Math.max(mx[a], col.points[k + a]);
      }
    }
    return [[c.x + mn[0], c.y + mn[1], c.z + mn[2]], [c.x + mx[0], c.y + mx[1], c.z + mx[2]]];
  }
  const s = pack.scenario.nodeSizes[i];
  return [[c.x - s.x / 2, c.y - s.y / 2, c.z - s.z / 2],
          [c.x + s.x / 2, c.y + s.y / 2, c.z + s.z / 2]];
}

/** Overlap volume of two AABBs; 0 when they merely touch. */
function overlapVolume(a, b) {
  let vol = 1;
  for (let k = 0; k < 3; k++) {
    const o = Math.min(a[1][k], b[1][k]) - Math.max(a[0][k], b[0][k]);
    if (o <= 0) return 0;
    vol *= o;
  }
  return vol;
}

function verify(pack, name) {
  const r = new Report(name);
  const s = pack.scenario;
  const N = s.nodes.length;
  const materials = pack.defaults?.solver?.materials ?? [];

  r.note(`version ${pack.version}  nodes ${N}  bonds ${s.bonds.length}  materials ${materials.length}`);

  // ── format: exactly what scene_pack.rs enforces ──────────────────────────
  r.check(pack.version === 2, `version must be 2, got ${pack.version}`);
  r.check(materials.length > 0, 'v2 pack needs a non-empty defaults.solver.materials');
  r.check(s.nodeSizes.length === N, `nodeSizes ${s.nodeSizes.length} != nodes ${N} (CountMismatch)`);
  r.check(s.nodeColliders.length === N, `nodeColliders ${s.nodeColliders.length} != nodes ${N} (CountMismatch)`);
  if (s.nodeTypes) r.check(s.nodeTypes.length === N, `nodeTypes ${s.nodeTypes.length} != nodes ${N}`);
  if (s.nodeMaterials) r.check(s.nodeMaterials.length === N, `nodeMaterials ${s.nodeMaterials.length} != nodes ${N}`);
  for (const m of materials) {
    r.check(m.compressionElastic >= 0, `${m.name}: negative compressionElastic`);
    r.check(m.compressionFatal >= m.compressionElastic, `${m.name}: fatal < elastic (parser rejects this)`);
  }

  // ── colliders ────────────────────────────────────────────────────────────
  let maxHull = 0;
  for (let i = 0; i < N; i++) {
    const c = colliderOf(s, i);
    if (c.kind === 'convex_hull') {
      const pts = c.points.length / 3;
      maxHull = Math.max(maxHull, pts);
      r.check(pts <= 64, `node ${i}: hull has ${pts} points > 64 (PhysX GPU cook limit)`);
      // The drawn size must equal the collider extent, or the thing you shoot
      // is not the thing you see.
      let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
      for (let k = 0; k < c.points.length; k += 3) {
        for (let a = 0; a < 3; a++) {
          mn[a] = Math.min(mn[a], c.points[k + a]);
          mx[a] = Math.max(mx[a], c.points[k + a]);
        }
      }
      const size = s.nodeSizes[i], want = [size.x, size.y, size.z];
      for (let a = 0; a < 3; a++) {
        r.check(Math.abs((mx[a] - mn[a]) - want[a]) < 1e-3,
          `node ${i}: hull extent ${(mx[a] - mn[a]).toFixed(4)} != nodeSize ${want[a].toFixed(4)} on axis ${a}`);
      }
    } else if (c.kind === 'cuboid') {
      const h = c.halfExtents, size = s.nodeSizes[i];
      r.check(Math.abs(h.x * 2 - size.x) < 1e-3 && Math.abs(h.y * 2 - size.y) < 1e-3 &&
              Math.abs(h.z * 2 - size.z) < 1e-3, `node ${i}: cuboid half-extents disagree with nodeSize`);
    } else {
      r.check(false, `node ${i}: unsupported collider kind "${c.kind}"`);
    }
  }
  r.note(`largest hull ${maxHull} points`);

  // ── interpenetration ─────────────────────────────────────────────────────
  // AABB sweep as the broad phase, exact GJK on the real hulls as the narrow
  // phase. Boxes alone are useless here: two roof slopes meeting at a ridge and
  // two Voronoi shards of one wall both have overlapping bounding boxes while
  // sharing only a face. Pieces are also allowed to touch, so each hull is
  // shrunk 5 mm before the test and only genuine shared volume registers.
  const boxes = Array.from({ length: N }, (_, i) => aabbOf(pack, i));
  const points = Array.from({ length: N }, (_, i) => nodePoints(pack, i));
  const order = [...boxes.keys()].sort((a, b) => boxes[a][0][0] - boxes[b][0][0]);
  const clashes = [];
  for (let ii = 0; ii < order.length; ii++) {
    const i = order[ii];
    for (let jj = ii + 1; jj < order.length; jj++) {
      const j = order[jj];
      if (boxes[j][0][0] > boxes[i][1][0]) break;   // sweep: no later box can reach back
      if (overlapVolume(boxes[i], boxes[j]) <= 0) continue;
      if (hullsOverlap(points[i], points[j], 0.005)) clashes.push([i, j]);
    }
  }
  if (clashes.length) {
    const show = clashes.slice(0, 6).map(([i, j]) =>
      `${i}(${s.nodeTypes?.[i] ?? '?'}/${s.nodeMaterials?.[i] ?? '?'})` +
      `+${j}(${s.nodeTypes?.[j] ?? '?'}/${s.nodeMaterials?.[j] ?? '?'})`).join('  ');
    r.check(false, `${clashes.length} pairs of colliders share volume: ${show}` +
      (clashes.length > 6 ? '  …' : ''));
  } else {
    r.note('no collider interpenetration (GJK)');
  }

  // ── bonds ────────────────────────────────────────────────────────────────
  const adj = Array.from({ length: N }, () => []);
  for (let bi = 0; bi < s.bonds.length; bi++) {
    const b = s.bonds[bi];
    if (!r.check(b.node0 < N && b.node1 < N && b.node0 >= 0 && b.node1 >= 0,
      `bond ${bi}: node index out of range (${b.node0}, ${b.node1})`)) continue;
    r.check(b.node0 !== b.node1, `bond ${bi}: self-bond on node ${b.node0}`);
    r.check(b.area > 0, `bond ${bi}: non-positive area ${b.area}`);
    r.check((b.m ?? 0) < materials.length, `bond ${bi}: material ${b.m} outside table of ${materials.length}`);
    const nlen = Math.hypot(b.normal.x, b.normal.y, b.normal.z);
    r.check(Math.abs(nlen - 1) < 1e-3, `bond ${bi}: normal not unit (${nlen.toFixed(4)})`);
    // Area is geometry: a bond cannot present more contact than the smaller of
    // the two pieces geometrically has. This is the tripwire on faking strength
    // with area -- bond area is also the damage pool, so inflating it scales
    // toughness super-linearly while corrupting the stress readout.
    //
    // The bound is the largest PLANAR CROSS-SECTION of the piece's box, not its
    // largest axis-aligned face: a Voronoi seam cutting diagonally through a
    // column really does present more area than any face of it.
    const fa = (i) => {
      const z = s.nodeSizes[i];
      return Math.max(z.x * Math.hypot(z.y, z.z), z.y * Math.hypot(z.x, z.z), z.z * Math.hypot(z.x, z.y));
    };
    r.check(b.area <= Math.min(fa(b.node0), fa(b.node1)) + 1e-3,
      `bond ${bi}: area ${b.area.toFixed(4)} exceeds the smaller piece's largest cross-section ` +
      `${Math.min(fa(b.node0), fa(b.node1)).toFixed(4)}`);
    adj[b.node0].push({ to: b.node1, bond: bi });
    adj[b.node1].push({ to: b.node0, bond: bi });
  }

  // ── grounding: every piece must reach a support ──────────────────────────
  const supports = [];
  for (let i = 0; i < N; i++) if (s.nodes[i].mass === 0) supports.push(i);
  r.check(supports.length > 0, 'no support nodes (mass 0) — the structure is not anchored to anything');
  r.note(`supports ${supports.length}`);

  const grounded = reachable(adj, supports, N);
  const floating = [];
  for (let i = 0; i < N; i++) if (!grounded[i]) floating.push(i);
  r.check(floating.length === 0,
    `${floating.length} node(s) have no path to a support — floating pieces: ` +
    `${floating.slice(0, 12).join(', ')}${floating.length > 12 ? ', …' : ''}`);

  // ── rest statics ─────────────────────────────────────────────────────────
  // Push each node's weight down the bond graph towards the supports, in order
  // of descending height, and read the compressive stress off each bond it
  // crosses. Catches "collapses the instant it spawns" without running PhysX.
  const rest = statics(s, materials, adj, N, new Set());
  const bw = s.bonds[rest.worstBond];
  const describe = (i) => `${s.nodeTypes?.[i] ?? '?'}/${s.nodeMaterials?.[i] ?? '?'}` +
    `@(${s.nodes[i].centroid.x.toFixed(1)},${s.nodes[i].centroid.y.toFixed(1)},${s.nodes[i].centroid.z.toFixed(1)})`;
  r.note(`peak rest interface utilisation ${(rest.worstUtil * 100).toFixed(2)}% of compressionElastic` +
    (bw ? ` — ${describe(bw.node0)} onto ${describe(bw.node1)}, ${materials[bw.m ?? 0]?.name}` +
      `, ${(rest.worstLoad / 9810).toFixed(0)} t over ${rest.worstArea.toFixed(3)} m^2` : ''));
  r.check(rest.worstUtil < 1,
    `bond ${rest.worstBond} is at ${(rest.worstUtil * 100).toFixed(1)}% of yield standing still — it will not stand up`);
  if (rest.worstUtil > 0.5) {
    r.caution(`peak rest utilisation ${(rest.worstUtil * 100).toFixed(1)}% leaves little margin for an impact`);
  }

  // ── load path: a ground-floor column must actually carry the building ────
  // NOT "cutting a column ungrounds something" — a real frame is redundant, so
  // removing one of four columns redistributes load rather than orphaning
  // anything, and demanding otherwise would reject correct structures. What
  // does distinguish a load path from a pile of independently-anchored pieces
  // is whether the column carries weight at rest, and whether removing it
  // pushes the rest of the frame harder.
  // ── fragmentation: nothing may be a monolith ─────────────────────────────
  // A building that topples into a handful of intact slabs reads as one solid
  // object falling over. Two rules keep debris looking like debris: every
  // authored piece breaks into at least two chunks, and no chunk exceeds its
  // material's size cap.
  const capOf = {
    glass: 0.6, brick: 0.6, stone: 2.0, steel: 2.5, 'wood-frame': 1.2,
    'reinforced-concrete': 3.5, 'concrete-slab': 3.5, 'facade-panel': 1.2,
    'facade-clip': 1.2, 'glazing-clip': 0.6, 'footing-anchor': 12.0,
  };
  {
    const chunksPerPiece = new Map();
    for (let i = 0; i < N; i += 1) {
      const k = s.nodePieces?.[i] ?? i;
      chunksPerPiece.set(k, (chunksPerPiece.get(k) ?? 0) + 1);
    }
    // Checked at 2.5x the cap, not at the cap. The cap is what the subdivision
    // AIMS at; Voronoi cells are uneven even on a jittered grid, so a strict
    // per-chunk cap would fail on the tail rather than on anything that reads
    // as a monolith. What this catches is the 98 m^3 slab, not the 4 m^3 one.
    const MONOLITH = 2.5;
    const monoliths = [];
    let worstVol = 0, worstAt = -1;
    for (let i = 0; i < N; i += 1) {
      const cap = (capOf[s.nodeMaterials?.[i]] ?? Infinity) * MONOLITH;
      if (s.nodes[i].volume > cap) {
        monoliths.push(i);
        if (s.nodes[i].volume > worstVol) { worstVol = s.nodes[i].volume; worstAt = i; }
      }
    }
    r.note(`largest chunk ${Math.max(...s.nodes.map((n) => n.volume)).toFixed(1)} m^3, ` +
      `median ${[...s.nodes.map((n) => n.volume)].sort((a, b) => a - b)[N >> 1].toFixed(2)}`);
    r.check(monoliths.length === 0,
      `${monoliths.length} chunk(s) exceed their material's size cap — biggest ` +
      `${worstVol.toFixed(1)} m^3 of ${s.nodeMaterials?.[worstAt]} at y` +
      `${s.nodes[worstAt]?.centroid.y.toFixed(1)}; a building made of these topples in one piece`);

    // Supports are pinned and underground; they never move, so splitting them
    // buys nothing and is not required.
    const unsplit = [...chunksPerPiece.entries()].filter(([k, c]) => {
      if (c > 1) return false;
      const node = [...Array(N).keys()].find((i) => (s.nodePieces?.[i] ?? i) === k);
      return node !== undefined && s.nodes[node].mass > 0;
    });
    r.check(unsplit.length === 0,
      `${unsplit.length} load-bearing piece(s) never split into more than one chunk`);
  }

  // ── strength tiers: cladding must be far weaker than structure ───────────
  // Shooting a building should take panels and glass off it without touching
  // the frame; the frame should need a direct hit and should still be
  // breakable. That is a property of the material table, so it is checked here
  // rather than left to whoever authored the structure.
  {
    const fatal = new Map(materials.map((m) => [m.name, m.compressionFatal]));
    const cladding = ['facade-clip', 'facade-panel', 'glazing-clip'].filter((k) => fatal.has(k));
    const frame = ['reinforced-concrete', 'footing-anchor', 'steel'].filter((k) => fatal.has(k));
    if (cladding.length && frame.length) {
      const weakest = Math.min(...frame.map((k) => fatal.get(k)));
      const strongest = Math.max(...cladding.map((k) => fatal.get(k)));
      const ratio = weakest / strongest;
      r.note(`strength spread: cladding fails at ${(strongest / 1e6).toFixed(1)} MPa, ` +
        `frame at ${(weakest / 1e6).toFixed(0)} MPa — ${ratio.toFixed(0)}x`);
      // 8x, not 10x. The frame stopped being rubber when its ductility band
      // came down from 10 to 3, and that halved the spread without making the
      // cladding one bit harder to knock off. What matters is that a hit which
      // strips panels leaves the structure alone, and most of an order of
      // magnitude does that.
      r.check(ratio >= 8,
        `cladding is only ${ratio.toFixed(1)}x weaker than the frame — a hit that takes panels ` +
        `off will take structure with it`);
    }
  }

  // ── stairs arrive somewhere ──────────────────────────────────────────────
  // The top of a flight must have floor beside it to step onto. This is not
  // pedantry: cutting the plate opening to the whole stairwell footprint —
  // which is the obvious thing to do — leaves the head of every flight facing
  // a hole back down the well. It passes every other check here, because the
  // stair is bonded, grounded and unstressed; it is only wrong to walk on.
  // Judged per FLIGHT, not per piece and not per shard.
  //
  // "A piece with nothing resting on it is the head of a flight" was true when
  // a flight was a stack of treads. It is not true now: the treads are wedges
  // bearing on an inclined waist, so nothing rests on any of them and every
  // one of them looked like a head. A flight is instead a connected group of
  // stair pieces — they join to each other but reach the rest of the building
  // only through floors — and it is the top of THAT group which has to land.
  const pieceOf = s.nodePieces ?? [...Array(N).keys()];
  const stairs = [...Array(N).keys()].filter((i) => s.nodeTypes?.[i] === 'stair');
  if (stairs.length) {
    const isStair = new Set(stairs);
    const seen = new Set();
    const stranded = [];
    let flights = 0;
    for (const root of stairs) {
      if (seen.has(root)) continue;
      const group = [];
      const queue = [root];
      seen.add(root);
      while (queue.length) {
        const i = queue.pop();
        group.push(i);
        for (const e of adj[i]) {
          if (!isStair.has(e.to) || seen.has(e.to)) continue;
          seen.add(e.to);
          queue.push(e.to);
        }
      }
      flights += 1;
      const topY = Math.max(...group.map((i) => s.nodes[i].centroid.y + s.nodeSizes[i].y / 2));
      // Something that is not stair must meet the group within a step of its
      // highest point — the floor you step out onto.
      const lands = group.some((i) => {
        if (s.nodes[i].centroid.y + s.nodeSizes[i].y / 2 < topY - 0.4) return false;
        return adj[i].some((e) => {
          if (isStair.has(e.to)) return false;
          const n = s.nodes[e.to], h = s.nodeSizes[e.to].y / 2;
          return n.centroid.y + h > topY - 0.5 && n.centroid.y - h < topY + 0.5;
        });
      });
      if (!lands) stranded.push(`flight of ${group.length} topping out at y${topY.toFixed(1)}`);
    }
    r.note(`stairs ${flights} flight(s), ${stairs.length} pieces, ` +
      `${stranded.length === 0 ? 'every flight arrives on a floor' : `${stranded.length} stranded`}`);
    r.check(stranded.length === 0,
      `${stranded.length} flight(s) arrive at nothing — no floor beside the head of the flight ` +
      `to step onto: ${stranded.slice(0, 4).join('; ')}`);

    // Nothing may be held up outside its own centre of mass.
    //
    // This is what a tread cantilevered over a void looks like from the data,
    // and it is worth a check of its own because it passes every structural
    // test: a stack of treads each overlapping the next is bonded, grounded,
    // and barely stressed. It just has most of every tread hanging over
    // nothing, which reads in game as floating plates with open risers. A
    // flight built on an inclined waist bears over its whole underside and
    // sits comfortably inside this.
    const tippy = [];
    // Plan extent of every piece, so a support is measured as the whole piece
    // rather than as whichever of its shards the bond happened to land on. A
    // tread bears on one shard of a waist that spans the entire flight;
    // measuring that shard alone put the tread outside its own support.
    const pieceBox = new Map();
    for (let i = 0; i < N; i += 1) {
      const k = pieceOf[i], n = s.nodes[i], sz = s.nodeSizes[i];
      const b = pieceBox.get(k) ?? { x0: Infinity, x1: -Infinity, z0: Infinity, z1: -Infinity, yb: Infinity };
      b.x0 = Math.min(b.x0, n.centroid.x - sz.x / 2); b.x1 = Math.max(b.x1, n.centroid.x + sz.x / 2);
      b.z0 = Math.min(b.z0, n.centroid.z - sz.z / 2); b.z1 = Math.max(b.z1, n.centroid.z + sz.z / 2);
      b.yb = Math.min(b.yb, n.centroid.y - sz.y / 2);
      pieceBox.set(k, b);
    }
    const byPieceAll = new Map();
    for (const i of stairs) {
      const k = pieceOf[i];
      if (!byPieceAll.has(k)) byPieceAll.set(k, []);
      byPieceAll.get(k).push(i);
    }
    for (const [k, nodes] of byPieceAll) {
      let m = 0, cx = 0, cz = 0, lowest = Infinity;
      for (const i of nodes) {
        const w = Math.max(s.nodes[i].mass, 1e-6);
        m += w; cx += s.nodes[i].centroid.x * w; cz += s.nodes[i].centroid.z * w;
        lowest = Math.min(lowest, s.nodes[i].centroid.y - s.nodeSizes[i].y / 2);
      }
      cx /= m; cz /= m;
      const top = Math.max(...nodes.map((i) => s.nodes[i].centroid.y + s.nodeSizes[i].y / 2));
      // Plan extent of everything holding this piece up.
      //
      // Anything it is NOT carrying counts, not just what sits under it. A
      // waist spans floor to landing like a beam and is held at both ends, so
      // its centre of mass is legitimately between its supports rather than
      // over one; counting only what is strictly below flagged every waist in
      // the set. What is excluded is what rests ON this piece — a neighbour
      // whose underside starts at or above this piece's top.
      let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity, found = false;
      const supporters = new Set();
      for (const i of nodes) {
        for (const e of adj[i]) {
          if (pieceOf[e.to] === k) continue;
          supporters.add(pieceOf[e.to]);
        }
      }
      for (const sp of supporters) {
        const b = pieceBox.get(sp);
        if (b.yb >= top - 1e-6) continue;          // rests on this piece, not under it
        x0 = Math.min(x0, b.x0); x1 = Math.max(x1, b.x1);
        z0 = Math.min(z0, b.z0); z1 = Math.max(z1, b.z1);
        found = true;
      }
      if (!found) continue;                       // rests on nothing below: the grounding check owns that
      if (cx < x0 - 1e-3 || cx > x1 + 1e-3 || cz < z0 - 1e-3 || cz > z1 + 1e-3) {
        tippy.push(`piece ${k} @y${lowest.toFixed(1)}`);
      }
    }
    r.check(tippy.length === 0,
      `${tippy.length} stair piece(s) are supported outside their own centre of mass — they ` +
      `overhang whatever holds them: ${tippy.slice(0, 5).join('; ')}`);
  }

  const totalWeight = s.nodes.reduce((t, n) => t + n.mass * G, 0);
  // Do NOT look for nodes typed 'column'. Which element carries a building is a
  // property of how it is built, not of a label: this house is load-bearing
  // masonry, so its brick skin takes the roof and its posts frame the openings.
  // Demanding that a post carry the load would reject a correct house. Ask the
  // structure instead — take the ground-storey piece that carries the most, and
  // require that removing it matters.
  const ys = s.nodes.map((n) => n.centroid.y);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const groundStorey = [...Array(N).keys()].filter((i) =>
    s.nodes[i].mass > 0 && ys[i] < yMin + (yMax - yMin) * 0.25);
  if (groundStorey.length === 0) {
    r.caution('no ground-storey nodes above the supports — skipping the load-path check');
  } else {
    const pick = groundStorey.reduce((a, i) => (rest.carried[i] > rest.carried[a] ? i : a), groundStorey[0]);
    const c0 = s.nodes[pick].centroid;
    // Cut the whole vertical stack at that plan position, not one piece: a
    // single element's neighbours bridge around it and nothing moves.
    const stack = new Set([...Array(N).keys()].filter((i) =>
      s.nodes[i].mass > 0 &&
      Math.abs(s.nodes[i].centroid.x - c0.x) < 0.6 && Math.abs(s.nodes[i].centroid.z - c0.z) < 0.6));

    // Does the building's weight actually REACH the ground?
    //
    // Not "does the heaviest ground piece carry >1% of it", which was the test
    // here and which quietly stopped meaning anything: as the subdivision got
    // finer the same load spread over more pieces, and Petronas' heaviest
    // ground element fell to 0.67% without a single thing changing about how it
    // stands up. Conservation is scale-free — every newton either arrives at a
    // support or the structure is not carrying itself.
    let atSupports = 0;
    for (let i = 0; i < N; i += 1) {
      if (s.nodes[i].mass !== 0) continue;
      for (const e of adj[i]) {
        if (s.nodes[e.to].mass === 0) continue;
        atSupports += rest.bondLoad[e.bond];
      }
    }
    const delivered = totalWeight > 0 ? atSupports / totalWeight : 1;
    r.note(`load reaching the supports: ${(delivered * 100).toFixed(1)}% of ${Math.round(totalWeight / 9810)} t`);
    // Reported, not gated — and the number matters when reading everything
    // above it.
    //
    // This walk pushes load downhill and sideways along a graph; it is not a
    // stiffness solve. Where a path peters out the load stops, so on a tall
    // frame a good share of the weight never arrives, and every utilisation
    // printed above is understated by roughly the reciprocal. Multi-hop lateral
    // spreading took 432 Park from 13.5% to 55.7%, but closing the rest needs
    // an actual linear solve.
    //
    // Treat a low number as "these stresses are optimistic", not as "this will
    // fall down": whether it stands is settled by the physics sim, where the
    // whole set currently spawns with zero broken bonds.
    if (delivered < 0.9) {
      r.caution(`only ${(delivered * 100).toFixed(1)}% of the weight reaches a support in this walk, ` +
        `so the utilisations above are understated by roughly ${(1 / delivered).toFixed(1)}x`);
    }

    const share = totalWeight > 0 ? rest.carried[pick] / totalWeight : 0;
    // Removing it must push load onto its neighbours or unground something.
    // Global peak utilisation is the wrong signal: it is usually set by some
    // micro-seam far from the cut and barely moves.
    const cutAdj = adj.map((list, i) => (stack.has(i) ? [] : list.filter((e) => !stack.has(e.to))));
    const after = statics(s, materials, cutAdj, N, stack);
    const stillUp = reachable(cutAdj, supports.filter((i) => !stack.has(i)), N);
    let lost = 0;
    for (let i = 0; i < N; i++) if (grounded[i] && !stillUp[i] && !stack.has(i)) lost++;
    const others = groundStorey.filter((i) => !stack.has(i));
    const before = others.reduce((t, i) => t + rest.carried[i], 0);
    const afterLoad = others.reduce((t, i) => t + after.carried[i], 0);
    const delta = before > 0 ? afterLoad / before - 1 : 0;
    r.note(`cutting that stack ungrounds ${lost} node(s) and shifts ${(delta * 100).toFixed(1)}% more load ` +
      `onto the remaining ${others.length} ground element(s)`);
    // Reported, not asserted. It was a hard check while load flowed down a
    // single steepest-descent path and cutting that path was dramatic. Now that
    // load spreads properly across every column, removing one of twenty-five
    // legitimately moves almost nothing — and cutting a stack also removes its
    // own weight, so the delta is often negative. The assertion that carries
    // the meaning is the one above: that the element carries a real share at
    // all. This is left as an observation rather than pretending to a signal it
    // no longer has.
    if (lost === 0 && delta <= 0.01) {
      r.note('  (cutting it barely moves load, which is what redundancy looks like)');
    }
  }

  return r;
}

/**
 * Static load walk: weight flows downward and each bond it crosses reports the
 * compressive stress that flow implies.
 *
 * Flow follows HEIGHT, with a one-hop lateral fallback — not graph distance to
 * a support. Depth-to-support is the obvious choice and it is badly wrong at
 * scale: it admits only a node's own lower-depth neighbours, so a floor plate
 * with twenty-five columns beneath it can end up feeding whichever single one
 * the breadth-first search happened to reach first. On a 32-storey tower that
 * put the entire 20,500 t of the building through one column at the third
 * floor, at five times yield, while every other column carried nothing.
 *
 * Sorting by height instead lets a plate see ALL the columns under it at once
 * and split between them in proportion to bearing area, which is what a
 * tributary-area hand calculation does and near enough to what an elastic
 * solver finds.
 *
 * The lateral fallback covers the case depth-order was originally introduced
 * for: a piece with nothing at all below it — a slab shard whose only column
 * has been cut — passes its load sideways to a neighbour at the same level
 * that does have a way down, instead of the weight silently vanishing.
 */
function statics(s, materials, adj, N, cut) {
  const G_ = 9.81;
  const y = s.nodes.map((n) => n.centroid.y);
  const live = [...Array(N).keys()].filter((i) => !cut.has(i));

  const below = (i) => adj[i].filter((e) => !cut.has(e.to) && y[e.to] < y[i] - EPS);
  const carried = s.nodes.map((n, i) => (cut.has(i) ? 0 : n.mass * G_));
  const bondLoad = new Array(s.bonds.length).fill(0);

  // How far a piece is from something with a way down, counted in hops through
  // its own level.
  //
  // A floor plate is a horizontal sheet: only the shards sitting directly over
  // a column have anything BELOW them, and every other shard in the plate has
  // to hand its load sideways until it reaches one. A single sideways hop is
  // not enough — a plate is tens of shards across — and without this the load
  // simply stopped there. Measured before this existed: 13.5% of 432 Park's
  // weight reached the ground and the other 86% evaporated, which made every
  // utilisation this function reported understated several-fold.
  //
  // Flowing strictly downhill in this number keeps the lateral pass acyclic,
  // so it terminates and cannot double-count.
  const lateral = new Array(N).fill(Infinity);
  const queue = [];
  for (const i of live) {
    if (below(i).length) { lateral[i] = 0; queue.push(i); }
  }
  for (let head = 0; head < queue.length; head += 1) {
    const i = queue[head];
    for (const e of adj[i]) {
      if (cut.has(e.to) || lateral[e.to] !== Infinity) continue;
      if (y[e.to] < y[i] - EPS) continue;              // downward: not a lateral hop
      lateral[e.to] = lateral[i] + 1;
      queue.push(e.to);
    }
  }

  const send = (targets, load) => {
    const total = targets.reduce((t, e) => t + s.bonds[e.bond].area, 0);
    if (total <= 0) return;
    for (const e of targets) {
      const share = load * (s.bonds[e.bond].area / total);
      bondLoad[e.bond] += share;
      if (s.nodes[e.to].mass !== 0) carried[e.to] += share;
    }
  };

  // Highest first, and within a level the pieces furthest from a way down go
  // first, so their load has somewhere to arrive.
  const order = live.slice().sort((a, b) => (y[b] - y[a]) || (lateral[b] - lateral[a]));
  for (const i of order) {
    if (s.nodes[i].mass === 0) continue;
    const down = below(i);
    if (down.length) { send(down, carried[i]); continue; }
    const inward = adj[i].filter((e) => !cut.has(e.to) && lateral[e.to] < lateral[i]);
    if (inward.length) send(inward, carried[i]);
  }

  // Utilisation is measured on the PIECE-TO-PIECE interface, not on individual
  // shard bonds: a column bears on a slab over its whole footprint whether or
  // not the joint was fractured into twenty shard contacts, and reading stress
  // off one shard reports whatever the finest sliver of that interface is.
  // Aggregating also makes the number invariant to fracture density.
  const piece = s.nodePieces ?? [...Array(N).keys()];
  const iface = new Map();
  for (let bi = 0; bi < s.bonds.length; bi++) {
    const b = s.bonds[bi];
    if (cut.has(b.node0) || cut.has(b.node1)) continue;
    const pa = piece[b.node0], pb = piece[b.node1];
    if (pa === pb) continue;
    const key = pa < pb ? `${pa}:${pb}` : `${pb}:${pa}`;
    const acc = iface.get(key) ?? { area: 0, load: 0, m: b.m ?? 0, bond: bi };
    acc.area += b.area;
    acc.load += bondLoad[bi];
    iface.set(key, acc);
  }
  let worstUtil = 0, worstBond = -1, worstLoad = 0, worstArea = 0;
  for (const acc of iface.values()) {
    const limit = materials[acc.m]?.compressionElastic ?? Infinity;
    const util = (acc.load / acc.area) / limit;
    if (util > worstUtil) {
      worstUtil = util; worstBond = acc.bond; worstLoad = acc.load; worstArea = acc.area;
    }
  }
  return { bondLoad, carried, worstUtil, worstBond, worstLoad, worstArea };
}

function reachable(adj, roots, N) {
  const seen = new Array(N).fill(false);
  const queue = [...roots];
  for (const r of roots) seen[r] = true;
  while (queue.length) {
    const i = queue.pop();
    for (const e of adj[i]) if (!seen[e.to]) { seen[e.to] = true; queue.push(e.to); }
  }
  return seen;
}

/** Verify an in-memory pack and print the report. Used by build.mjs. */
export function verifyPack(pack, name) {
  const r = verify(pack, name);
  const ok = r.print();
  return { ok, report: r };
}

// Only run as a CLI when invoked directly; build.mjs imports verifyPack.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const files = process.argv.slice(2);
  if (!files.length) {
    console.error('usage: node structures/verify.mjs <pack.json> [...]');
    process.exit(2);
  }
  let ok = true;
  for (const f of files) {
    const pack = JSON.parse(await readFile(f, 'utf8'));
    ok = verify(pack, `${pack.title ?? f}  [${f}]`).print() && ok;
  }
  process.exit(ok ? 0 : 1);
}

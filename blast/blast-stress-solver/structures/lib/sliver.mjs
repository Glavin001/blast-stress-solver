/**
 * Dropping contacts too small to be structure.
 *
 * ## What goes wrong
 *
 * Voronoi neighbours do not all meet across a proper face. Some meet almost at
 * a corner, and the seam between them comes out a few millimetres wide between
 * shards a metre across. Measured on a 47,631-chunk walled city: a quarter of
 * all seams under 5 cm, the narrowest at 1 mm. Those seams are what tore the
 * building apart under its own weight — they read as three times their limit
 * while their neighbours sat at five percent of theirs, cracked, and handed
 * their share on.
 *
 * ## Why it is the model, not the building
 *
 * The solver allocates force by solving equilibrium over bonds it treats as
 * equally stiff rigid links. Area does not enter that solve at all. It enters
 * afterwards, in `stress = force / area`. So a sliver is handed the force of a
 * full-sized joint and then divided by almost nothing.
 *
 * A real joint that thin is COMPLIANT. Stiffness scales with area, force
 * distributes in proportion to stiffness, and a hairline contact therefore
 * attracts almost no force and reports an unremarkable stress. The runaway is
 * an artefact of treating every bond as equally rigid.
 *
 * The right fix is an area-aware solve. Until then, the honest model of a
 * contact that would carry almost nothing is no contact: dropping it removes a
 * load path that was never really there and leaves the ones that were.
 *
 * ## Why the threshold is relative
 *
 * "Too thin" is a statement about the pieces being joined, not about an area.
 * Five centimetres is a proper joint between two pebbles and a hairline
 * between two wall blocks. The absolute floor already in the fracturer
 * (MIN_BOND_AREA, 80 cm^2) cannot tell those apart, which is why it did not
 * catch these.
 *
 * ## Where it runs
 *
 * After auto-bonding, not during fracture. The auto-bonder re-measures every
 * contact against NvBlast's own generator and adds ones the closed-form pass
 * missed, so it is the only point at which the areas are final. Culling before
 * it just gets the slivers added back.
 */

/**
 * Fraction of the smaller chunk's largest face below which a contact is not
 * structure.
 *
 * Measured on the walled city: 0.02 keeps 89% of bonds with nothing orphaned
 * and nothing ungrounded. The safe range is wide — 5% still orphans nothing,
 * 10% starts to — so this sits well inside it. 0 disables the cull.
 */
export const SLIVER_FRACTION = Number(process.env.BLAST_SLIVER_FRACTION ?? 0.02);

/**
 * Share of a chunk's bonded area that must survive the cull.
 *
 * A sliver is only negligible if the chunk has better paths. A chunk held up
 * by many small contacts is genuinely relying on them, and taking them all
 * leaves a model that is wrong in the other direction -- which is what
 * happened to the Petronas towers before this existed.
 */
const RETAIN_AREA = Number(process.env.BLAST_SLIVER_RETAIN ?? 0.85);

/** Bonds a chunk keeps no matter how small they are. */
const MIN_DEGREE = 2;

/** Largest face of a chunk's bounding box: its most it could ever touch through. */
function largestFace(size) {
  const dims = [Math.abs(size.x), Math.abs(size.y), Math.abs(size.z)].sort((a, b) => a - b);
  return dims[1] * dims[2];
}

/**
 * Remove sliver bonds from a built pack, in place.
 *
 * Nothing is ever left unbonded: a chunk about to lose its last bond keeps the
 * largest of the ones being taken from it. On the city that measured worst the
 * net is never needed, but a pack that fractures differently should not be
 * able to shed a chunk into space because a threshold moved.
 *
 * @returns {{dropped:number, kept:number, rescued:number}}
 */
export function cullSliverBonds(pack, { fraction = SLIVER_FRACTION } = {}) {
  const s = pack.scenario;
  const before = s.bonds.length;
  if (!(fraction > 0) || before === 0) return { dropped: 0, kept: before, rescued: 0 };

  const faces = s.nodeSizes.map(largestFace);
  const degree = new Array(s.nodes.length).fill(0);
  for (const b of s.bonds) {
    degree[b.node0] += 1;
    degree[b.node1] += 1;
  }

  const doomed = [];
  for (let i = 0; i < s.bonds.length; i += 1) {
    const b = s.bonds[i];
    const reference = Math.min(faces[b.node0], faces[b.node1]);
    if (reference > 0 && b.area < reference * fraction) doomed.push(i);
  }
  // Largest first, so a chunk that needs one back keeps its best.
  doomed.sort((x, y) => s.bonds[y].area - s.bonds[x].area);

  // How much bonded area each chunk has to begin with. A sliver is only
  // negligible if the chunk it belongs to has better paths; a chunk held up by
  // nothing BUT small contacts is relying on them, however thin they look.
  // Stripping those is what took the Petronas towers from standing to a joint
  // at 216% of yield doing nothing at all.
  const totalArea = new Array(s.nodes.length).fill(0);
  for (const b of s.bonds) {
    totalArea[b.node0] += b.area;
    totalArea[b.node1] += b.area;
  }
  const remainingArea = totalArea.slice();

  const dropped = new Set();
  let rescued = 0;
  for (const i of doomed) {
    const b = s.bonds[i];
    if (degree[b.node0] <= MIN_DEGREE || degree[b.node1] <= MIN_DEGREE) {
      rescued += 1;
      continue;
    }
    // Neither end may lose so much of its bonded area that what is left is no
    // longer a fair account of how it is held.
    const after0 = remainingArea[b.node0] - b.area;
    const after1 = remainingArea[b.node1] - b.area;
    if (after0 < totalArea[b.node0] * RETAIN_AREA || after1 < totalArea[b.node1] * RETAIN_AREA) {
      rescued += 1;
      continue;
    }
    degree[b.node0] -= 1;
    degree[b.node1] -= 1;
    remainingArea[b.node0] = after0;
    remainingArea[b.node1] = after1;
    dropped.add(i);
  }

  // Keeping every chunk bonded to SOMETHING is not enough: a chunk can hold on
  // to a neighbour that is itself now cut off from the ground, and a group
  // floating together is as wrong as a chunk floating alone. Villa Savoye lost
  // 2 that way and the Petronas towers 108, both caught by the grounding gate
  // rather than by the degree count above.
  //
  // So the real invariant is the one the verifier checks -- every chunk
  // reaches a support -- and it is restored the same way it was broken:
  // put back the largest dropped bond that reconnects something, until
  // everything is connected again.
  rescued += restoreGrounding(s, dropped);

  if (dropped.size > 0) {
    s.bonds = s.bonds.filter((_, i) => !dropped.has(i));
  }
  return { dropped: dropped.size, kept: s.bonds.length, rescued };
}

/**
 * Put back the fewest, largest dropped bonds needed for every chunk to reach a
 * support again.
 *
 * @returns how many bonds were restored
 */
function restoreGrounding(s, dropped) {
  const supports = [];
  for (let i = 0; i < s.nodes.length; i += 1) {
    if (s.nodes[i].mass === 0) supports.push(i);
  }
  if (supports.length === 0) return 0;

  const adjacency = () => {
    const adj = Array.from({ length: s.nodes.length }, () => []);
    for (let i = 0; i < s.bonds.length; i += 1) {
      if (dropped.has(i)) continue;
      adj[s.bonds[i].node0].push(s.bonds[i].node1);
      adj[s.bonds[i].node1].push(s.bonds[i].node0);
    }
    return adj;
  };
  const reachable = () => {
    const adj = adjacency();
    const seen = new Uint8Array(s.nodes.length);
    const stack = supports.slice();
    for (const node of supports) seen[node] = 1;
    while (stack.length) {
      const node = stack.pop();
      for (const next of adj[node]) {
        if (!seen[next]) { seen[next] = 1; stack.push(next); }
      }
    }
    return seen;
  };

  // Candidates largest first: reconnect with the strongest link available.
  const candidates = [...dropped].sort((x, y) => s.bonds[y].area - s.bonds[x].area);
  let restored = 0;
  let seen = reachable();
  let progress = true;
  while (progress) {
    progress = false;
    for (const i of candidates) {
      if (!dropped.has(i)) continue;
      const b = s.bonds[i];
      // Exactly one end already reaches the ground: this bond brings the other
      // end back with it.
      if (seen[b.node0] !== seen[b.node1]) {
        dropped.delete(i);
        restored += 1;
        seen = reachable();
        progress = true;
      }
    }
  }
  return restored;
}

/**
 * How many pieces a thing breaks into: a function of its MATERIAL and its
 * SURFACE SIZE.
 *
 * The shipped city packs fix shard counts per structural element class
 * (SHARDS_PER_PANEL, SHARDS_PER_SLAB, ...), which means a 2 m window and a
 * 30 m curtain wall shatter into the same number of pieces. Keying on the
 * fractured face's area instead makes a big pane produce more, smaller shards
 * and a small one produce a few large ones, which is what both brick and glass
 * actually do.
 *
 * Each material declares the area it wants per shard (`cellArea`) plus a floor
 * and ceiling. The ceiling is what keeps a 200 m^2 facade from becoming a
 * thousand bodies; the floor is what keeps a small piece from staying whole.
 */

/**
 * cellArea: m^2 of fractured face per shard.
 * min/max:  hard clamp on the resulting count.
 */
/**
 * The largest a single chunk may be, in m^3, per material.
 *
 * This is the cap that matters for how a collapse READS. `shardsFor` bounds the
 * shard COUNT, which is the wrong quantity on its own: clamping a 640 m^2 floor
 * plate to eight shards produces eight pieces of 80 m^3 each, and a building
 * toppling into eight intact slabs the size of tennis courts looks like one
 * solid object falling over rather than a building coming apart.
 *
 * Capping volume instead is also cheap, because the size distribution is very
 * skewed — the parking garage's slab chunks averaged 7.8 m^3 against a 98.9 m^3
 * maximum, so capping at 3.5 only splits the outliers.
 */
const MAX_CHUNK_VOLUME = {
  glass: 0.6,
  brick: 0.6,
  stone: 2.0,
  steel: 2.5,
  'wood-frame': 1.2,
  'reinforced-concrete': 3.5,
  'concrete-slab': 3.5,
  'facade-panel': 1.2,
  'facade-clip': 1.2,
  'glazing-clip': 0.6,
  // Foundations are underground and are supports; they never move, so their
  // size is invisible and splitting them buys nothing.
  'footing-anchor': 12.0,
};

export function maxChunkVolume(materialName) {
  const v = MAX_CHUNK_VOLUME[materialName];
  if (!v) throw new Error(`no chunk-volume cap for material "${materialName}"`);
  return v;
}

const RULES = {
  // Glass is clamped to 2-4 shards, by request: a window that becomes 30
  // shards reads as sand rather than glass. The cell size only decides where in
  // that range a pane lands -- a small one splits in two, a full-height lobby
  // pane in four -- and the clamp does the rest.
  glass: { cellArea: 2.5, min: 2, max: 4 },
  // Brick courses into many small pieces -- it is the finest of the solids.
  brick: { cellArea: 0.35, min: 3, max: 24 },
  // Stone is chunkier: bigger blocks, fewer of them.
  stone: { cellArea: 0.50, min: 3, max: 16 },
  // Steel is ductile (band 12): it bends and tears rather than shattering, so
  // it gets the fewest pieces of any solid despite being the strongest.
  steel: { cellArea: 1.20, min: 2, max: 8 },
  // The two structural concretes take a much larger cell than the masonry
  // above. A 0.6 m^2 cell suits a facade panel; applied to a 31 m^2 floor plate
  // it asks for fifty shards and the tower came out at 10k bodies, downtown
  // scale for one building. Big plates genuinely break into big pieces, and the
  // max clamp is what stops a large surface becoming a thousand of them.
  'reinforced-concrete': { cellArea: 2.00, min: 2, max: 10 },
  'concrete-slab': { cellArea: 2.50, min: 2, max: 8 },
  'wood-frame': { cellArea: 0.80, min: 2, max: 6 },
  'footing-anchor': { cellArea: 0.60, min: 2, max: 6 },
  'glazing-clip': { cellArea: 2.5, min: 2, max: 4 },
  // Cladding: panels, spandrels, parapets. Breaks into small pieces because
  // that is what a facade panel does when it comes off a building.
  'facade-panel': { cellArea: 0.45, min: 2, max: 12 },
  'facade-clip': { cellArea: 0.45, min: 2, max: 12 },
};

export function fractureRule(materialName) {
  const r = RULES[materialName];
  if (!r) throw new Error(`no fracture rule for material "${materialName}"`);
  return r;
}

/**
 * Shard count for a piece whose fractured face has area `faceArea` (m^2).
 *
 * `faceArea` is the area of the face the Voronoi cells are cut on -- the
 * LARGEST face, so a wall is cut across its width and height rather than
 * through its thickness.
 */
/**
 * Cell size for the subdivision grid, as a target VOLUME.
 *
 * Sized so a cell fractures into the material's maximum shard count and each
 * shard lands right on the volume cap. Picking the cell size independently of
 * the shard rule makes the two fight: an 8.75 m^3 cell asking for its rule's
 * eight shards produced 1.1 m^3 chunks against a 3.5 m^3 cap, and the parking
 * garage went from 700 chunks to 5,682 for no visible gain.
 */
export function cellVolumeFor(materialName, fractured = true) {
  const cap = maxChunkVolume(materialName);
  // An unfractured piece yields exactly one chunk per cell, so the cell IS the
  // cap. Sizing it as if the cell would be Voronoi-cut left 60 m^3 foundations.
  if (!fractured) return cap;
  // At most three shards to a cell. Relying on more makes the cell so large
  // that Voronoi's unevenness puts the biggest shard well over the cap even on
  // a jittered grid.
  return cap * Math.min(fractureRule(materialName).max, 3);
}

export function shardsFor(materialName, faceArea, cellVolume = 0) {
  const { cellArea, min, max } = fractureRule(materialName);
  const byArea = Math.round(faceArea / cellArea);
  // The volume cap is a floor on the count, not just a ceiling: a cell must be
  // cut into at least enough pieces that none of them exceeds it.
  // 1.4x headroom: even on a jittered grid the Voronoi cells vary, so aiming
  // exactly at the cap leaves the largest shard over it.
  const byVolume = cellVolume > 0
    ? Math.ceil((cellVolume * 1.4) / maxChunkVolume(materialName))
    : 0;
  return Math.max(min, byVolume, Math.min(max, byArea));
}

/** Human-readable histogram line, printed by build.mjs so an odd count is visible. */
export function shardHistogram(counts) {
  const byMaterial = new Map();
  for (const { material, shards } of counts) {
    if (!byMaterial.has(material)) byMaterial.set(material, []);
    byMaterial.get(material).push(shards);
  }
  const lines = [];
  for (const [name, list] of [...byMaterial].sort()) {
    list.sort((a, b) => a - b);
    const sum = list.reduce((s, n) => s + n, 0);
    lines.push(
      `  ${name.padEnd(21)} pieces=${String(list.length).padStart(5)}` +
      ` shards=${String(sum).padStart(6)}` +
      ` min=${list[0]} med=${list[list.length >> 1]} max=${list[list.length - 1]}`,
    );
  }
  return lines.join('\n');
}

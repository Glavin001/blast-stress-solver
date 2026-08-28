/**
 * Reading a pack's colliders, whether or not it uses a shape library.
 *
 * A pack that bounds its fracture patterns stores each distinct shard once in
 * `scenario.shapeLibrary` and points at it with `{kind:'shape', shape:N}`. The
 * Rust parser resolves those at load, so nothing downstream sees the
 * difference — but the JS validators read the pack directly and would
 * otherwise find a collider with no geometry on it.
 */

/** The collider for node `i`, with any shape reference resolved. */
export function colliderOf(scenario, i) {
  const c = scenario.nodeColliders[i];
  if (c.kind !== 'shape') return c;
  const entry = scenario.shapeLibrary?.[c.shape];
  if (!entry) throw new Error(`node ${i} references shape ${c.shape}, library has ${scenario.shapeLibrary?.length ?? 0}`);
  return entry;
}

/**
 * Replaces repeated hulls with references into a shape library.
 *
 * Only shapes used more than once move: a one-of-a-kind shard costs an extra
 * indirection and saves nothing, and the format says as much — `shape_id` is
 * `None` for packs whose shards are all unique.
 */
export function buildShapeLibrary(scenario) {
  const uses = new Map();
  const keys = new Array(scenario.nodeColliders.length);
  for (let i = 0; i < scenario.nodeColliders.length; i += 1) {
    const c = scenario.nodeColliders[i];
    if (c.kind !== 'convex_hull') continue;
    const key = c.points.join(',');
    keys[i] = key;
    uses.set(key, (uses.get(key) ?? 0) + 1);
  }
  const library = [];
  const slotOf = new Map();
  let shared = 0;
  for (let i = 0; i < scenario.nodeColliders.length; i += 1) {
    const key = keys[i];
    if (key === undefined || uses.get(key) < 2) continue;
    let slot = slotOf.get(key);
    if (slot === undefined) {
      slot = library.length;
      slotOf.set(key, slot);
      library.push(scenario.nodeColliders[i]);
    }
    scenario.nodeColliders[i] = { kind: 'shape', shape: slot };
    shared += 1;
  }
  if (library.length > 0) scenario.shapeLibrary = library;
  const hulls = keys.filter((k) => k !== undefined).length;
  return { hulls, distinct: uses.size, library: library.length, shared, inline: hulls - shared };
}

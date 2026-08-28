/**
 * Which fracturer cuts a piece into shards.
 *
 * There is one in use and one worth having, and they differ in a way that is
 * not a matter of taste — so the choice is a named, selectable thing rather
 * than an assumption baked into the builder.
 *
 * ## voronoi-2d (default, in use)
 *
 * Cuts a piece's flat cross-section into Voronoi cells by half-plane clipping
 * and extrudes each one through the piece's thickness. Its virtues are exactly
 * the three defects it was written to fix, when this project fractured with
 * three-pinata and inferred bonds by proximity:
 *
 *   - cells TILE the cross-section, so shards cannot interpenetrate — the old
 *     path measured 44 m^3 of solid inside solid, every pair a loaded spring;
 *   - the seam between two cells is their exact bisector, so bond area is
 *     closed-form geometry rather than a proximity heuristic with fudge
 *     factors;
 *   - one seeded RNG drives it, so identical panels shatter identically and
 *     collapse to one shape-library upload. three-pinata's RNG could not be
 *     seeded, so the same input gave different packs run to run.
 *
 * Its limitation is structural, not incidental: it is two-dimensional. Cuts
 * through a piece's DEPTH can only come from subdividing it into a grid of
 * cells and cutting each separately — and where two independently-cut cells
 * meet there is no bisector, only two patterns that happen to abut. Measured
 * on a 47,631-chunk masonry structure, that is where the hairline seams live:
 *
 *     piece cut into    1-4 shards (one cell)    0.6% of seams under 5 cm
 *     piece cut into   5-16 shards              15.2%
 *     piece cut into  17-64 shards              20.7%
 *     piece cut into    65+ shards              24.6%
 *
 * A seam a centimetre wide between two metre-wide shards carries ordinary load
 * through almost no area, so it reads as overloaded, cracks, and hands its
 * share to its neighbours. That is a building slowly failing under its own
 * weight, and no material change fixes it.
 *
 * ## nvblast-3d (not wired)
 *
 * NvBlast's own `FractureTool` cuts in three dimensions, so one diagram covers
 * a whole piece and every seam is a real bisector — the grid, and with it the
 * hairlines, stops being necessary. It also offers noisy fracture surfaces and
 * cutout fracture, neither of which we can express at all today.
 *
 * The two things that disqualified three-pinata do not apply to it:
 * `voronoiFracturing` accepts explicit cell points, so determinism stays ours;
 * and `bondsFromPrefractured` in EXACT mode gives true contact areas — we
 * already use it to audit this backend's bonds, and it agrees to 0.0% mean
 * error.
 *
 * What stands in the way is plumbing, and it is worth being exact about it,
 * because it is easy to assume this is closer than it is:
 *
 *   1. The shipped WASM exposes the stress solver's C ABI and the bond
 *      generator. It does NOT expose fracturing. `FractureTool` appears in the
 *      binary only as a type name inside the bond generator's symbols, which
 *      is not the same as being callable.
 *   2. So it needs new C ABI entry points around `FractureTool` /
 *      `VoronoiSitesGenerator`, in the shape of the existing `ext_stress_*`
 *      and `authoring_*` ones.
 *   3. And a WASM rebuild, which needs emscripten — not currently installed on
 *      this machine.
 *   4. Its chunks are general convex hulls rather than extruded prisms, so the
 *      collider path, the shape library and the interpenetration check all
 *      need to be looked at rather than assumed.
 *
 * None of that is hard; all of it is real. The seam exists so that work lands
 * as one backend rather than as a rewrite of the builder.
 */

/** Backend a piece uses when it asks for nothing in particular. */
export const DEFAULT_BACKEND = process.env.BLAST_FRACTURE_BACKEND ?? 'voronoi-2d';

export const FRACTURE_BACKENDS = ['voronoi-2d', 'nvblast-3d'];

/**
 * Resolve the backend for a piece.
 *
 * @param pieceBackend  per-piece override, or undefined for the default
 */
export function resolveBackend(pieceBackend) {
  const name = pieceBackend ?? DEFAULT_BACKEND;
  if (!FRACTURE_BACKENDS.includes(name)) {
    throw new Error(
      `unknown fracture backend "${name}" (have: ${FRACTURE_BACKENDS.join(', ')})`,
    );
  }
  return name;
}

/**
 * Fail with the specific reason rather than a missing-function error.
 *
 * A backend that is selectable but unimplemented should say what implementing
 * it involves, at the moment someone selects it.
 */
export function unavailable(name) {
  if (name !== 'nvblast-3d') {
    throw new Error(`fracture backend "${name}" is not implemented`);
  }
  throw new Error(
    'fracture backend "nvblast-3d" is not wired up yet.\n'
    + '  NvBlast\'s FractureTool cuts in 3D, which is what removes the hairline\n'
    + '  seams the 2D backend produces where its grid cells meet. To enable it:\n'
    + '    1. add C ABI entry points around FractureTool / VoronoiSitesGenerator\n'
    + '       (mirror the existing ext_stress_* and authoring_* exports),\n'
    + '    2. rebuild the WASM with emscripten (not installed here),\n'
    + '    3. check colliders, the shape library and the interpenetration test\n'
    + '       against general convex hulls rather than extruded prisms.\n'
    + '  Until then: BLAST_FRACTURE_BACKEND=voronoi-2d (the default).',
  );
}

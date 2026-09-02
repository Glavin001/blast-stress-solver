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
 * ## Measured: a 3D cutter does NOT fix this
 *
 * The obvious inference from that table is that the grid is the problem and a
 * 3D cut, needing no grid, would remove the hairlines. That inference is
 * wrong, and it is worth writing down so nobody spends a week on it.
 *
 * Cutting one 4 x 2.5 x 0.8 m wall into 12 shards, both fracturers, bond areas
 * measured by NvBlast's own EXACT auto-bonder so neither is scored by its own
 * arithmetic (experiments/fracture-backends.mjs):
 *
 *                          ours (2D+extrude)   three-pinata (3D voronoi)
 *     seam width p10             4.3 cm                1.0 cm
 *     seam width median         25.3 cm               16.7 cm
 *     seams under 5 cm            13%                   33%
 *     volume recovered           100%                  100%
 *     vertices per piece         36-60                72-225
 *
 * The 3D cutter produces MORE hairlines, not fewer -- which makes sense once
 * seen: cells in three dimensions have more neighbours than cells in two, and
 * every extra adjacency is another chance for two of them to meet at almost
 * nothing. Hairline seams are a property of Voronoi adjacency itself, not of
 * our grid.
 *
 * Both tile the volume exactly, so neither has an interpenetration problem.
 * But three-pinata's pieces carry 72-225 vertices against a 64-point collider
 * budget, so they would need hull simplification before they could be used at
 * all.
 *
 * The conclusion that follows: the fix for hairline seams is to handle sliver
 * CONTACTS explicitly -- merge them into a neighbour, or give a bond a
 * minimum effective area for stress -- rather than to change fracturer.
 *
 * ## nvblast-3d (not wired)
 *
 * NvBlast's own `FractureTool` cuts in three dimensions. On the evidence above
 * that will NOT by itself fix the hairlines -- a 3D cut has more adjacencies,
 * not fewer -- so it should not be sold as the cure for them. What it does
 * offer that nothing else here does is noisy fracture surfaces and cutout
 * fracture: shards with rough faces instead of flat planes, which is a real
 * fidelity gain and the actual reason to want it.
 *
 * Note that one of the two things that disqualified three-pinata no longer
 * applies to three-pinata either: v2 takes a `seed` and explicit Voronoi
 * `seedPoints`, and produces byte-identical fragments from the same seed
 * (verified, not taken from the docs). The determinism objection is history.
 * What rules it out now is the measurement above, which is a better reason.
 *
 * For NvBlast specifically:
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
 *   3. A build. Emscripten is present (/root/emsdk) if a WASM path is wanted,
 *      but the authoring library is C++ and this pipeline is offline Node, so
 *      a native binary called as a subprocess -- as the rest-report tool is --
 *      is the simpler target and needs no emscripten at all.
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
    + '  What it buys is noisy fracture surfaces and cutout fracture, NOT an end\n'
    + '  to hairline seams -- a 3D cut measured worse on those, see this file.\n'
    + '  To enable it:\n'
    + '    1. add C ABI entry points around FractureTool / VoronoiSitesGenerator\n'
    + '       (mirror the existing ext_stress_* and authoring_* exports),\n'
    + '    2. build it -- native subprocess is simpler than WASM for an offline\n'
    + '       pipeline, and needs no emscripten,\n'
    + '    3. check colliders, the shape library and the interpenetration test\n'
    + '       against general convex hulls rather than extruded prisms.\n'
    + '  Until then: BLAST_FRACTURE_BACKEND=voronoi-2d (the default).',
  );
}

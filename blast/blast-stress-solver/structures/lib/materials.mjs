/**
 * Material table for authored structures: strength AND appearance.
 *
 * Strength is the six stress limits the solver reads, in absolute pascals, with
 * the same meaning as `MATERIALS` in scripts/export-fractured-city.mjs. The
 * appearance fields (`color`, `opacity`, `textureKey`, `roughness`,
 * `metalness`) are additions. `scene_pack.rs` has no `deny_unknown_fields`, so
 * they ride along in the pack and the Rust parser ignores them; the standalone
 * viewer in vibe-land-2 reads them.
 *
 * DUCTILITY IS THE MAIN BEHAVIOURAL DIAL, and it is independent of raw
 * strength: it is the width of the fatal-elastic band, expressed here as
 * `band = fatal / elastic`. A wide band takes partial damage over many frames
 * (reinforced concrete yielding). A narrow band snaps in one, at the impact
 * site, which is both what glass does and what makes a hit readable.
 *
 * NEVER FAKE STRENGTH WITH BOND AREA. Bond area is simultaneously the
 * denominator of stress and the bond's damage pool, so inflating it corrupts
 * the stress readout and scales toughness super-linearly. Area is geometry;
 * material is strength; they are separate axes.
 */

/** kg/m^3. Applied at authoring time and baked into node mass, as upstream does. */
export const DENSITY = {
  glass: 2500,
  brick: 1900,
  stone: 2600,
  steel: 7850,
  'reinforced-concrete': 2400,
  'concrete-slab': 2400,
  'wood-frame': 600,
  'facade-panel': 2000,
  'facade-clip': 2000,
  'glazing-clip': 2500,
  'footing-anchor': 2400,
};

/**
 * `elastic` is the compression yield point; tension and shear are given as
 * fractions of it, following the ratios the shipped city pack uses (tension
 * ~1/8 of compression, shear ~1/7). `band` multiplies every elastic value to
 * get its fatal counterpart.
 *
 * A negative tension/shear would mean "inherit compression" to the parser; we
 * always write explicit numbers so the pack reads the same as it behaves.
 */
//
// ## Tension is a REINFORCED capacity
//
// These fractions were set when bending was folded into the axial stress and
// checked against the compression limit, so the tension numbers were never
// really exercised and drifted toward plain-concrete cracking stress (~3 MPa).
// Now that a bending moment correctly produces tension on one face, tension is
// the mode that governs every slab, beam and cantilever in the set — and the
// right figure for a JOINT IN A REINFORCED FRAME is not plain concrete's. What
// carries tension across a construction joint in a real building is the steel
// through it; that is the entire purpose of reinforcement. Modelling these
// joints at plain-concrete tension would model a building nobody would be
// allowed to occupy.
//
// Calibrated so an intact structure standing still sits near 0.4 of its
// elastic limit: enough headroom that redistributing load off a lost column
// does not immediately fail the survivors, little enough that losing a whole
// side of them does. Measured per rig by `measure_rest_utilisation`.
//
// ## The light structures needed raising too
//
// Bending is now scaled by a section modulus, and a section modulus grows as
// the joint gets SMALL: the amplification is 6/sqrt(area), so a 0.02 m^2 joint
// between a floor slab and a timber post feels bending forty times more
// strongly than it used to. That is the right direction physically -- a slender
// member is weak in bending -- but at the old figures it tipped the two houses
// into shedding bonds under their own weight. The masonry and timber elastic
// limits are therefore roughly doubled; they were conservative to begin with
// (the note on wood-frame below records the same argument being had once
// already) and the at-rest invariant is not negotiable.
const SPEC = [
  // name                  elastic  band  tensionFrac shearFrac
  // Band 3, not 10. The band is how far past yield a material goes before it
  // actually breaks, and 10 makes concrete behave like rubber: the Algedra
  // tower dropped on its side from 18 m broke 6% of its bonds and landed
  // essentially in one piece. Reinforced concrete is not that ductile — it
  // cracks. 3 keeps a frame that yields visibly before it fails while letting
  // an impact of that size do real damage.
  ['reinforced-concrete',   48e6,    3,    0.30,       0.40],
  ['concrete-slab',         28e6,    2.5,  0.25,       0.28],
  ['stone',                 34e6,    3,    0.09,       0.18],
  // Masonry, not the brick unit. A fired brick reaches 20-50 MPa on its own;
  // a wall of them bedded in mortar fails at a fraction of that, and fails in
  // tension at almost nothing. The stronger figure left a house able to be
  // dropped on its side from 18 m and settle with 19 pieces off it — stress
  // from an impact scales with the mass above it, so a light structure needs
  // genuinely weak materials or nothing touches it.
  ['brick',                 16e6,    2.5,  0.11,       0.22],
  ['steel',                180e6,   12,    1.00,       0.60],
  // Softwood loaded ALONG the grain, which is how a post works: ~30-45 MPa.
  // The 8 MPa figure that looks right for timber is compression ACROSS the
  // grain, and using it put a house post at 138% of yield standing still.
  ['wood-frame',            36e6,    2.5,  0.40,       0.28],
  // Glass: BRITTLE, not weak. Band 1.05 means ~5% from yield to failure, so a
  // struck pane lets go at once instead of draining a damage pool for six
  // seconds while nothing visibly moves — that narrow band is the whole
  // character, and it is independent of raw strength.
  //
  // The strength itself was 4 MPa, which was simply wrong: five times weaker
  // than the flimsiest material in any shipped pack, and the panes tore
  // themselves out of the building under their own settling transients before
  // anyone shot at anything. Glass is enormously strong in compression
  // (~1 GPa); what makes it fragile is that it will not yield.
  ['glass',                 60e6,    1.05, 0.05,       0.09],
  // The seam holding a pane in its frame: a structural gasket. Weaker than the
  // pane and equally brittle, so the window pops OUT of the opening rather than
  // tearing the frame with it — but still an order of magnitude stronger than
  // the 2 MPa it started at, which could not hold a pane up.
  ['glazing-clip',          14e6,    1.05, 0.20,       0.34],
  // ── cladding ────────────────────────────────────────────────────────────
  // Non-structural: spandrels, parapets, infill panels, the free facade. An
  // order of magnitude weaker than the frame behind it and deliberately
  // brittle (band 1.2), so a hit takes panels off without touching the
  // structure — which is what a facade does, and what makes a building
  // readable to shoot at.
  ['facade-panel',          12e6,    1.2,  0.20,       0.29],
  // The fixing between a panel and the frame. Weaker again, so a panel comes
  // AWAY whole before it cracks up, and cracks up when it lands.
  ['facade-clip',            8e6,    1.2,  0.26,       0.36],
  ['footing-anchor',        1.0e8,  10,    0.13,       0.50],
];

/** Appearance. `textureKey` resolves to a layer in the client's city texture array. */
const LOOK = {
  // Untextured, near-white, and deliberately so. This is the architectural
  // white concrete of the balcony bands and parapets — smooth rendered
  // formwork, which has no grain to show at any distance you look at a
  // building from. Every scanned concrete set available is a weathered grey or
  // beige that reads as a multi-storey car park, and Poly Haven's
  // "white_plaster" is a mid-tone tan.
  'reinforced-concrete': { color: '#e9e7e2', textureKey: 'white-concrete', roughness: 0.82, metalness: 0.0 },
  'concrete-slab':       { color: '#c9c6bf', textureKey: 'concrete-floor', roughness: 0.95, metalness: 0.0 },
  stone:                 { color: '#9b9287', textureKey: 'stone',   roughness: 0.92, metalness: 0.0 },
  brick:                 { color: '#9c5b45', textureKey: 'brick',   roughness: 0.90, metalness: 0.0 },
  // Untextured on purpose. The scanned metal set is very dark (mean linear
  // ~0.05), so a steel stilt rendered through it reads as a black block;
  // a plain metallic surface reads as architectural steel.
  steel:                 { color: '#c3ccd6', textureKey: 'metal',    roughness: 0.30, metalness: 0.85 },
  'wood-frame':          { color: '#8b5a2b', textureKey: null,      roughness: 0.85, metalness: 0.02 },
  // Blue, semi-transparent, simple -- per the brief. No transmission or
  // refraction: those cost a render pass and buy nothing at these distances.
  // 0.55, not 0.38. Below about a half the panes on the near and far facades
  // are both legible at once and the building reads as a transparent shell
  // with its frame floating inside — still obviously semi-transparent at
  // this value, just no longer see-through end to end.
  glass:                 { color: '#7fb6e0', opacity: 0.55, textureKey: null, roughness: 0.06, metalness: 0.0 },
  'glazing-clip':        { color: '#7fb6e0', opacity: 0.55, textureKey: null, roughness: 0.06, metalness: 0.0 },
  'facade-panel':        { color: '#e4e2dd', textureKey: 'white-concrete', roughness: 0.88, metalness: 0.0 },
  'facade-clip':         { color: '#e4e2dd', textureKey: 'white-concrete', roughness: 0.88, metalness: 0.0 },
  'footing-anchor':      { color: '#8d8a86', textureKey: 'concrete-wall', roughness: 0.95, metalness: 0.0 },
};

function entry(name, elastic, band, tensionFrac, shearFrac) {
  const t = elastic * tensionFrac;
  const s = elastic * shearFrac;
  return {
    name,
    compressionElastic: elastic, compressionFatal: elastic * band,
    tensionElastic: t, tensionFatal: t * band,
    shearElastic: s, shearFatal: s * band,
    ...LOOK[name],
    density: DENSITY[name],
  };
}

/** The material table, in pack order. Index into this is what `m` refers to. */
export const MATERIALS = SPEC.map((row) => entry(...row));

/** name -> index, the `m` a bond or node carries. */
export const MATERIAL_INDEX = Object.fromEntries(MATERIALS.map((m, i) => [m.name, i]));

export function material(name) {
  const m = MATERIALS[MATERIAL_INDEX[name]];
  if (!m) throw new Error(`unknown material "${name}" (have: ${MATERIALS.map((x) => x.name).join(', ')})`);
  return m;
}

/**
 * The material of the bond between two pieces.
 *
 * Glass touching anything else is held by a glazing clip, never by the glass or
 * the frame material -- that is what makes a struck pane leave its opening
 * cleanly. Otherwise the joint is only as good as its weaker side, measured by
 * fatal stress since that is what actually decides when it lets go.
 */
export function bondMaterialName(a, b) {
  if (a === b) return a;
  // Glass is held in its opening by a clip, never by the glass or the frame:
  // that is what makes a struck pane leave the building cleanly.
  if (a === 'glass' || b === 'glass') return 'glazing-clip';
  // Cladding is FIXED to the frame, not bonded to it. The fixing is the weakest
  // thing in the wall, so a panel detaches whole rather than tearing a piece of
  // structure out with it — and the structure behind is untouched.
  if (a === 'facade-panel' || b === 'facade-panel') return 'facade-clip';
  return material(a).compressionFatal <= material(b).compressionFatal ? a : b;
}

/**
 * The material table as the pack's `defaults.solver.materials`.
 *
 * `scene_pack.rs` rejects a v2 pack whose table is missing or empty, rejects a
 * negative compression value, and rejects `compressionFatal < compressionElastic`
 * ("a bond that breaks before it yields is not a weaker material, it is an
 * incoherent one"). Assert all three here so a bad edit fails at authoring time
 * rather than at game load.
 */
export function materialTable() {
  for (const m of MATERIALS) {
    if (!(m.compressionElastic >= 0)) throw new Error(`${m.name}: negative compressionElastic`);
    if (m.compressionFatal < m.compressionElastic) throw new Error(`${m.name}: fatal < elastic`);
  }
  return MATERIALS;
}

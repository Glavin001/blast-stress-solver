/**
 * Single-storey house — brick skin on a wood post-and-beam frame, gabled roof.
 *
 * The structural model is the one from src/scenarios/houseScenario.ts, which
 * exists because a uniform box floats when you knock its walls out:
 *
 *   static FOOTING -> floor SLAB -> wood POSTS -> top-plate ring BEAM and
 *   RIDGE beam -> ROOF resting on both.
 *
 * The brick is an outer SKIN and the frame stands INBOARD of it. Two things
 * follow, and both are the point of the arrangement: the skin carries only
 * itself, so blowing out a wall panel leaves the house standing; and the plate
 * is wide enough to bear on the skin and the posts together, so taking out a
 * post drops the bay above it.
 *
 * Nothing here may share space with anything else. A post run up INTO the plate
 * rather than up to its underside is not a stiffer joint — the contact finder
 * sees mutual containment, makes no bond at all, and the plate ends up floating.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { wallWithOpenings, roofSlope } from './lib/elements.mjs';

export const HOUSE_1 = {
  width: 10.0,        // X
  depth: 8.0,         // Z
  wallHeight: 2.7,    // floor to top plate
  wallThickness: 0.25,
  slabThickness: 0.2,
  footingDepth: 0.45,
  postSize: 0.18,
  plateDepth: 0.2,
  roofRise: 1.7,      // ridge above the top plate
  roofThickness: 0.14,
  eaveOverhang: 0.5,
};

/** @returns {{ pack, builder }} — the builder carries the shard statistics. */
export function buildHouse1(cfg = {}) {
  const C = { ...HOUSE_1, ...cfg };
  const b = new ScenePackBuilder({
    key: 'house_1story',
    title: 'House — single storey (brick skin on wood frame)',
    seed: 0x4011,
  });

  const hw = C.width / 2, hd = C.depth / 2;
  const slabTop = C.slabThickness;
  const eave = slabTop + C.wallHeight;      // top of the plate
  const wallTop = eave - C.plateDepth;      // underside of the plate: skin and posts stop here
  const ridge = eave + C.roofRise;
  const ps = C.postSize / 2;
  const T = C.wallThickness;
  // The plate spans the skin AND the frame behind it, so it bears on both.
  const plateW = T + C.postSize;
  const ridgeTop = ridge - C.roofThickness;
  // The ridge beam must seat UNDER the roof sheets, not inside them. The two
  // slopes meet at (0, ridgeTop), so their undersides drop away either side of
  // the apex; a beam of half-width `rbHalf` has to top out at the height the
  // underside has already fallen to by then, or it occupies the same space as
  // the roof and the contact finder makes no bond at all.
  const rbHalf = 0.09;
  const roofSlopePerX = (ridgeTop - (eave - ((ridgeTop - eave) * C.eaveOverhang) / hw)) / (hw + C.eaveOverhang);
  const ridgeBeamTop = ridgeTop - rbHalf * roofSlopePerX;
  const ridgeBottom = ridgeBeamTop - 0.22;

  // ── foundation: a perimeter footing, pinned to the world ─────────────────
  const F = 0.35;
  for (const [min, max] of [
    [[-hw - F, -C.footingDepth, -hd - F], [hw + F, 0, -hd + F]],
    [[-hw - F, -C.footingDepth, hd - F], [hw + F, 0, hd + F]],
    [[-hw - F, -C.footingDepth, -hd + F], [-hw + F, 0, hd - F]],
    [[hw - F, -C.footingDepth, -hd + F], [hw + F, 0, hd - F]],
  ]) b.box({ type: 'foundation', material: 'footing-anchor', min, max, fixed: true, fracture: false });

  // ── floor slab ───────────────────────────────────────────────────────────
  b.box({ type: 'slab', material: 'concrete-slab', min: [-hw, 0, -hd], max: [hw, slabTop, hd] });

  // ── posts, inboard of the skin ───────────────────────────────────────────
  const pX = hw - T - ps, pZ = hd - T - ps;
  const posts = [
    [-pX, -pZ], [0, -pZ], [pX, -pZ],
    [-pX, 0], [pX, 0],
    [-pX, pZ], [0, pZ], [pX, pZ],
  ];
  for (const [x, z] of posts) {
    b.box({ type: 'post', material: 'wood-frame',
      min: [x - ps, slabTop, z - ps], max: [x + ps, wallTop, z + ps] });
  }

  // ── top-plate ring beam: skin + post width, so it seats on both ──────────
  for (const [min, max] of [
    [[-hw, wallTop, -hd], [hw, eave, -hd + plateW]],
    [[-hw, wallTop, hd - plateW], [hw, eave, hd]],
    [[-hw, wallTop, -hd + plateW], [-hw + plateW, eave, hd - plateW]],
    [[hw - plateW, wallTop, -hd + plateW], [hw, eave, hd - plateW]],
  ]) b.box({ type: 'beam', material: 'wood-frame', min, max });

  // ── brick skin, with openings ────────────────────────────────────────────
  const sill = slabTop + 0.9, head = slabTop + 2.1;
  wallWithOpenings(b, {
    material: 'brick', axis: 'z', at: [-hd, -hd + T],
    u0: -hw, u1: hw, v0: slabTop, v1: wallTop,
    openings: [
      { u0: -3.6, u1: -2.0, v0: sill, v1: head },
      { u0: -0.55, u1: 0.55, v0: slabTop, v1: slabTop + 2.1 },   // door
      { u0: 2.0, u1: 3.6, v0: sill, v1: head },
    ],
  });
  wallWithOpenings(b, {
    material: 'brick', axis: 'z', at: [hd - T, hd],
    u0: -hw, u1: hw, v0: slabTop, v1: wallTop,
    openings: [
      { u0: -3.7, u1: -2.1, v0: sill, v1: head },
      { u0: -0.8, u1: 0.8, v0: sill, v1: head },
      { u0: 2.1, u1: 3.7, v0: sill, v1: head },
    ],
  });
  for (const [x0, x1] of [[-hw, -hw + T], [hw - T, hw]]) {
    wallWithOpenings(b, {
      material: 'brick', axis: 'x', at: [x0, x1],
      u0: -hd + T, u1: hd - T, v0: slabTop, v1: wallTop,
      openings: [{ u0: -1.9, u1: -0.5, v0: sill, v1: head }, { u0: 0.5, u1: 1.9, v0: sill, v1: head }],
    });
  }

  // ── gables: wall-top to ridge, flat-topped to seat the ridge beam ────────
  // A pure triangle would meet the beam at a point, which is no bearing at all.
  for (const [z0, z1] of [[-hd, -hd + T], [hd - T, hd]]) {
    b.piece({
      type: 'wall', material: 'brick', axis: 'z', lo: z0, hi: z1,
      poly: [[-hw, eave], [hw, eave], [rbHalf, ridgeBottom], [-rbHalf, ridgeBottom]],
    });
  }

  // ── ridge beam, seated on the two gables ─────────────────────────────────
  b.box({ type: 'beam', material: 'wood-frame',
    min: [-rbHalf, ridgeBottom, -hd], max: [rbHalf, ridgeBeamTop, hd] });

  // ── roof: two slopes, each a prism whose cross-section carries the pitch ──
  // The underside passes exactly through (hw, eave) — the outer top corner of
  // the plate — so the sheet lands ON the plate instead of through it, and only
  // the overhang hangs below that line.
  const eaveY = eave - ((ridgeTop - eave) * C.eaveOverhang) / hw;
  for (const s of [-1, 1]) {
    roofSlope(b, {
      material: 'wood-frame', ridgeAxis: 'z',
      lo: -hd - C.eaveOverhang, hi: hd + C.eaveOverhang,
      eaveU: s * (hw + C.eaveOverhang), eaveV: eaveY, ridgeU: 0, ridgeV: ridgeTop,
      thickness: C.roofThickness,
    });
  }

  b.build();
  return { pack: b.emit({ cameraTarget: [0, ridge * 0.45, 0], cameraDistance: 26 }), builder: b };
}

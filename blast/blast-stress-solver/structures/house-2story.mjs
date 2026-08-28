/**
 * Two-storey house — stone base course, brick above, wood frame and roof.
 *
 * Same structural idea as the single-storey house: masonry skin outside, wood
 * post-and-beam frame inboard of it, plate wide enough to bear on both. What
 * the second storey adds is an intermediate concrete slab, which is the piece
 * that makes the load path interesting — everything upstairs reaches the
 * ground only through that slab and the walls under it.
 *
 * The stone base course is not decoration. Stone is three times brick's
 * compressive strength here and the base carries twice the load, so putting the
 * stronger material at the bottom is the same reasoning a real mason uses.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { wallWithOpenings, roofSlope, staircase, slabWithOpening } from './lib/elements.mjs';

export const HOUSE_2 = {
  width: 10.0,
  depth: 8.0,
  storeyHeight: 2.7,
  wallThickness: 0.3,
  slabThickness: 0.22,
  footingDepth: 0.5,
  postSize: 0.2,
  plateDepth: 0.22,
  roofRise: 1.9,
  roofThickness: 0.15,
  eaveOverhang: 0.55,
  balconyDepth: 1.3,
  railHeight: 1.0,
};

/** @returns {{ pack, builder }} — the builder carries the shard statistics. */
export function buildHouse2(cfg = {}) {
  const C = { ...HOUSE_2, ...cfg };
  const b = new ScenePackBuilder({
    key: 'house_2story',
    title: 'House — two storeys (stone base, brick upper, wood frame)',
    seed: 0x4022,
  });

  const hw = C.width / 2, hd = C.depth / 2;
  const T = C.wallThickness;
  const ground = C.slabThickness;                     // top of the ground slab
  const upperBot = ground + C.storeyHeight;           // underside of the mid slab
  const upper = upperBot + C.slabThickness;           // top of the mid slab
  const eave = upper + C.storeyHeight;                // top of the plate
  const wallTop = eave - C.plateDepth;
  const ridge = eave + C.roofRise;
  const ridgeTop = ridge - C.roofThickness;
  const ps = C.postSize / 2;
  const plateW = T + C.postSize;

  const rbHalf = 0.1;
  const eaveY = eave - ((ridgeTop - eave) * C.eaveOverhang) / hw;
  const roofSlopePerX = (ridgeTop - eaveY) / (hw + C.eaveOverhang);
  const ridgeBeamTop = ridgeTop - rbHalf * roofSlopePerX;
  const ridgeBottom = ridgeBeamTop - 0.24;

  // ── foundation ───────────────────────────────────────────────────────────
  const F = 0.4;
  for (const [min, max] of [
    [[-hw - F, -C.footingDepth, -hd - F], [hw + F, 0, -hd + F]],
    [[-hw - F, -C.footingDepth, hd - F], [hw + F, 0, hd + F]],
    [[-hw - F, -C.footingDepth, -hd + F], [-hw + F, 0, hd - F]],
    [[hw - F, -C.footingDepth, -hd + F], [hw + F, 0, hd - F]],
  ]) b.box({ type: 'foundation', material: 'footing-anchor', min, max, fixed: true, fracture: false });

  // ── slabs: ground and intermediate ───────────────────────────────────────
  b.box({ type: 'slab', material: 'concrete-slab', min: [-hw, 0, -hd], max: [hw, ground, hd] });
  // The intermediate floor is a plate with a hole in it, for the stair. A
  // plate with a hole is concave and colliders here are convex, so it arrives
  // as four panels around the void.
  const stair = staircase(b, {
    at: [-hw + 0.5, -1.25], y0: ground, y1: upper, axis: 'x', material: 'wood-frame',
  });
  slabWithOpening(b, {
    min: [-hw, upperBot, -hd], max: [hw, upper, hd], opening: stair.void,
  });

  // ── posts, both storeys, inboard of the skin ─────────────────────────────
  const pX = hw - T - ps, pZ = hd - T - ps;
  const plan = [
    [-pX, -pZ], [0, -pZ], [pX, -pZ],
    [-pX, 0], [pX, 0],
    [-pX, pZ], [0, pZ], [pX, pZ],
  ];
  for (const [x, z] of plan) {
    b.box({ type: 'post', material: 'wood-frame', min: [x - ps, ground, z - ps], max: [x + ps, upperBot, z + ps] });
    b.box({ type: 'post', material: 'wood-frame', min: [x - ps, upper, z - ps], max: [x + ps, wallTop, z + ps] });
  }

  // ── top-plate ring beam ──────────────────────────────────────────────────
  for (const [min, max] of [
    [[-hw, wallTop, -hd], [hw, eave, -hd + plateW]],
    [[-hw, wallTop, hd - plateW], [hw, eave, hd]],
    [[-hw, wallTop, -hd + plateW], [-hw + plateW, eave, hd - plateW]],
    [[hw - plateW, wallTop, -hd + plateW], [hw, eave, hd - plateW]],
  ]) b.box({ type: 'beam', material: 'wood-frame', min, max });

  // ── masonry skin: stone downstairs, brick upstairs ───────────────────────
  const storey = (matName, v0, v1, openings) => {
    wallWithOpenings(b, { material: matName, axis: 'z', at: [-hd, -hd + T],
      u0: -hw, u1: hw, v0, v1, openings: openings.front });
    wallWithOpenings(b, { material: matName, axis: 'z', at: [hd - T, hd],
      u0: -hw, u1: hw, v0, v1, openings: openings.back });
    for (const [x0, x1] of [[-hw, -hw + T], [hw - T, hw]]) {
      wallWithOpenings(b, { material: matName, axis: 'x', at: [x0, x1],
        u0: -hd + T, u1: hd - T, v0, v1, openings: openings.side });
    }
  };
  const gSill = ground + 0.95, gHead = ground + 2.15;
  storey('stone', ground, upperBot, {
    front: [
      { u0: -3.7, u1: -2.1, v0: gSill, v1: gHead },
      { u0: -0.6, u1: 0.6, v0: ground, v1: ground + 2.15 },      // door
      { u0: 2.1, u1: 3.7, v0: gSill, v1: gHead },
    ],
    back: [{ u0: -3.6, u1: -1.6, v0: gSill, v1: gHead }, { u0: 1.6, u1: 3.6, v0: gSill, v1: gHead }],
    side: [{ u0: -1.7, u1: -0.4, v0: gSill, v1: gHead }, { u0: 0.4, u1: 1.7, v0: gSill, v1: gHead }],
  });
  const uSill = upper + 0.85, uHead = upper + 2.05;
  storey('brick', upper, wallTop, {
    // The middle opening on the front is the balcony door: full height.
    front: [
      { u0: -3.7, u1: -2.1, v0: uSill, v1: uHead },
      { u0: -0.8, u1: 0.8, v0: upper, v1: uHead },
      { u0: 2.1, u1: 3.7, v0: uSill, v1: uHead },
    ],
    back: [{ u0: -3.6, u1: -1.6, v0: uSill, v1: uHead }, { u0: 1.6, u1: 3.6, v0: uSill, v1: uHead }],
    side: [{ u0: -1.7, u1: -0.4, v0: uSill, v1: uHead }, { u0: 0.4, u1: 1.7, v0: uSill, v1: uHead }],
  });

  // ── gables ───────────────────────────────────────────────────────────────
  for (const [z0, z1] of [[-hd, -hd + T], [hd - T, hd]]) {
    b.piece({ type: 'wall', material: 'brick', axis: 'z', lo: z0, hi: z1,
      poly: [[-hw, eave], [hw, eave], [rbHalf, ridgeBottom], [-rbHalf, ridgeBottom]] });
  }

  // ── ridge beam and roof ──────────────────────────────────────────────────
  b.box({ type: 'beam', material: 'wood-frame', min: [-rbHalf, ridgeBottom, -hd], max: [rbHalf, ridgeBeamTop, hd] });
  for (const s of [-1, 1]) {
    roofSlope(b, {
      material: 'wood-frame', ridgeAxis: 'z',
      lo: -hd - C.eaveOverhang, hi: hd + C.eaveOverhang,
      eaveU: s * (hw + C.eaveOverhang), eaveV: eaveY, ridgeU: 0, ridgeV: ridgeTop,
      thickness: C.roofThickness,
    });
  }

  // ── first-floor balcony: concrete deck cantilevered off the mid slab, with
  //    a steel rail. Steel is ductile (band 12), so the rail bends rather than
  //    shattering when something hits it — the visual opposite of the glass.
  const bz0 = -hd - C.balconyDepth, bz1 = -hd;
  b.box({ type: 'balcony', material: 'concrete-slab', min: [-2.2, upperBot, bz0], max: [2.2, upper, bz1] });
  const R = 0.05;
  b.box({ type: 'rail', material: 'steel', min: [-2.2, upper, bz0], max: [2.2, upper + C.railHeight, bz0 + 2 * R] });
  for (const x of [-2.2, 2.2 - 2 * R]) {
    b.box({ type: 'rail', material: 'steel', min: [x, upper, bz0 + 2 * R], max: [x + 2 * R, upper + C.railHeight, bz1] });
  }

  b.build();
  return { pack: b.emit({ cameraTarget: [0, ridge * 0.45, 0], cameraDistance: 30 }), builder: b };
}

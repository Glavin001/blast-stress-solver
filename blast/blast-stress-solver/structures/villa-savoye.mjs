/**
 * Villa Savoye — Le Corbusier, Poissy, 1931.
 *
 * The building that states the "five points": pilotis lifting the box clear of
 * the ground, a free facade, ribbon windows, a free plan, and a roof garden.
 * All five are structural claims as much as stylistic ones, and the first is
 * the reason this is worth having here: the whole house stands on twelve
 * slender columns, so knocking one out drops a corner of the box. Nothing else
 * in the set makes a load path that legible.
 *
 * The ground floor is glazed and set BACK from the column line, so the box
 * visibly floats rather than resting on walls — which also means the glazing
 * carries nothing, exactly as with the tower's curtain wall.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { wallWithOpenings, staircase, slabWithOpening } from './lib/elements.mjs';

export const SAVOYE = {
  width: 21.0,          // X
  depth: 19.0,          // Z
  groundHeight: 3.6,    // underside of the box, on pilotis
  storeyHeight: 3.0,
  slabThickness: 0.28,
  // The real pilotis are slenderer than this. 0.34 left the column-to-slab
  // joint at 75% of yield at rest, and the tower taught that anything near
  // that mark comes apart once a converging solver has it.
  pilotiDiameter: 0.44,
  pilotiBays: 4,        // per side
  groundSetback: 3.4,   // how far the glazed ground floor sits inside the columns
  wallThickness: 0.24,
  ribbonSill: 1.05,     // ribbon window band, off the floor
  ribbonHead: 2.25,
  parapetHeight: 1.05,
  screenRadius: 3.6,    // the curved roof solarium screens
  footingDepth: 0.9,
};

/** Regular octagon inscribed in a circle — a round column the format can hold. */
function octagon(cx, cz, d) {
  const r = d / 2;
  return Array.from({ length: 8 }, (_, i) => {
    const a = (Math.PI / 4) * i + Math.PI / 8;
    return [cx + r * Math.cos(a), cz + r * Math.sin(a)];
  });
}

export function buildVillaSavoye(cfg = {}) {
  const C = { ...SAVOYE, ...cfg };
  const b = new ScenePackBuilder({
    key: 'villa_savoye',
    title: 'Villa Savoye — pilotis, ribbon windows, roof solarium',
    seed: 0x5a04,
  });

  const hw = C.width / 2, hd = C.depth / 2;
  const T = C.wallThickness;
  const groundTop = 0.3;                       // the thin ground slab under the pilotis
  const floorBase = groundTop + C.groundHeight;
  const floorTop = floorBase + C.slabThickness;
  const roofBase = floorTop + C.storeyHeight;
  const roofTop = roofBase + C.slabThickness;

  // Column lines: an even grid, which is what a free plan needs.
  const lineX = Array.from({ length: C.pilotiBays + 1 }, (_, i) => -hw + (2 * hw * i) / C.pilotiBays);
  const lineZ = Array.from({ length: C.pilotiBays + 1 }, (_, i) => -hd + (2 * hd * i) / C.pilotiBays);
  // Only the perimeter plus one interior line carries; a free plan is exactly
  // the absence of walls doing it.
  const columns = [];
  for (const x of lineX) for (const z of lineZ) {
    const perimeter = x === lineX[0] || x === lineX[lineX.length - 1]
      || z === lineZ[0] || z === lineZ[lineZ.length - 1];
    if (perimeter || (Math.abs(x) < 1e-6 && Math.abs(z) < 1e-6)) columns.push([x, z]);
  }

  // ── footings and ground slab ─────────────────────────────────────────────
  for (const [x, z] of columns) {
    b.box({ type: 'foundation', material: 'footing-anchor',
      min: [x - 0.45, -C.footingDepth, z - 0.45], max: [x + 0.45, 0, z + 0.45],
      fixed: true, fracture: false });
  }
  // reinforced-concrete, not concrete-slab. Twelve pilotis land on this and
  // nothing else; at a floor plate's 12 MPa the house stood at 177% of yield
  // doing nothing. A ground slab that carries columns is a raft, and a raft is
  // reinforced.
  b.box({ type: 'slab', material: 'reinforced-concrete', min: [-hw, 0, -hd], max: [hw, groundTop, hd] });

  // ── pilotis ──────────────────────────────────────────────────────────────
  for (const [x, z] of columns) {
    b.piece({
      type: 'column', material: 'reinforced-concrete', axis: 'y',
      poly: octagon(x, z, C.pilotiDiameter), lo: groundTop, hi: floorBase,
    });
  }

  // ── glazed ground floor, set back inside the columns ─────────────────────
  const gx = hw - C.groundSetback, gz = hd - C.groundSetback;
  const GLASS = 0.05;
  for (const [axis, sign] of [['z', 1], ['z', -1], ['x', 1], ['x', -1]]) {
    const along = axis === 'z' ? gx : gz - 2 * GLASS;
    const at = sign * (axis === 'z' ? gz : gx);
    const panes = Math.max(2, Math.round((2 * along) / 2.6));
    for (let i = 0; i < panes; i += 1) {
      const a0 = -along + (2 * along * i) / panes, a1 = -along + (2 * along * (i + 1)) / panes;
      b.piece({
        type: 'glazing', material: 'glass', axis,
        lo: at - GLASS / 2, hi: at + GLASS / 2,
        poly: axis === 'z'
          ? [[a0, groundTop], [a1, groundTop], [a1, floorBase - 0.02], [a0, floorBase - 0.02]]
          : [[groundTop, a0], [groundTop, a1], [floorBase - 0.02, a1], [floorBase - 0.02, a0]],
      });
    }
  }

  // ── the box: first-floor slab, with the stair void ───────────────────────
  const stair = staircase(b, {
    // Inside the glazed ground floor and clear of the centre piloti.
    at: [-6.4, -1.3], y0: groundTop, y1: floorTop, axis: 'x',
    material: 'reinforced-concrete',
  });
  // Reinforced throughout, which is both what the building actually is — it is
  // famous for being an early reinforced-concrete house — and what the load
  // path needs: twelve slender pilotis land on these plates, and a floor
  // plate's 12 MPa put the joint at over twice yield standing still.
  slabWithOpening(b, {
    material: 'reinforced-concrete',
    min: [-hw, floorBase, -hd], max: [hw, floorTop, hd], opening: stair.void,
  });

  // ── ribbon windows: one continuous band, all four sides ──────────────────
  // The point of a ribbon window is that it runs THROUGH the corner, which a
  // load-bearing wall cannot do. Here it is a run of panes with slim mullions,
  // and the wall above and below is the free facade carrying only itself.
  const sill = floorTop + C.ribbonSill, head = floorTop + C.ribbonHead;
  const ribbon = (span) => {
    const bays = Math.max(3, Math.round((2 * span) / 2.2));
    const mullion = 0.14;
    return Array.from({ length: bays }, (_, i) => {
      const a0 = -span + (2 * span * i) / bays, a1 = -span + (2 * span * (i + 1)) / bays;
      return { u0: a0 + mullion / 2, u1: a1 - mullion / 2, v0: sill, v1: head };
    });
  };
  // Cladding. The "free facade" is Le Corbusier's second point and it means
  // exactly this: the pilotis carry the building, so the wall carries nothing.
  for (const [z0, z1] of [[-hd, -hd + T], [hd - T, hd]]) {
    wallWithOpenings(b, {
      material: 'facade-panel', axis: 'z', at: [z0, z1],
      u0: -hw, u1: hw, v0: floorTop, v1: roofBase, openings: ribbon(hw - 0.3),
    });
  }
  for (const [x0, x1] of [[-hw, -hw + T], [hw - T, hw]]) {
    wallWithOpenings(b, {
      material: 'facade-panel', axis: 'x', at: [x0, x1],
      u0: -hd + T, u1: hd - T, v0: floorTop, v1: roofBase, openings: ribbon(hd - T - 0.3),
    });
  }

  // ── roof terrace and the curved solarium screens ─────────────────────────
  b.box({ type: 'slab', material: 'reinforced-concrete', min: [-hw, roofBase, -hd], max: [hw, roofTop, hd] });
  for (const [min, max] of [
    [[-hw, roofTop, -hd], [hw, roofTop + C.parapetHeight, -hd + 0.18]],
    [[-hw, roofTop, hd - 0.18], [hw, roofTop + C.parapetHeight, hd]],
    [[-hw, roofTop, -hd + 0.18], [-hw + 0.18, roofTop + C.parapetHeight, hd - 0.18]],
    [[hw - 0.18, roofTop, -hd + 0.18], [hw, roofTop + C.parapetHeight, hd - 0.18]],
  ]) b.box({ type: 'parapet', material: 'facade-panel', min, max });

  // The solarium screens: two quarter-round walls, which is the one curve in
  // the building and the reason its roof reads as a deck rather than a lid.
  // Faceted, because the format has no curved primitive and no rotation.
  const SEG = 7, SCREEN_T = 0.2, screenTop = roofTop + 2.3;
  for (const [cx, cz, a0] of [[hw - C.screenRadius, -hd + C.screenRadius, Math.PI / 2],
                              [hw - C.screenRadius, hd - C.screenRadius, Math.PI]]) {
    for (let i = 0; i < SEG; i += 1) {
      const t0 = a0 + (Math.PI / 2) * (i / SEG), t1 = a0 + (Math.PI / 2) * ((i + 1) / SEG);
      const r0 = C.screenRadius, r1 = C.screenRadius - SCREEN_T;
      b.piece({
        type: 'parapet', material: 'reinforced-concrete', axis: 'y',
        lo: roofTop, hi: screenTop,
        poly: [
          [cx + r1 * Math.cos(t0), cz + r1 * Math.sin(t0)],
          [cx + r0 * Math.cos(t0), cz + r0 * Math.sin(t0)],
          [cx + r0 * Math.cos(t1), cz + r0 * Math.sin(t1)],
          [cx + r1 * Math.cos(t1), cz + r1 * Math.sin(t1)],
        ],
      });
    }
  }

  b.build();
  return { pack: b.emit({ cameraTarget: [0, roofTop * 0.5, 0], cameraDistance: 52 }), builder: b };
}

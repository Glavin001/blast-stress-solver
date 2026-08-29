/**
 * A multi-storey parking garage.
 *
 * Not a landmark, and that is the point of including it. Everything else in
 * this set either topples (432 Park, the Petronas towers) or shrugs off local
 * damage (the houses). A garage is the one structure that PANCAKES: flat plates
 * on a sparse column grid, no walls, no core, nothing to redistribute through.
 * Take out a column on the ground floor and the plate above has nowhere to
 * hand the load, so it comes down onto the plate below, which was never
 * designed to catch it. That failure mode exists nowhere else here.
 *
 * The geometry is honest about it: wide bays, thin plates, no infill, and a
 * perimeter upstand that is a barrier rather than a wall.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { staircase, slabWithOpening, roofSlope } from './lib/elements.mjs';

export const GARAGE = {
  width: 46.0,          // X
  depth: 32.0,          // Z
  levels: 5,
  levelHeight: 3.1,
  slabThickness: 0.3,
  columnSize: 0.55,
  bayX: 7.6,
  bayZ: 8.0,
  // Drop panels: a thickened pad of slab around each column head.
  //
  // This is how a real flat plate survives punching shear, and it is the
  // detail this garage was missing. The check that sizes it, at OUR gravity
  // rather than Earth's:
  //
  //   self weight   w = rho * t * g = 2400 * 0.30 * 20 = 14.4 kPa
  //   at one column V = w * bayX * bayZ = 876 kN per level
  //   plain slab    resisting area u*d = 4*(0.55+0.27)*0.27 = 0.89 m^2
  //                 demand 0.99 MPa against 0.35*sqrt(f'c) = 2.07 MPa
  //
  // That passes intact. It does NOT pass once neighbouring columns are gone
  // and the tributary area multiplies, which is exactly when a real garage
  // punches -- Pipers Row, 1997. A drop panel raises the effective depth
  // locally, which raises both the perimeter and the resisting area, so the
  // connection survives its design load with margin and still punches when
  // asked to carry several bays.
  // 1.0 m, not the 2.4 m a drawing would show, and the reason is a limit of
  // the model rather than of the building. Real punching capacity grows with
  // the PERIMETER at the panel edge: a 2.4 m panel takes u*d from 0.89 to
  // 2.9 m^2, about 3.2x. Here the joint's capacity is its bond AREA, so a
  // 2.4 m panel is 5.76 m^2 against the column's 0.30 -- nineteen times, six
  // times more credit than the real detail earns. At that size nothing
  // punches at all: 90% of the columns can go and the deck does not move.
  //
  // Sized to deliver the real 3.2x instead. At 1.0 m the pad reads as a
  // column capital rather than a drop panel, which is the same detail at the
  // other end of its size range and honest about what it is doing.
  dropPanelSize: 2.4,      // plan size of the thickened pad
  dropPanelThickness: 0.2, // extra depth below the slab soffit
  spandrelHeight: 1.0,
  spandrelThickness: 0.22,
  rampWidth: 6.5,
  footingDepth: 1.2,
};

export function buildParkingGarage(cfg = {}) {
  const C = { ...GARAGE, ...cfg };
  const b = new ScenePackBuilder({
    key: 'parking_garage',
    title: 'Parking garage — flat plates on a sparse grid',
    seed: 0x9a4a,
  });

  const hw = C.width / 2, hd = C.depth / 2;
  const cs = C.columnSize / 2;
  const deckTop = (k) => k * C.levelHeight;      // level 0 is the ground slab
  const deckBase = (k) => deckTop(k) - C.slabThickness;

  const nx = Math.round(C.width / C.bayX), nz = Math.round(C.depth / C.bayZ);
  // Column lines pulled inside the perimeter, so the outer ones do not stand in
  // the same space as the edge barrier.
  // Inside the barrier, not merely inside the building line: clamping to the
  // edge still left every outer column standing in the upstand.
  const edge = C.spandrelThickness + cs;
  const clampX = (v) => Math.max(-hw + edge, Math.min(hw - edge, v));
  const clampZ = (v) => Math.max(-hd + edge, Math.min(hd - edge, v));
  const lineX = Array.from({ length: nx + 1 }, (_, i) => clampX(-hw + (2 * hw * i) / nx));
  const lineZ = Array.from({ length: nz + 1 }, (_, i) => clampZ(-hd + (2 * hd * i) / nz));

  // The ramp occupies the last bay in X; the plates there are the ramp instead.
  const rampX0 = lineX[nx - 1];

  // ── footings and ground slab ─────────────────────────────────────────────
  for (const x of lineX) {
    for (const z of lineZ) {
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [x - 0.7, -C.footingDepth, z - 0.7], max: [x + 0.7, 0, z + 0.7],
        fixed: true, fracture: false });
    }
  }
  b.box({ type: 'slab', material: 'reinforced-concrete',
    min: [-hw, 0, -hd], max: [hw, C.slabThickness, hd] });

  // The stair, open — a garage stair is a bare concrete flight, not a core.
  const stair = staircase(b, {
    at: [-hw + 0.8, -hd + 0.8], y0: C.slabThickness, y1: deckTop(1), axis: 'z',
    material: 'reinforced-concrete',
  });

  for (let k = 1; k <= C.levels; k += 1) {
    // ── columns for this level ─────────────────────────────────────────────
    for (const x of lineX) {
      for (const z of lineZ) {
        // The ramp bay is left clear between its two landings. A column there
        // stands in the path of the slope and the ramp drives straight through
        // it — which is also why real garages keep that bay open.
        const inRampRun = x > rampX0 - 0.01
          && z > -hd + C.rampWidth && z < hd - C.rampWidth;
        if (inRampRun) continue;
        const lo = k === 1 ? C.slabThickness : deckTop(k - 1);
        const dpT = C.dropPanelThickness;
        // The column stops at the drop panel's soffit; the panel carries the
        // last 200 mm up to the deck. Running it to deckBase instead puts the
        // two inside each other -- 725 shared-volume pairs, caught by the gate.
        b.box({ type: 'column', material: 'reinforced-concrete',
          min: [x - cs, lo, z - cs], max: [x + cs, deckBase(k) - dpT, z + cs] });
        // The drop panel sits directly under the deck this column supports,
        // spreading the punching perimeter before the load reaches the plate.
        const dp = C.dropPanelSize / 2;
        b.box({ type: 'slab', material: 'reinforced-concrete',
          min: [x - dp, deckBase(k) - C.dropPanelThickness, z - dp],
          max: [x + dp, deckBase(k), z + dp] });
      }
    }

    // ── the deck, stopping short of the ramp bay ───────────────────────────
    slabWithOpening(b, {
      material: 'reinforced-concrete',
      min: [-hw, deckBase(k), -hd], max: [rampX0, deckTop(k), hd],
      opening: stair.void,
    });
    // A landing at BOTH ends of the ramp bay. With only one, the ramp climbed
    // to the far side of the building and arrived at nothing — thirty floating
    // pieces per level.
    for (const [z0, z1] of [[-hd, -hd + C.rampWidth], [hd - C.rampWidth, hd]]) {
      b.box({ type: 'slab', material: 'reinforced-concrete',
        min: [rampX0, deckBase(k), z0], max: [hw, deckTop(k), z1] });
    }

    // ── the ramp up to the next level ──────────────────────────────────────
    // A prism whose cross-section carries the slope, exactly like a roof pitch:
    // the format has no rotation, so a ramp cannot be a tilted box.
    if (k < C.levels) {
      roofSlope(b, {
        type: 'ramp', material: 'reinforced-concrete', ridgeAxis: 'x',
        lo: rampX0, hi: hw,
        // Offset down by one slab thickness so the ramp's TOP surface runs
        // deck to deck. Aligned by its underside instead, it met each landing
        // along a line rather than a face and both ends floated.
        eaveU: -hd + C.rampWidth, eaveV: deckTop(k) - C.slabThickness,
        ridgeU: hd - C.rampWidth, ridgeV: deckTop(k + 1) - C.slabThickness,
        thickness: C.slabThickness,
      });
    }

    // ── perimeter upstand: a barrier, not a wall ───────────────────────────
    const y0 = deckTop(k), y1 = y0 + C.spandrelHeight, T = C.spandrelThickness;
    for (const [min, max] of [
      [[-hw, y0, -hd], [rampX0, y1, -hd + T]],
      [[-hw, y0, hd - T], [rampX0, y1, hd]],
      [[-hw, y0, -hd + T], [-hw + T, y1, hd - T]],
    ]) b.box({ type: 'parapet', material: 'facade-panel', min, max });
  }

  b.build();
  const top = deckTop(C.levels) + C.spandrelHeight;
  return { pack: b.emit({ cameraTarget: [0, top * 0.5, 0], cameraDistance: 90 }), builder: b };
}

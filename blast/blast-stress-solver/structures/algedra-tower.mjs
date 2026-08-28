/**
 * Algedra concept apartment block — seven storeys of stacked curved white
 * balcony bands over a stilted, fully glazed ground floor.
 *
 * Modelled on the reference at
 * algedra.ae/en/architectural-designs/projects/algedra-concept-architecture
 *
 * The signature of that building is the balconies: soft-edged white slabs that
 * bulge and recede as they run around each floor, with continuous blue glazing
 * set back behind them. The pack format has NO ROTATION and no curved
 * primitives, so the curve cannot be a swept surface — it has to be baked into
 * convex hulls. Each band is therefore a run of straight trapezoids whose
 * OUTWARD DEPTH varies sinusoidally: no single piece is curved, and at sixteen
 * segments per facade the silhouette reads as one.
 *
 * Structurally it is an ordinary reinforced-concrete frame — footings, columns,
 * slabs — because that is what carries a building of this shape, and because a
 * frame is what makes the destruction legible: take out a ground-floor stilt
 * and the corner above it comes down, blow out a pane and only the pane goes.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { facadeBand, bandCorner, parapetOn, staircase, stairShaft, slabWithOpening, shaftVoid, unionStairFootprints, DRY_RUN } from './lib/elements.mjs';

export const ALGEDRA = {
  // The reference block is markedly wider than it is tall — roughly 2:1 in
  // elevation. At 34 m it came out square and read as a tower rather than as
  // the low, spreading apartment building it is.
  width: 46.0,          // X
  depth: 24.0,          // Z
  floors: 7,
  floorHeight: 3.4,
  groundHeight: 4.6,    // the tall, open, glazed base
  slabThickness: 0.35,
  columnSize: 0.6,
  // A stilt here carries ~900 t. At 0.5 m square that is 35 MPa on the
  // concrete it lands on; 0.8 m brings it to 14 MPa, which is what a base
  // column of a seven-storey building actually looks like.
  stiltSize: 0.8,
  glassThickness: 0.06,
  balconyMin: 1.15,     // trough of the wave
  balconyMax: 2.35,     // crest
  balconyWaves: 2.5,    // wave crests per facade
  parapetHeight: 1.05,
  parapetThickness: 0.16,
  beamDepth: 0.65,
  // Wider than anything that lands on it — the 0.8 m stilts and the 0.6 m
  // columns. A beam narrower than its column leaves the column's corners
  // bearing on the slab panels either side, which is where the load then goes.
  beamWidth: 0.9,
  segments: 16,         // per facade, per floor
  paneWidth: 3.0,
  footingDepth: 1.2,
};

/** @returns {{ pack, builder }} — the builder carries the shard statistics. */
export function buildAlgedra(cfg = {}) {
  const C = { ...ALGEDRA, ...cfg };
  const b = new ScenePackBuilder({
    key: 'algedra_tower',
    title: 'Algedra concept apartment block (7 storeys, curved balcony bands)',
    seed: 0xa16e,
  });

  const hx = C.width / 2, hz = C.depth / 2;
  const cs = C.columnSize / 2, ss = C.stiltSize / 2;
  const half = [hx, hz];

  // Level k slab occupies [slabBase(k), slabBase(k) + slabThickness].
  const podiumTop = 0.4;
  // Level 0 is the podium, which sits on the footings; levels 1..roofLevel are
  // the suspended floors.
  const slabBase = (k) => (k === 0 ? 0 : podiumTop + C.groundHeight + (k - 1) * C.floorHeight);
  const slabTop = (k) => (k === 0 ? podiumTop : slabBase(k) + C.slabThickness);
  const roofLevel = C.floors + 1;                    // the roof is one more slab

  // Column grid: five lines in X, three in Z, with the outer lines carried out
  // to the facade.
  //
  // They used to stop at 0.72 and 0.62 of the half-width, which left 6.4 m of
  // slab in X and 4.6 m in Z hanging off the last column with nothing under it
  // but the lobby glazing -- for seven storeys, plus the balcony bands beyond
  // that. A building does not work that way: the perimeter of every floor was
  // being carried by cantilever action back to a line six metres inboard, and
  // the only thing standing where the facade meets the ground was glass.
  //
  // Perimeter columns are what a real stilted base has, and they are also what
  // makes the ground floor worth shooting at: cut one and the bay above it has
  // somewhere to try to redistribute to, which is the whole point. Kept just
  // clear of the glazing plane (bx below) so the two never share space.
  const perimeterX = hx - 1.6;
  const perimeterZ = hz - 1.4;
  const colX = [-perimeterX, -hx * 0.36, 0, hx * 0.36, perimeterX];
  const colZ = [-perimeterZ, 0, perimeterZ];

  const bw = C.beamWidth / 2;
  const beamBottom = (k) => slabBase(k) - C.beamDepth;
  // Beams stop just inside the perimeter glazing plane. Run to the slab edge
  // and they pass straight through the glass.
  const bx = hx - 2 * C.glassThickness, bz = hz - 2 * C.glassThickness;

  // ── footings, pinned to the world ────────────────────────────────────────
  for (const x of colX) {
    for (const z of colZ) {
      // Exactly the ground beam's width. A footing wider than the beam pokes
      // out from under it, the podium panels come to rest on those corners, and
      // because a support is the shortest path to ground the static walk sends
      // the floor's weight through 0.2 m^2 of 12 MPa slab instead of down the
      // beam — 152% of yield standing still.
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [x - bw, -C.footingDepth, z - bw], max: [x + bw, 0, z + bw],
        fixed: true, fracture: false });
    }
  }

  // ── podium: ground beams on the column lines, panels between them ────────
  // The stilts land on the BEAMS, not on the slab. Sitting them on a 12 MPa
  // floor plate put 900 t through 0.25 m^2 at three times yield — the single
  // worst joint in the building, and an entirely self-inflicted one, since a
  // ground beam over the footing is what carries a column in any real frame.
  for (const z of colZ) {
    b.box({ type: 'beam', material: 'reinforced-concrete',
      min: [-bx, 0, z - bw], max: [bx, podiumTop, z + bw] });
  }
  {
    const cuts = [-bz, ...colZ, bz];
    for (const x of colX) {
      for (let j = 0; j + 1 < cuts.length; j++) {
        const z0 = j === 0 ? cuts[0] : cuts[j] + bw;
        const z1 = j + 2 === cuts.length ? cuts[j + 1] : cuts[j + 1] - bw;
        if (!(z1 - z0 > 0.1)) continue;
        b.box({ type: 'beam', material: 'reinforced-concrete',
          min: [x - bw, 0, z0], max: [x + bw, podiumTop, z1] });
      }
    }
  }

  // ── ground floor: steel stilts, stone cores, a glazed lobby box ──────────
  // Steel is the most ductile material in the table (band 12), so the stilts
  // yield over many frames instead of snapping — the base deforms before it
  // fails, which is what makes a hit here read as a building being wounded.
  for (const x of colX) {
    for (const z of colZ) {
      b.box({ type: 'stilt', material: 'steel',
        min: [x - ss, podiumTop, z - ss], max: [x + ss, slabBase(1) - C.beamDepth, z + ss] });
    }
  }
  // Two stone-clad circulation cores, which is what actually braces a stilted
  // base against sway.
  for (const s of [-1, 1]) {
    b.box({ type: 'core', material: 'stone',
      min: [s * 6.2 - 1.6, podiumTop, -2.4], max: [s * 6.2 + 1.6, slabBase(1) - C.beamDepth, 2.4] });
  }
  // Lobby glazing: a set-back glass box between the stilts.
  const lobbyX = hx * 0.5, lobbyZ = hz * 0.55;
  const LOBBY_GAP = C.beamDepth + 0.02;   // clears the first-floor beams too
  for (const [face, fixedAxis] of [['+z', 'z'], ['-z', 'z'], ['+x', 'x'], ['-x', 'x']]) {
    const sign = face[0] === '+' ? 1 : -1;
    const along = fixedAxis === 'z' ? lobbyX : lobbyZ - 2 * C.glassThickness;
    const at = fixedAxis === 'z' ? sign * lobbyZ : sign * lobbyX;
    const panes = Math.max(2, Math.round((2 * along) / C.paneWidth));
    for (let i = 0; i < panes; i++) {
      const a0 = -along + (2 * along * i) / panes, a1 = -along + (2 * along * (i + 1)) / panes;
      b.piece({
        type: 'glazing', material: 'glass', axis: fixedAxis,
        lo: at - C.glassThickness / 2, hi: at + C.glassThickness / 2,
        poly: fixedAxis === 'z'
          ? [[a0, podiumTop], [a1, podiumTop], [a1, slabBase(1) - LOBBY_GAP], [a0, slabBase(1) - LOBBY_GAP]]
          : [[podiumTop, a0], [podiumTop, a1], [slabBase(1) - LOBBY_GAP, a1], [slabBase(1) - LOBBY_GAP, a0]],
      });
    }
  }

  // ── floor slabs, as a grid of cells ──────────────────────────────────────
  // A grid rather than one plate, for the reason the city exporter records:
  // every Voronoi cell of a whole plate is a different shape, whereas cutting
  // each CELL keys shard shapes on the cell size and makes them reusable.
  // Panel boundaries sit ON the column lines, so every panel lands on the beams
  // below it. An arbitrary grid leaves panels stranded between beams, and their
  // weight then has to travel to a support through slab-to-slab seams — which
  // is how a flat plate ends up with seams at four times yield standing still.
  const cellX = [-hx, ...colX, hx];
  const cellZ = [-hz, ...colZ, hz];
  // Panels stop at the beam faces. Every interior boundary IS a column line and
  // therefore has a beam standing in it, so the inset is what the beam fills —
  // it leaves no gap in the floor.
  const inset = (arr, i) => [
    i > 0 ? arr[i] + bw : arr[i],
    i + 2 < arr.length ? arr[i + 1] - bw : arr[i + 1],
  ];
  // The stair core rises through one panel of the grid, in the same place on
  // every floor. A tower's core also BRACES it, which is why this one is
  // enclosed while the houses get an open feature stair.
  const STAIR_CELL = { i: 2, j: 1 };
  // Clear of the lobby glazing plane at z = -hz*0.55 and of every beam line;
  // the shaft walls stand 0.25 m outside the footprint, which is what the
  // first placement forgot and drove straight through the ground-floor glass.
  const stairAt = [-7.4, -6.0];
  // Measured from dry runs of the stair builder rather than hardcoded: flight
  // length falls out of the storey height and the tread size, so a literal
  // would silently stop matching the moment either changed.
  //
  // Per storey, because the ground floor is 4.6 m against a typical 3.4 m —
  // its flight is 5.1 m long where the others are 3.7 m, and sizing every
  // shaft from a typical floor drove the ground-storey stair straight out
  // through its own shaft wall.
  const dryRun = (k) => staircase(DRY_RUN, {
    at: stairAt, y0: slabTop(k), y1: slabTop(k + 1), axis: 'x',
  });
  const stairVoid = unionStairFootprints(
    Array.from({ length: C.floors + 1 }, (_, k) => dryRun(k)),
  );
  const CORE_T = 0.25;
  // The hole is the shaft INTERIOR less the arrival strip. Not expanded to the
  // wall faces: the walls stop under each plate now, so the plate covers their
  // band and they bear on it.
  const stairOpening = stairVoid.void;
  const stairFootprint = (k) => staircase(b, {
    at: stairAt, y0: slabTop(k), y1: slabTop(k + 1), axis: 'x',
    material: 'reinforced-concrete', newelPost: false,
  });

  for (let k = 0; k <= roofLevel; k++) {
    for (let i = 0; i + 1 < cellX.length; i++) {
      const [x0, x1] = inset(cellX, i);
      for (let j = 0; j + 1 < cellZ.length; j++) {
        const [z0, z1] = inset(cellZ, j);
        const min = [x0, slabBase(k), z0], max = [x1, slabTop(k), z1];
        // Every floor the stair passes through needs a void in this panel; the
        // podium does not, because nothing arrives from below it.
        if (k > 0 && i === STAIR_CELL.i && j === STAIR_CELL.j) {
          slabWithOpening(b, { min, max, opening: stairOpening });
        } else {
          b.box({ type: 'slab', material: 'concrete-slab', min, max });
        }
      }
    }
  }

  // ── stair core ───────────────────────────────────────────────────────────
  for (let k = 0; k <= C.floors; k++) {
    stairFootprint(k);
    // Spans THROUGH the plate above, not up to its underside. A shaft that
    // stops at slab level sits entirely inside the floor void and touches
    // nothing at all — 1,364 stair pieces and the whole core came back as
    // floating. Running it through puts the slab edge against the wall face,
    // which is the joint a core actually has.
      // Stops UNDER the plate, which then passes over it. The alternative —
      // running the wall through the plate — needs the plate cut away wherever
      // the wall is, and that cut-out cannot coexist with the arrival strip:
      // the side walls run the full length of the shaft, including alongside
      // the strip. Stopping the wall here gives the core a far better joint
      // anyway: its whole top face bears on the slab, rather than the slab's
      // edge meeting its side.
    stairShaft(b, {
      footprint: stairVoid, y0: slabTop(k), y1: slabBase(k + 1),
      material: 'reinforced-concrete', openSide: '-x', thickness: CORE_T,
    });
  }

  // ── downstand beams on the column lines ──────────────────────────────────
  // A 34 x 22 m plate on twelve columns is not a flat slab in any real
  // building, and modelling it as one shows why: with the only paths to a
  // column being slab-to-slab seams, a whole floor's weight converges on the
  // one cell above each column and the seams feeding it run at four times
  // yield standing still. Beams give that load a member with real area to
  // travel along, which is exactly what they are for.
  //
  // The beams run from below the slab up to the SLAB TOP, not merely to its
  // underside, so the column above lands on the beam rather than on the slab.
  // That distinction is the whole ball game: a column bearing on 0.35 m of
  // 12 MPa slab put 490 t through 0.14 m^2 at 291% of yield, while the same
  // load on the beam's full 0.36 m^2 of 48 MPa concrete sits under a third of
  // it. Slabs do not carry columns in real frames either.
  //
  // The X beams run the full width and the Z beams are segmented between them,
  // so the two sets meet at the column lines instead of sharing space.
  for (let k = 1; k <= roofLevel; k++) {
    for (const z of colZ) {
      b.box({ type: 'beam', material: 'reinforced-concrete',
        min: [-bx, beamBottom(k), z - bw], max: [bx, slabTop(k), z + bw] });
    }
    const cuts = [-bz, ...colZ, bz];
    for (const x of colX) {
      for (let j = 0; j + 1 < cuts.length; j++) {
        const z0 = j === 0 ? cuts[0] : cuts[j] + bw;
        const z1 = j + 2 === cuts.length ? cuts[j + 1] : cuts[j + 1] - bw;
        if (!(z1 - z0 > 0.1)) continue;
        b.box({ type: 'beam', material: 'reinforced-concrete',
          min: [x - bw, beamBottom(k), z0], max: [x + bw, slabTop(k), z1] });
      }
    }
  }

  // ── columns between slabs ────────────────────────────────────────────────
  // Stops at the UNDERSIDE of the beam it carries, not at the slab: running it
  // into the beam would put the two in the same space and produce no joint.
  for (let k = 1; k <= C.floors; k++) {
    for (const x of colX) {
      for (const z of colZ) {
        b.box({ type: 'column', material: 'reinforced-concrete',
          min: [x - cs, slabTop(k), z - cs], max: [x + cs, beamBottom(k + 1), z + cs] });
      }
    }
  }

  // ── perimeter glazing, set back behind the balconies ─────────────────────
  // The pane stops short of the slab above it. Without that gap the glazing
  // bridges slab to slab and becomes a structural column: the static walk finds
  // it is a shorter path to ground than the concrete frame and routes the
  // building's weight through a 21 cm^2 glazing clip at eighty times yield.
  // Real curtain wall is hung off one slab edge and carries only itself; the
  // gap is the movement joint that makes that true here.
  const HEAD_GAP = 0.02;
  for (let k = 1; k <= C.floors; k++) {
    const y0 = slabTop(k), y1 = slabBase(k + 1) - C.beamDepth - HEAD_GAP;
    for (const [fixedAxis, sign] of [['z', 1], ['z', -1], ['x', 1], ['x', -1]]) {
      // The X facades stop short of the Z ones, which run the full width;
      // otherwise the two glazing planes cross at every corner of the building.
      const along = fixedAxis === 'z' ? hx : hz - 2 * C.glassThickness;
      const at = sign * ((fixedAxis === 'z' ? hz : hx) - C.glassThickness);
      const panes = Math.max(2, Math.round((2 * along) / C.paneWidth));
      for (let i = 0; i < panes; i++) {
        const a0 = -along + (2 * along * i) / panes, a1 = -along + (2 * along * (i + 1)) / panes;
        b.piece({
          type: 'glazing', material: 'glass', axis: fixedAxis,
          lo: at - C.glassThickness / 2, hi: at + C.glassThickness / 2,
          poly: fixedAxis === 'z'
            ? [[a0, y0], [a1, y0], [a1, y1], [a0, y1]]
            : [[y0, a0], [y0, a1], [y1, a1], [y1, a0]],
        });
      }
    }
  }

  // ── the balcony bands: the building's signature ──────────────────────────
  // Depth waves along each facade. The phase is offset per floor so the crests
  // do not stack into vertical columns, which is what makes the stack read as
  // organic rather than extruded.
  const depthAt = (k) => (t) => {
    const phase = (k % 2) * 0.5;
    const w = 0.5 + 0.5 * Math.cos(2 * Math.PI * (t * C.balconyWaves + phase));
    return C.balconyMin + (C.balconyMax - C.balconyMin) * w;
  };

  for (let k = 1; k <= roofLevel; k++) {
    const lo = slabBase(k), hi = slabTop(k);
    const depth = depthAt(k);
    const decks = [];
    for (const face of ['+z', '-z', '+x', '-x']) {
      decks.push(...facadeBand(b, {
        material: 'reinforced-concrete', face, half, lo, hi, segments: C.segments, depth,
      }));
    }
    // Fill the four corner notches where perpendicular bands meet.
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      bandCorner(b, { material: 'reinforced-concrete', half, lo, hi, sx, sz, depth: C.balconyMin });
    }
    // White parapets on the outer lip. Every floor gets one; on the roof it is
    // the building's crown.
    for (const deck of decks) {
      // Cladding, not structure: a parapet carries nothing but itself, so it
      // should come off when the building is hit without the floor it stands on
      // caring. The balcony DECK below stays reinforced — it is a real
      // cantilever holding people up.
      parapetOn(b, deck, {
        material: 'facade-panel', lo: hi, hi: hi + C.parapetHeight,
        thickness: C.parapetThickness,
      });
    }
  }

  b.build();
  const top = slabTop(roofLevel) + C.parapetHeight;
  return { pack: b.emit({ cameraTarget: [0, top * 0.45, 0], cameraDistance: 95 }), builder: b };
}

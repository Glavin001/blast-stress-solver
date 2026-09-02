/**
 * Test rigs: the smallest structures that can answer one question each.
 *
 * The seven authored buildings are the product; these are the instruments. A
 * question like "at what overhang does a floor give way" is unanswerable on
 * Petronas — 23,576 chunks of everything happening at once, and twenty minutes
 * of simulation per attempt. It is answerable in seconds on a wall with six
 * shelves sticking out of it at known lengths.
 *
 * Each rig is deliberately austere: no cladding, no stairs, no detail that is
 * not load-bearing for the question. They are also small enough (tens to a few
 * hundred chunks) that a whole scenario suite runs in minutes rather than
 * hours, which is what makes it a thing you run on every change instead of
 * once a week.
 *
 * They share the real material table and the real fracture rules. A rig
 * calibrated against special-cased physics would prove nothing about the
 * buildings.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { columnGrid, floorPlate, frame, gridLines } from './lib/elements.mjs';

const CONCRETE = 'reinforced-concrete';

/** One column on a footing, carrying a slab cap. */
export function buildRigColumn(cfg = {}) {
  const C = {
    height: 3.2, size: 0.55, capSpan: 2.4, capThickness: 0.3, footingDepth: 1.0, ...cfg,
  };
  const b = new ScenePackBuilder({
    key: 'rig_column',
    title: 'Rig — one column under a cap',
    seed: 0x8101,
  });
  const half = C.size / 2;
  b.box({
    type: 'foundation', material: 'footing-anchor',
    min: [-0.7, -C.footingDepth, -0.7], max: [0.7, 0, 0.7],
    fixed: true, fracture: false,
  });
  b.box({
    type: 'column', material: CONCRETE,
    min: [-half, 0, -half], max: [half, C.height, half],
  });
  floorPlate(b, {
    material: CONCRETE,
    min: [-C.capSpan / 2, -C.capSpan / 2], max: [C.capSpan / 2, C.capSpan / 2],
    y: C.height + C.capThickness, thickness: C.capThickness,
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, C.height * 0.5, 0], cameraDistance: 14 }),
    builder: b,
  };
}

/** Two columns and a beam: the smallest thing that can span. */
export function buildRigPortal(cfg = {}) {
  const C = {
    span: 6.0, height: 3.2, size: 0.5, beamDepth: 0.5, deckThickness: 0.3,
    depth: 3.0, footingDepth: 1.0, ...cfg,
  };
  const b = new ScenePackBuilder({
    key: 'rig_portal',
    title: 'Rig — portal frame',
    seed: 0x8102,
  });
  const xs = [-C.span / 2, C.span / 2];
  const half = C.size / 2;
  for (const x of xs) {
    b.box({
      type: 'foundation', material: 'footing-anchor',
      min: [x - 0.7, -C.footingDepth, -0.7], max: [x + 0.7, 0, 0.7],
      fixed: true, fracture: false,
    });
  }
  columnGrid(b, {
    material: CONCRETE, xs, zs: [0], size: C.size,
    y0: 0, y1: C.height - C.beamDepth,
  });
  // The beam spans the columns and the deck sits on the beam, so load walks
  // deck -> beam -> column -> footing with a real face at every hand-off.
  b.box({
    type: 'beam', material: CONCRETE,
    min: [xs[0] - half, C.height - C.beamDepth, -half],
    max: [xs[1] + half, C.height, half],
  });
  floorPlate(b, {
    material: CONCRETE,
    min: [xs[0] - half, -C.depth / 2], max: [xs[1] + half, C.depth / 2],
    y: C.height + C.deckThickness, thickness: C.deckThickness,
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, C.height * 0.5, 0], cameraDistance: 20 }),
    builder: b,
  };
}

/**
 * A wall with shelves sticking out of it at increasing lengths.
 *
 * The ductility instrument. Each rung is the same slab at a different
 * cantilever, so the only variable is the overhang, and the question "how far
 * can a floor reach before its own weight tears it off the wall" has a
 * readable answer: which rungs survive, and — for the ones that do not — do
 * the marginal ones take longer to go than the hopeless ones.
 *
 * Rungs are spaced far apart along the wall and tagged by role
 * (`rung-1` ... `rung-N`) so a scenario can name one without geometry, and so
 * simulator jitter cannot reorder two lengths that differ by metres.
 */
export function buildRigCantilever(cfg = {}) {
  // Lengths chosen to STRADDLE the failure point rather than to look dramatic.
  // A 0.2 m slab three metres wide weighs 14.1 kN per metre of reach, so the
  // moment at its root is wL^2/2 against a section modulus of bh^2/6 = 0.02 m^3
  // — which puts the extreme fibre at the reinforced tension limit at about
  // 6.4 m. Half the ladder should therefore hold and half should not, which is
  // the only arrangement that can tell "fails by length" from "fails" or
  // "holds".
  //
  // The first version used 0.3 m slabs reaching 6 m and nothing failed, which
  // was the rig being wrong rather than the physics: that is a stout cantilever
  // and a real one would stand up too.
  const C = {
    overhangs: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    spacing: 5.0, width: 3.0, thickness: 0.2,
    wallThickness: 0.6, wallHeight: 4.0, footingDepth: 1.2, ...cfg,
  };
  const b = new ScenePackBuilder({
    key: 'rig_cantilever',
    title: 'Rig — cantilever ladder',
    seed: 0x8103,
  });
  const n = C.overhangs.length;
  const spanZ = C.spacing * n;
  const z0 = -spanZ / 2;
  const wallX1 = C.wallThickness / 2;

  b.box({
    type: 'foundation', material: 'footing-anchor',
    min: [-wallX1 - 0.3, -C.footingDepth, z0 - 0.6],
    max: [wallX1 + 0.3, 0, z0 + spanZ + 0.6],
    fixed: true, fracture: false,
  });
  // A single thick wall, so every rung is rooted in the same thing and any
  // difference between them is the overhang and nothing else.
  b.box({
    type: 'wall', material: CONCRETE,
    min: [-wallX1, 0, z0 - 0.6], max: [wallX1, C.wallHeight, z0 + spanZ + 0.6],
  });

  C.overhangs.forEach((overhang, index) => {
    const centre = z0 + C.spacing * (index + 0.5);
    b.box({
      type: `rung-${index + 1}`, material: CONCRETE,
      min: [wallX1, C.wallHeight - C.thickness, centre - C.width / 2],
      max: [wallX1 + overhang, C.wallHeight, centre + C.width / 2],
    });
  });

  b.build();
  return {
    pack: b.emit({ cameraTarget: [2, C.wallHeight * 0.6, 0], cameraDistance: 34 }),
    builder: b,
  };
}

/**
 * A parking garage reduced to its skeleton: a column grid under flat plates.
 *
 * This is the structure the whole exercise is about. It is the arrangement
 * with nowhere to redistribute — no walls, no core, no frame action worth the
 * name — so taking columns out has to show up as load arriving somewhere else,
 * and taking enough of them out has to bring the plates down.
 *
 * Sized to be a real redistribution problem (4x4 bays, two levels) while
 * staying a few hundred chunks, so a scenario runs in seconds.
 */
export function buildRigGarage(cfg = {}) {
  const C = {
    width: 24.0, depth: 24.0, levels: 2, storyHeight: 3.2,
    slabThickness: 0.3, columnSize: 0.55, bays: 4, footingDepth: 1.0, ...cfg,
  };
  const b = new ScenePackBuilder({
    key: 'rig_garage',
    title: 'Rig — pier grid under plates',
    seed: 0x8104,
  });
  const hw = C.width / 2, hd = C.depth / 2;
  const xs = gridLines(-hw, hw, C.bays, C.columnSize);
  const zs = gridLines(-hd, hd, C.bays, C.columnSize);
  const built = frame(b, {
    material: CONCRETE, xs, zs,
    levels: C.levels, storyHeight: C.storyHeight,
    columnSize: C.columnSize, slabThickness: C.slabThickness,
    min: [-hw, -hd], max: [hw, hd], footingDepth: C.footingDepth,
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, built.topY * 0.5, 0], cameraDistance: 46 }),
    builder: b,
  };
}

/** A glazed panel in a frame: the minimum-shatter instrument. */
export function buildRigPane(cfg = {}) {
  const C = { width: 2.4, height: 2.0, thickness: 0.02, mullion: 0.12, ...cfg };
  const b = new ScenePackBuilder({
    key: 'rig_pane',
    title: 'Rig — glazed panel',
    seed: 0x8105,
  });
  const hw = C.width / 2, m = C.mullion;
  b.box({
    type: 'foundation', material: 'footing-anchor',
    min: [-hw - m, -0.6, -m / 2], max: [hw + m, 0, m / 2],
    fixed: true, fracture: false,
  });
  // The mullions stop at the head and the transom sits ON them: they meet over
  // a real face, which is a bearing the solver can carry load through. Running
  // both full-width instead makes them share volume, which is not a joint —
  // it is two pieces occupying the same space, and verify rejects it.
  for (const x of [-hw - m / 2, hw + m / 2]) {
    b.box({
      type: 'mullion', material: 'steel',
      min: [x - m / 2, 0, -m / 2], max: [x + m / 2, C.height, m / 2],
    });
  }
  b.box({
    type: 'transom', material: 'steel',
    min: [-hw - m, C.height, -m / 2], max: [hw + m, C.height + m, m / 2],
  });
  b.box({
    type: 'glazing', material: 'glass',
    min: [-hw, 0, -C.thickness / 2], max: [hw, C.height, C.thickness / 2],
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, C.height * 0.5, 0], cameraDistance: 8 }),
    builder: b,
  };
}

/** A brick wall on a footing: the weak-material shot response. */
export function buildRigWall(cfg = {}) {
  const C = { length: 8.0, height: 3.0, thickness: 0.35, footingDepth: 0.8, ...cfg };
  const b = new ScenePackBuilder({
    key: 'rig_wall',
    title: 'Rig — brick wall',
    seed: 0x8106,
  });
  const hl = C.length / 2, ht = C.thickness / 2;
  b.box({
    type: 'foundation', material: 'footing-anchor',
    min: [-hl - 0.2, -C.footingDepth, -ht - 0.2], max: [hl + 0.2, 0, ht + 0.2],
    fixed: true, fracture: false,
  });
  b.box({
    type: 'wall', material: 'brick',
    min: [-hl, 0, -ht], max: [hl, C.height, ht],
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, C.height * 0.5, 0], cameraDistance: 16 }),
    builder: b,
  };
}

/**
 * A heavy slab resting on one end ledge, most of it hanging over nothing.
 *
 * The re-grounding instrument. A building knocked onto its side is no longer
 * standing on its foundations — it is lying on the ground, with its weight
 * arriving through whatever happens to be touching. That contact is a real
 * load path and it should crush what it bears on.
 *
 * Authored lying down rather than rotated at runtime: a quarter turn is the
 * only rotation this format can do exactly, and the interesting angles are not
 * quarter turns.
 */
export function buildRigToppled(cfg = {}) {
  const C = {
    length: 9.0, width: 3.0, thickness: 0.9,
    ledgeLength: 1.5, ledgeHeight: 2.0, footingDepth: 1.0, ...cfg,
  };
  const b = new ScenePackBuilder({
    key: 'rig_toppled',
    title: 'Rig — slab on one ledge',
    seed: 0x8107,
  });
  const hw = C.width / 2, hl = C.length / 2;
  b.box({
    type: 'foundation', material: 'footing-anchor',
    min: [-hl - 0.3, -C.footingDepth, -hw - 0.3],
    max: [-hl + C.ledgeLength + 0.3, 0, hw + 0.3],
    fixed: true, fracture: false,
  });
  b.box({
    type: 'ledge', material: CONCRETE,
    min: [-hl, 0, -hw], max: [-hl + C.ledgeLength, C.ledgeHeight, hw],
  });
  b.box({
    type: 'slab', material: CONCRETE,
    min: [-hl, C.ledgeHeight, -hw], max: [hl, C.ledgeHeight + C.thickness, hw],
  });
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, C.ledgeHeight, 0], cameraDistance: 22 }),
    builder: b,
  };
}

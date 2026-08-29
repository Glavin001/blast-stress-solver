/**
 * Structural components: pieces small enough to prove, shaped to compose.
 *
 * ## Why
 *
 * Every building fixed so far was fixed the same way -- audit the whole thing,
 * find one detail carrying what it cannot, change that detail, audit again.
 * That works and it is slow, because the unit of iteration is a whole tower:
 * 432 Park takes 84 seconds to reach a verdict and the walled city 96, so a
 * wrong guess costs minutes and a right one is only confirmed at the end.
 *
 * It is also the wrong unit for finding out what is TRUE. When 432 Park was
 * unstable it was not because "the tower" was wrong, it was because a plate
 * spanned 28 m over a stub, and later because 1,536 columns were never emitted.
 * Both are statements about a bay, not about a building.
 *
 * So: build bays. Prove a bay converges, standing on its own, with known
 * headroom. Then compose bays and prove the composition. The same discipline as
 * functions and unit tests -- and the same payoff, which is that when a big
 * thing breaks you already know which small thing to suspect.
 *
 * ## What a component is here
 *
 * A function that emits into an EXISTING builder at a given origin, so the same
 * code makes the standalone test and the real building. Not a separate "test
 * version": a component proven in a rig and then used through a different path
 * has been proven about the rig.
 *
 * ## The two questions a component has to answer
 *
 *   1. Does it converge on its own, at the SHIPPING iteration count, quickly?
 *      Quickly is the point. Iterations are fixed at 32 because 128 costs
 *      56 ms/tick on a large structure and a 60 Hz frame is 16.7 ms, so how
 *      fast a thing settles at 32 is a property of the thing. A bay that needs
 *      a minute is badly conditioned and will not get better in company.
 *   2. Does it still converge when composed? Stability is not automatically
 *      additive -- a row shares load through its joints, and the middle of a
 *      row is not the same problem as a bay alone. `row()` exists to ask that
 *      at 1, 2, 4 and 8 bays, so the answer is a curve rather than a guess.
 */

import { ScenePackBuilder } from './lib/pack.mjs';
import { columnGrid, beamsBetween, floorPlate } from './lib/elements.mjs';

/** Depth of the anchor block under a component under test. */
const FOOTING = 0.8;

/**
 * One bay of a framed floor: four columns, beams between them, a plate on top.
 *
 * The unit almost every tower here is really made of. 432 Park's ribs, Algedra's
 * grid and the parking garage's decks are all this, repeated.
 */
export function frameBay(b, {
  at = [0, 0, 0],
  span = 6.0,          // centre to centre, both directions
  storey = 3.4,
  columnSize = 0.6,
  beamDepth = 0.6,
  beamWidth = 0.45,
  slabThickness = 0.3,
  material = 'reinforced-concrete',
  footings = true,
  // Which boundary column line this bay owns. Adjacent bays in a row SHARE
  // their common line, so the bay to the right does not re-emit it -- two bays
  // each emitting the same columns is two columns inside each other, and two
  // plates lapping over the shared line by a column width. Naive tiling does
  // not compose; a grid does, and this is the difference between them.
  skipMinusX = false,
} = {}) {
  const [ox, oy, oz] = at;
  const h = span / 2;
  const xs = [ox - h, ox + h];
  const zs = [oz - h, oz + h];
  const colXs = skipMinusX ? [xs[1]] : xs;
  const deckBase = oy + storey - beamDepth;

  if (footings) {
    const c = columnSize / 2 + 0.15;
    for (const x of colXs) {
      for (const z of zs) {
        b.box({
          type: 'foundation', material: 'footing-anchor',
          min: [x - c, oy - FOOTING, z - c], max: [x + c, oy, z + c],
          fixed: true, fracture: false,
        });
      }
    }
  }
  columnGrid(b, { material, xs: colXs, zs, y0: oy, y1: deckBase, size: columnSize });
  // Beams both ways, spanning between the column centres.
  beamsBetween(b, {
    material, axis: 'x', lines: zs, from: xs[0], to: xs[1],
    y: oy + storey, depth: beamDepth, width: beamWidth,
  });
  // The Z beams butt INTO the X beams rather than crossing them. Four beams
  // round a bay meet at four corners, and left full-length they share volume
  // there -- the same mistake that produced 1,434 overlapping pairs when a
  // beam grid was first tried on 432 Park. Continuous primary, secondary
  // framing into it, which is also how it is actually built.
  beamsBetween(b, {
    material, axis: 'z', lines: colXs,
    from: zs[0] + beamWidth / 2, to: zs[1] - beamWidth / 2,
    y: oy + storey, depth: beamDepth, width: beamWidth,
  });
  // Plate spans column CENTRE to centre, so two bays side by side meet exactly
  // on the shared line instead of lapping over it.
  floorPlate(b, {
    material,
    min: [xs[0], zs[0] - columnSize / 2],
    max: [xs[1], zs[1] + columnSize / 2],
    y: oy + storey + slabThickness, thickness: slabThickness,
  });
}

/**
 * One bay of load-bearing masonry wall on its footing.
 *
 * The walled city is this repeated around an arc, which is the case that
 * prompted componentising at all.
 */
export function wallBay(b, {
  at = [0, 0, 0],
  length = 6.0,
  height = 6.0,
  thickness = 0.8,
  material = 'white-stone',
  parapet = true,
  footings = true,
} = {}) {
  const [ox, oy, oz] = at;
  const hl = length / 2, ht = thickness / 2;
  if (footings) {
    b.box({
      type: 'foundation', material: 'footing-anchor',
      min: [ox - hl, oy - FOOTING, oz - ht - 0.2],
      max: [ox + hl, oy, oz + ht + 0.2],
      fixed: true, fracture: false,
    });
  }
  b.box({
    type: 'wall', material,
    min: [ox - hl, oy, oz - ht], max: [ox + hl, oy + height, oz + ht],
  });
  if (parapet) {
    // A plain capping course rather than elements.parapetOn, which wants a
    // deck descriptor (alongX/sign/offsets) that only makes sense on a real
    // floor edge. A wall's own coping is a box.
    b.box({
      type: 'parapet', material,
      min: [ox - hl, oy + height, oz - ht], max: [ox + hl, oy + height + 0.9, oz + ht],
    });
  }
}

export const COMPONENTS = {
  'frame-bay': frameBay,
  'wall-bay': wallBay,
};

/**
 * Wrap one component in a pack of its own, so it can be audited alone.
 */
export function standalone(name, opts = {}) {
  const emit = COMPONENTS[name];
  if (!emit) throw new Error(`unknown component "${name}" (have: ${Object.keys(COMPONENTS).join(', ')})`);
  const b = new ScenePackBuilder({
    key: `component_${name.replace(/-/g, '_')}`,
    title: `Component — ${name}`,
    seed: 0xC0FFEE,
  });
  emit(b, { ...opts, at: [0, 0, 0] });
  b.build();
  return { pack: b.emit({ cameraTarget: [0, 3, 0], cameraDistance: 22 }), builder: b };
}

/**
 * The same component repeated in a line, sharing joints with its neighbours.
 *
 * This is the composition question made measurable. A bay proven alone has
 * been proven alone; the middle of a row carries its neighbours' thrust and is
 * a different problem. Running this at 1, 2, 4 and 8 turns "does it compose"
 * into a curve -- and if convergence time or peak utilisation climbs with
 * count, that is the interface between bays talking, which is exactly the
 * thing a whole-building audit cannot isolate.
 */
export function row(name, count, opts = {}) {
  const emit = COMPONENTS[name];
  if (!emit) throw new Error(`unknown component "${name}"`);
  const pitch = opts.pitch ?? (name === 'wall-bay' ? (opts.length ?? 6.0) : (opts.span ?? 6.0));
  const b = new ScenePackBuilder({
    key: `component_${name.replace(/-/g, '_')}_x${count}`,
    title: `Component — ${name} x${count}`,
    seed: 0xC0FFEE,
  });
  const first = -((count - 1) * pitch) / 2;
  for (let i = 0; i < count; i += 1) {
    // Every bay but the first hands its left-hand column line to its
    // neighbour, which is what makes this a grid rather than a pile of bays.
    emit(b, { ...opts, at: [first + i * pitch, 0, 0], skipMinusX: i > 0 });
  }
  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, 3, 0], cameraDistance: 22 + count * 6 }),
    builder: b,
  };
}

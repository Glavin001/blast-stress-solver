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
 * How a component is held while it is being tested on its own.
 *
 * A bay tested only on footings has been proven standing on the ground, which
 * is not where most of them live. The pieces that gave the most trouble were
 * the ones attached to something else -- roofs on masonry, balconies off a
 * plate, a spandrel hanging from the floor above -- and their failures were
 * all AT the attachment.
 *
 * So a component declares what it is mounted on, and the mount is emitted as a
 * fixed anchor standing in for the neighbour. The component then meets the
 * same interface it will meet in the building: same face, same material, same
 * bond. What it does against the mock is what it will do in company.
 *
 *   ground   footings underneath, for anything that stands
 *   slab     a fixed plate beneath, for a storey stacked on the one below
 *   wall     a fixed wall face alongside, for anything hung off a wall
 *   free     nothing at all, to find out what a piece cannot do unaided
 *
 * `free` is not a mistake: a component that stands up with no mount is one
 * that is carrying itself, and knowing which pieces those are tells you where
 * a structure's real load paths have to be.
 */
export const MOUNTS = ['ground', 'slab', 'wall', 'free'];

/**
 * Emit the mock support for a component footprint.
 *
 * @param footprint {min:[x,z], max:[x,z]} the plan the component occupies
 */
export function mount(b, kind, footprint, { y = 0, material = 'footing-anchor' } = {}) {
  const [x0, z0] = footprint.min;
  const [x1, z1] = footprint.max;
  switch (kind) {
    case 'ground':
      b.box({
        type: 'foundation', material,
        min: [x0, y - FOOTING, z0], max: [x1, y, z1],
        fixed: true, fracture: false,
      });
      return;
    case 'slab':
      // The floor below, as the component meets it: a plate of real thickness
      // whose top face is the bearing surface, not a block in the ground.
      b.box({
        type: 'foundation', material,
        min: [x0 - 0.3, y - 0.35, z0 - 0.3], max: [x1 + 0.3, y, z1 + 0.3],
        fixed: true, fracture: false,
      });
      return;
    case 'wall':
      // A wall face on the -X side, which is what a balcony, a landing or a
      // lean-to roof actually hangs from.
      b.box({
        type: 'foundation', material,
        min: [x0 - 0.6, y - FOOTING, z0 - 0.3], max: [x0, y + 4.0, z1 + 0.3],
        fixed: true, fracture: false,
      });
      return;
    case 'free':
      return;
    default:
      throw new Error(`unknown mount "${kind}" (have: ${MOUNTS.join(', ')})`);
  }
}

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
  mountKind = 'ground',
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
        mount(b, mountKind, { min: [x - c, z - c], max: [x + c, z + c] }, { y: oy });
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
  mountKind = 'ground',
} = {}) {
  const [ox, oy, oz] = at;
  const hl = length / 2, ht = thickness / 2;
  if (footings) {
    mount(b, mountKind, {
      min: [ox - hl, oz - ht - 0.2], max: [ox + hl, oz + ht + 0.2],
    }, { y: oy });
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


/**
 * Four wall bays enclosing a space: the next rung up from a single wall.
 *
 * The interesting thing a room has that a wall does not is CORNERS, and
 * corners are where masonry either braces itself or does not. Two walls meeting
 * at a right angle each stop the other buckling out of plane, which is why a
 * room stands at proportions a single wall of the same thickness will not.
 */
export function room(b, {
  at = [0, 0, 0],
  size = 6.0,
  height = 4.0,
  thickness = 0.6,
  material = 'white-stone',
  mountKind = 'ground',
  parapet = false,
} = {}) {
  const [ox, oy, oz] = at;
  const h = size / 2, ht = thickness / 2;
  // The two X-walls run the full length and own the corners; the Z-walls stop
  // short of them. Sharing an exact boundary is what makes these bond --
  // overlapping them puts every corner over the interpenetration limit.
  for (const sz of [-1, 1]) {
    wallBay(b, {
      at: [ox, oy, oz + sz * (h - ht)],
      length: size, height, thickness, material, parapet, mountKind,
    });
  }
  const inner = size - thickness * 2;
  for (const sx of [-1, 1]) {
    if (inner <= 0.2) continue;
    const cx = ox + sx * (h - ht);
    if (mountKind !== 'free') {
      // Inset past the X-walls' own footing margin. wallBay pads its footing
      // 0.2 m beyond the wall so the wall lands well inside it; at a corner
      // that padding is exactly what the cross-wall's footing runs into.
      const pad = 0.25;
      mount(b, mountKind, {
        min: [cx - ht, oz - inner / 2 + pad], max: [cx + ht, oz + inner / 2 - pad],
      }, { y: oy });
    }
    b.box({
      type: 'wall', material,
      min: [cx - ht, oy, oz - inner / 2], max: [cx + ht, oy + height, oz + inner / 2],
    });
  }
}

/**
 * A room with a floor over it: one storey, ready to be stacked.
 *
 * Emitted with mountKind 'slab' this is the middle of a stack -- the plate
 * below is a mock, so the storey meets the same bearing it will meet in the
 * real building. That is the whole point of mounts: this measurement is about
 * a storey in a tower, not a storey standing alone in a field.
 */
export function storey(b, {
  at = [0, 0, 0],
  size = 6.0,
  height = 4.0,
  thickness = 0.6,
  slabThickness = 0.3,
  material = 'white-stone',
  deckMaterial = 'reinforced-concrete',
  mountKind = 'ground',
} = {}) {
  const [ox, oy, oz] = at;
  const h = size / 2;
  room(b, { at, size, height, thickness, material, mountKind, parapet: false });
  floorPlate(b, {
    material: deckMaterial,
    min: [ox - h, oz - h], max: [ox + h, oz + h],
    y: oy + height + slabThickness, thickness: slabThickness,
  });
}

/**
 * Storeys stacked, each bearing on the plate of the one below.
 *
 * The rung that answers "how tall before it stops working". Only the ground
 * storey is mounted; the rest stand on the real plate beneath them, so this is
 * a genuine stack rather than a column of independently-anchored boxes.
 */
export function stack(b, {
  at = [0, 0, 0],
  floors = 4,
  size = 6.0,
  height = 4.0,
  thickness = 0.6,
  slabThickness = 0.3,
  material = 'white-stone',
  mountKind = 'ground',
} = {}) {
  const [ox, oy, oz] = at;
  for (let k = 0; k < floors; k += 1) {
    storey(b, {
      at: [ox, oy + k * (height + slabThickness), oz],
      size, height, thickness, slabThickness, material,
      mountKind: k === 0 ? mountKind : 'free',
    });
  }
}

export const COMPONENTS = {
  'frame-bay': frameBay,
  'wall-bay': wallBay,
  room,
  storey,
  stack,
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

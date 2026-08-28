/**
 * All three authored structures in one pack, so a single scene shows the whole
 * set side by side.
 *
 * A ScenePack describes one scene, and the city builder replicates a pack
 * across a grid rather than composing different ones — so "the tower and both
 * houses together" has to be a pack of its own. Merging is only possible
 * because every structure here is built from the SAME material table
 * (`materialTable()`), which means material indices carry over untouched. If
 * that ever stops being true this has to remap them.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { materialTable } from './lib/materials.mjs';
import { buildAlgedra } from './algedra-tower.mjs';
import { buildHouse1 } from './house-1story.mjs';
import { buildHouse2 } from './house-2story.mjs';
import { buildVillaSavoye } from './villa-savoye.mjs';
import { buildPark432 } from './park-432.mjs';
import { buildParkingGarage } from './parking-garage.mjs';
import { buildPetronas } from './petronas.mjs';

const round = (n) => Math.round(n * 1e5) / 1e5;

/** Placement of each structure, chosen so nothing overlaps anything else. */
export const PLACEMENTS = [
  { build: buildAlgedra, at: [0, 0, 0] },
  { build: buildHouse1, at: [-42, 0, 26] },
  { build: buildHouse2, at: [42, 0, 26] },
];

/**
 * Everything, clustered around the ORIGIN.
 *
 * The spawn is random on a circle: measured at (146, 0), (-121, 0), (-189, 0)
 * and (0, 189) across five sessions, roughly 120-190 m out in an arbitrary
 * direction. Nothing can be placed relative to it. A line along one axis was
 * tried and is worse than a cluster — land at the wrong end and the whole
 * skyline is behind you.
 *
 * Clustered at the origin there is one rule that always works: walk towards the
 * middle. Whichever way you spawn, everything is in front of you, and the
 * furthest walk is about 200 m.
 *
 * Spacing is more than each structure's own height, so no two can reach each
 * other falling over and they stay independent to test against.
 */
export const SKYLINE = [
  { build: buildPetronas, at: [0, 0, -96] },
  { build: buildPark432, at: [-72, 0, 8] },
  { build: buildAlgedra, at: [4, 0, 16] },
  { build: buildParkingGarage, at: [74, 0, 12] },
  { build: buildVillaSavoye, at: [-42, 0, 74] },
  { build: buildHouse1, at: [2, 0, 76] },
  { build: buildHouse2, at: [34, 0, 74] },
];

export function buildSkyline() {
  return merge(SKYLINE, 'skyline', 'Skyline — every authored structure', [0, 30, 0], 420);
}

export function buildNeighbourhood() {
  return merge(PLACEMENTS, 'neighbourhood', 'Neighbourhood — Algedra block and two houses', [0, 14, 0], 130);
}

function merge(placements, key, title, cameraTarget, cameraDistance) {
  const table = materialTable();
  const nodes = [], nodeTypes = [], nodeMaterials = [], nodePieces = [],
    nodeSizes = [], nodeColliders = [], bonds = [];
  let nodeBase = 0, pieceBase = 0;

  for (const { build, at } of placements) {
    const { pack } = build();
    const s = pack.scenario;
    if (pack.defaults.solver.materials.length !== table.length) {
      throw new Error('structures no longer share one material table; indices would be wrong');
    }
    for (const node of s.nodes) {
      nodes.push({
        centroid: {
          x: round(node.centroid.x + at[0]),
          y: round(node.centroid.y + at[1]),
          z: round(node.centroid.z + at[2]),
        },
        mass: node.mass,
        volume: node.volume,
        m: node.m,
      });
    }
    // Colliders need no offset: a cuboid is half-extents and a hull's points
    // are centroid-relative, so both travel with the centroid.
    nodeTypes.push(...s.nodeTypes);
    nodeMaterials.push(...s.nodeMaterials);
    nodePieces.push(...s.nodePieces.map((p) => p + pieceBase));
    nodeSizes.push(...s.nodeSizes);
    nodeColliders.push(...s.nodeColliders);
    for (const bond of s.bonds) {
      bonds.push({
        ...bond,
        node0: bond.node0 + nodeBase,
        node1: bond.node1 + nodeBase,
        centroid: {
          x: round(bond.centroid.x + at[0]),
          y: round(bond.centroid.y + at[1]),
          z: round(bond.centroid.z + at[2]),
        },
      });
    }
    nodeBase += s.nodes.length;
    pieceBase += Math.max(0, ...s.nodePieces) + 1;
  }

  // Borrow the builder only for its emit shape, so the pack header stays in one
  // place rather than being written out twice.
  const b = new ScenePackBuilder({ key, title });
  b.nodes = nodes; b.nodeTypes = nodeTypes; b.nodeMaterials = nodeMaterials;
  b.nodePieces = nodePieces; b.nodeSizes = nodeSizes; b.nodeColliders = nodeColliders;
  b.bonds = bonds; b.built = true;
  return { pack: b.emit({ cameraTarget, cameraDistance }), builder: b };
}

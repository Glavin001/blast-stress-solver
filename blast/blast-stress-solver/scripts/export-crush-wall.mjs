/**
 * The punch-through wall: the scenario where chunk crushing IS the behaviour.
 *
 * A free-standing masonry wall, two chunks thick, anchored at its base. The
 * demo's launcher fires along ±x, so the wall spans the z-y plane and takes
 * every projectile square on its face.
 *
 * The A/B this is built for:
 *
 *   --no-crush   the wall can only come APART. The projectile must knock a
 *                plug of intact chunks out through bond fracture, dumping its
 *                momentum into them; it slows sharply or bounces.
 *
 *   crush on     the material in the projectile's path is comminuted and
 *                REMOVED. The projectile keeps most of its speed and exits
 *                through a hole ringed by fractured-but-intact chunks.
 *
 * The readout is the projectile's own forward speed (frames.csv:
 * projectile_max_forward_speed) plus crushed-mass telemetry -- the difference
 * is measurable before it is visible.
 *
 * Material doctrine as everywhere else: bond area is geometry (0.16 m^2 faces),
 * strength lives in the material table, and the crush cone is DERIVED from the
 * material's own compressive strength, not dialled (see SCENE_PACK_FORMAT.md).
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT = path.resolve(__dirname, '../assets/reference/crush-wall.json');

// ── Geometry ────────────────────────────────────────────────────────────────
const CHUNK = 0.4;              // m, cubic block
const WIDTH = 12;               // chunks along z
const HEIGHT = 8;               // chunks along y
const DEPTH = 2;                // chunks along x (0.8 m of wall to get through)
const DENSITY = 1900;           // kg/m^3, rendered blockwork
const FACE_AREA = CHUNK * CHUNK; // m^2, every bonded face

// ── Materials ───────────────────────────────────────────────────────────────
const DP_SLOPE = 1.2;
const fc = 8e6; // unreinforced blockwork compressive strength
const MATERIALS = [
  {
    name: 'block-wall',
    // Joints tough enough that an ordinary hit cannot cheaply blow a plug out:
    // penetration should have to EARN its hole by comminuting the material.
    compressionElastic: fc, compressionFatal: 3 * fc,
    tensionElastic: 1.5e6, tensionFatal: 4e6,
    shearElastic: 2.0e6, shearFatal: 5e6,
    crush: {
      capPressure: 2.5 * fc,
      cohesion: fc * (1 - DP_SLOPE / 3),
      frictionSlope: DP_SLOPE,
      // Friable, rendered/aerated blockwork. Measured on this wall: at 4e6 a
      // 4.8 t ball at 39 m/s comminutes ~10 chunks and passes through at
      // ~12 m/s; the SAME impact with --no-crush is stopped dead. At 2e7
      // (dense concrete territory) nothing punches through at any speed the
      // pack authors. This is the knob that decides whether a wall is
      // something you can shoot through.
      crushEnergy: 4e6,
      crushViscosity: 2e5,
    },
  },
  {
    // Anchor course. No crush block: a vaporizing footing is never the story.
    name: 'footing-anchor',
    compressionElastic: 24e6, compressionFatal: 240e6,
    tensionElastic: 3e6, tensionFatal: 30e6,
    shearElastic: 4e6, shearFatal: 40e6,
  },
];
const [M_WALL, M_ANCHOR] = [0, 1];

const round = (n) => Math.round(n * 1e5) / 1e5;
const v = (x, y, z) => ({ x: round(x), y: round(y), z: round(z) });

const nodes = [];
const nodeTypes = [];
const nodeSizes = [];
const nodeColliders = [];
const bonds = [];
const index = new Map();
const key = (ix, iy, iz) => `${ix},${iy},${iz}`;

function addNode(role, centre, fixed, material) {
  const volume = CHUNK ** 3;
  const node = {
    centroid: v(...centre),
    mass: fixed ? 0 : round(volume * DENSITY),
    volume: round(volume),
    m: material,
  };
  nodes.push(node);
  nodeTypes.push(role);
  nodeSizes.push(v(CHUNK, CHUNK, CHUNK));
  nodeColliders.push({ kind: 'cuboid', halfExtents: v(CHUNK / 2, CHUNK / 2, CHUNK / 2) });
  return nodes.length - 1;
}

// iy = 0 is the anchor course; 1..HEIGHT are wall rows.
for (let iy = 0; iy <= HEIGHT; ++iy) {
  for (let iz = 0; iz < WIDTH; ++iz) {
    for (let ix = 0; ix < DEPTH; ++ix) {
      const centre = [
        (ix - (DEPTH - 1) / 2) * CHUNK,
        CHUNK / 2 + iy * CHUNK,
        (iz - (WIDTH - 1) / 2) * CHUNK,
      ];
      const anchor = iy === 0;
      index.set(key(ix, iy, iz),
        addNode(anchor ? 'foundation' : 'wall', centre, anchor, anchor ? M_ANCHOR : M_WALL));
    }
  }
}

function addBond(a, b, normal, material) {
  const na = nodes[a].centroid;
  const nb = nodes[b].centroid;
  bonds.push({
    node0: a,
    node1: b,
    centroid: v((na.x + nb.x) / 2, (na.y + nb.y) / 2, (na.z + nb.z) / 2),
    normal: v(...normal),
    area: round(FACE_AREA),
    m: material,
  });
}

for (let iy = 0; iy <= HEIGHT; ++iy) {
  for (let iz = 0; iz < WIDTH; ++iz) {
    for (let ix = 0; ix < DEPTH; ++ix) {
      const here = index.get(key(ix, iy, iz));
      // +x (through-thickness), +z (along the wall): wall-wall joints only --
      // the anchor course is bonded to the wall, not to itself.
      if (iy > 0) {
        if (ix + 1 < DEPTH) addBond(here, index.get(key(ix + 1, iy, iz)), [1, 0, 0], M_WALL);
        if (iz + 1 < WIDTH) addBond(here, index.get(key(ix, iy, iz + 1)), [0, 0, 1], M_WALL);
      }
      // +y: course-to-course; the bottom joint runs into the anchors and is
      // deliberately the strong link so the wall fails in its field, not by
      // tipping off its base.
      if (iy + 1 <= HEIGHT) {
        addBond(here, index.get(key(ix, iy + 1, iz)), [0, 1, 0], iy === 0 ? M_ANCHOR : M_WALL);
      }
    }
  }
}

const pack = {
  version: 3,
  key: 'crush-wall',
  title: 'Punch-through wall (12x8x2 blockwork)',
  defaults: {
    camera: { target: v(0, 1.8, 0), distance: 14 },
    // The measured stopped-vs-through window for this wall: heavy and fast
    // enough that crushing lets it through, slow enough that bond fracture
    // alone cannot. See the A/B in the crushEnergy comment.
    projectile: { radius: 0.45, mass: 4800, speed: 35, ttlMs: 6000 },
    solver: { gravity: -9.81, materials: MATERIALS },
    physics: { friction: 0.3, restitution: 0, contactForceScale: 1, skipSingleBodies: false },
  },
  scenario: { nodeTypes, nodes, bonds, nodeSizes, nodeColliders },
  nodeMeshes: [],
};

await mkdir(path.dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, `${JSON.stringify(pack, null, 1)}\n`, 'utf8');
const mass = nodes.reduce((sum, n) => sum + n.mass, 0);
console.log(`wrote ${OUTPUT}`);
console.log(`nodes=${nodes.length} bonds=${bonds.length} mass=${Math.round(mass)}kg`);

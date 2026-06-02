/**
 * Generate the shared "high-rise" scene-pack JSON consumed by BOTH the web demo
 * and the Bevy Rust demo (single source of truth).
 *
 * Run after `npm run build:ts` (imports the compiled composer from ../dist).
 *   node scripts/export-high-rise.mjs
 *
 * Notes:
 * - Uses the pure-JS proximity bonder (no WASM / Emscripten required).
 * - Every chunk is a box, so `nodeMeshes` is omitted entirely; both loaders derive a
 *   box mesh from `nodeSizes`/`nodeColliders` at load time. Combined with compact JSON
 *   and rounded floats this keeps the (git-ignored, generated) artifact small.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildHighRiseScenario } from '../dist/scenarios.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Canonical output for the Rust/Bevy demo + Rust headless tests, plus a copy in the
// library's own dist/ so the web demo can fetch it via the /vendor/blast-stress-solver
// serve route. Both locations are git-ignored and regenerated from this script.
const OUTPUT_PATHS = [
  path.resolve(__dirname, '../../blast-stress-demo-rs/assets/scenes/high-rise.json'),
  path.resolve(__dirname, '../dist/high-rise.json'),
];

const TITLE = 'High-Rise Apartment';

// Scene defaults shared by both runtimes. `solver.limits` is the optional schema
// extension carrying realistic, DECOUPLED concrete limits (Pa): strong in
// compression (holds gravity), weak in tension/shear (cracks locally under impact),
// with a wide elastic->fatal band for ductility (non-glass behavior).
const DEFAULTS = {
  camera: { target: { x: 0, y: 12, z: 0 }, distance: 52 },
  projectile: { radius: 0.6, mass: 2500, speed: 18, ttlMs: 8000 },
  solver: {
    gravity: -9.81,
    materialScale: 1e10,
    limits: {
      compressionElastic: 12e6,
      compressionFatal: 30e6,
      tensionElastic: 1.2e6,
      tensionFatal: 3e6,
      shearElastic: 1.6e6,
      shearFatal: 4e6,
    },
  },
  physics: {
    debrisCollisionMode: 'all',
    friction: 0.25,
    restitution: 0.0,
    contactForceScale: 30,
    skipSingleBodies: false,
  },
  // CONTACT-DAMAGE layer (per-chunk health + splash) — the local-destruction knob.
  // OFF BY DEFAULT: the web demo exposes a "Custom damage system" toggle that flips
  // this on at runtime, so the default experience is the pure stress-solver behavior.
  // When enabled, this is what makes a wrecking ball punch a *local* hole instead of
  // collapsing the whole structure. Impacts deposit per-chunk health damage (with a
  // splash AOE); when a chunk's health hits zero it detaches, and the stress solver
  // simply redistributes gravity around the missing nodes (which it does robustly).
  // Impacts are decoupled from the stress solver via the core's damageContactStressScale
  // (default 0 when damage is enabled), so a large contact force no longer drives the
  // global stress cascade. Per-chunk health = strengthPerVolume * volume, so the thin
  // drywall infill blows out easily while the larger concrete columns/slabs resist.
  // Params tuned against the full Rapier pipeline: a realistic ball blows out ~a panel
  // (comDrop ~0.02 m, building stands); heavier/faster impacts make bigger holes and
  // begin chipping columns. See highRise.damage.rapier.test.ts.
  damage: {
    enabled: false,
    strengthPerVolume: 200,
    kImpact: 0.15,
    contactDamageScale: 1,
    minImpulseThreshold: 5,
    internalMinImpulseThreshold: 8,
    splashRadius: 3.0,
    splashFalloffExp: 1.5,
  },
  optimization: {
    smallBodyDampingMode: 'always',
    debrisCleanupMode: 'always',
    debrisTtlMs: 10000,
    maxCollidersForDebris: 3,
  },
};

const round = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * 1e5) / 1e5;
};
const roundVec = (v) => ({ x: round(v.x), y: round(v.y), z: round(v.z) });

function serializeScenePack(scenario) {
  const fragmentSizes = scenario.parameters?.fragmentSizes ?? [];
  const nodeTypes = scenario.parameters?.highRise?.fragmentTypes ?? [];

  return {
    version: 1,
    key: 'high_rise',
    title: TITLE,
    defaults: DEFAULTS,
    scenario: {
      // Per-node structural role (column/slab/infill/foundation). Ignored by the
      // generic Bevy scene-pack loader (serde skips unknown fields), used by the
      // Rust headless tests/sweeps and observability to classify bonds by type.
      nodeTypes,
      nodes: scenario.nodes.map((node) => ({
        centroid: roundVec(node.centroid),
        mass: round(node.mass),
        volume: round(node.volume),
      })),
      bonds: scenario.bonds.map((bond) => ({
        node0: bond.node0,
        node1: bond.node1,
        centroid: roundVec(bond.centroid),
        normal: roundVec(bond.normal),
        area: round(bond.area),
      })),
      nodeSizes: fragmentSizes.map((size) => roundVec(size)),
      nodeColliders: fragmentSizes.map((size) => ({
        kind: 'cuboid',
        halfExtents: { x: round(size.x * 0.5), y: round(size.y * 0.5), z: round(size.z * 0.5) },
      })),
    },
    // Omitted on purpose: every chunk is a box, so loaders derive meshes from
    // nodeSizes. Kept as an empty array so the schema shape is stable.
    nodeMeshes: [],
  };
}

function summarize(scenario) {
  const types = scenario.parameters?.highRise?.fragmentTypes ?? [];
  const byType = {};
  for (const t of types) byType[t ?? 'unknown'] = (byType[t ?? 'unknown'] ?? 0) + 1;

  // Count bonds by the unordered type pair.
  const bondByPair = {};
  for (const b of scenario.bonds) {
    const a = types[b.node0] ?? '?';
    const c = types[b.node1] ?? '?';
    const key = [a, c].sort().join('~');
    bondByPair[key] = (bondByPair[key] ?? 0) + 1;
  }

  const supports = scenario.nodes.filter((n) => n.mass === 0).length;
  let totalMass = 0;
  for (const n of scenario.nodes) totalMass += n.mass;
  return { byType, bondByPair, supports, totalMass };
}

async function main() {
  const scenario = buildHighRiseScenario();
  const pack = serializeScenePack(scenario);

  // Compact JSON: this is a generated, git-ignored artifact, so favor small size
  // and fast (re)generation over human-readable diffs.
  const payload = `${JSON.stringify(pack)}\n`;
  for (const out of OUTPUT_PATHS) {
    await mkdir(path.dirname(out), { recursive: true });
    await writeFile(out, payload, 'utf8');
  }

  const { byType, bondByPair, supports, totalMass } = summarize(scenario);
  console.log(`wrote ${OUTPUT_PATHS.join(' and ')}`);
  console.log(
    `nodes=${pack.scenario.nodes.length} bonds=${pack.scenario.bonds.length} ` +
      `supports=${supports} totalMass=${Math.round(totalMass)}kg`,
  );
  console.log('nodes by type:', JSON.stringify(byType));
  console.log('bonds by pair:', JSON.stringify(bondByPair));

  for (const geometry of scenario.parameters?.fragmentGeometries ?? []) {
    geometry.dispose?.();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

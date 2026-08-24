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

const floorCount = Number.parseInt(process.env.HIGH_RISE_FLOORS ?? '9', 10);
if (!Number.isInteger(floorCount) || floorCount < 1 || floorCount > 40) {
  throw new Error('HIGH_RISE_FLOORS must be an integer between 1 and 40');
}
// Local-damage profile: weaken wall/infill attachments and harden foundation
// anchors so projectiles tear facade panels instead of shoving the whole tower.
const localDamage = process.env.HIGH_RISE_LOCAL_DAMAGE === '1';
const outputStem =
  floorCount === 9 && !localDamage
    ? 'high-rise'
    : `high-rise-${floorCount}f${localDamage ? '-local' : ''}`;
// Local-damage authoring = civil load-path model, not collapse choreography.
//
// Real buildings stand because connection *capacity* exceeds gravity *demand*
// along a designed path (slab → column → footing). They collapse when damage
// removes a capacity link and demand redistributes onto neighbors that then
// fail — progressive collapse.
//
// A multiplier here scales bond AREA, and bond area is the denominator of
// stress (pressure = force / area) as well as the bond's damage pool (Blast
// treats bond health as remaining contact area). So a multiplier is a claim
// about *material* capacity relative to plain concrete, and it has to stay
// within a range a material can actually justify:
//
//   > 1   rebar continuity / monolithic pour (frame joints)
//   ~ 1   plain concrete, geometric truth
//   < 1   drywall + light clip attachment (non-structural facade)
//
// These deliberately reuse the library's own DEFAULT_HIGH_RISE_MULTIPLIERS
// band (foundation ~24, frame 7–14, drywall 0.015–0.03) instead of inventing a
// local table. An earlier revision of this file used 2e5–1e6 here, which put
// the frame ~7 orders of magnitude below its elastic limit: the skeleton could
// not break under *any* impulse the sim can produce, so "partial destruction"
// was satisfied by an indestructible structure rather than by a working load
// path, and the demo needed a 650 t projectile and a 99x contact-force
// multiplier to move anything at all. Keep these O(10).
// Calibrated against a stated physical target, measured by the demo's
// "gravity load path" report (peak stress / elastic limit under self-weight):
//
//   base anchor   safety factor ~30-70   (never the failure point)
//   frame joints  safety factor ~5-15    (textbook structural design margin)
//   facade        safety factor ~2-4     (the deliberate weak link)
//
// Every value sits within an order of magnitude of geometric truth, so a
// multiplier is a claim about rebar continuity or a cladding track rather than
// a way to make a joint unbreakable. Re-derive them by running any grid with
// tiny projectiles and reading the report — do not guess.
const LOCAL_DAMAGE_MULTIPLIERS = {
  // Footing is a genuinely beefier section than the column it receives.
  foundationColumn: 2.0,
  foundationSkeleton: 1.5,
  // Reinforced concrete: rebar carried through the joint roughly doubles the
  // capacity of the plain-concrete section. Strong, and still breakable.
  columnColumn: 2.0,
  columnSlab: 2.0,
  slabSlab: 1.5,
  // Non-structural facade. These are BELOW geometric truth on purpose: the
  // solver has one global material (concrete), so a drywall panel's weaker
  // material is expressed as a smaller effective bonded area. The hard
  // constraint is that a panel must still hang off its own track under
  // gravity — an earlier revision used 0.015-0.03 here and the cladding tore
  // itself off the frame during warmup (column~infill safety factor 0.03).
  infillInfill: 1.0, // taped panel-to-panel seam over the full panel edge
  slabInfill: 0.09, // panel head/base track into the slab band
  frameInfill: 1.5, // full-height track against a column ~0.5 m^2
};

// Canonical output for the Rust/Bevy demo + Rust headless tests, plus a copy in the
// library's own dist/ so the web demo can fetch it via the /vendor/blast-stress-solver
// serve route. Both locations are git-ignored and regenerated from this script.
const OUTPUT_PATHS = [
  path.resolve(__dirname, `../../blast-stress-demo-rs/assets/scenes/${outputStem}.json`),
  path.resolve(__dirname, `../dist/${outputStem}.json`),
];

const TITLE = `High-Rise Apartment ${floorCount}F`;

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
    // Unity: the engine→solver path already converts a solved contact impulse
    // (N·s) into a force (N) by dividing by dt, so the physically correct
    // transfer is 1.0. Anything above that is an unexplained gain that has to
    // be cancelled somewhere else (weaker limits, heavier projectiles) and
    // destroys the correspondence between the sim and its own material model.
    contactForceScale: 1,
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
    contactDamageScale: 2,
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
  const scenario = buildHighRiseScenario({
    floorCount,
    ...(localDamage ? { multipliers: LOCAL_DAMAGE_MULTIPLIERS } : {}),
  });
  const pack = serializeScenePack(scenario);

  if (localDamage) {
    // World-fix only true ground anchors. Columns/slabs keep authored mass so
    // gravity demand flows through the stress graph: remove a column → neighbors
    // pick up load → joints that exceed capacity fail. Pinning columns to mass 0
    // made them immortal kinematics (balls bounced; towers never crumbled).
    const types = pack.scenario.nodeTypes ?? [];
    for (let i = 0; i < pack.scenario.nodes.length; ++i) {
      if (types[i] === 'foundation') {
        pack.scenario.nodes[i].mass = 0;
      }
    }
  }

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

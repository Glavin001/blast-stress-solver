#!/usr/bin/env node
/**
 * High-rise wrecking-ball parameter sweep (full Rapier pipeline, JS side).
 *
 * Loads the shared high-rise scene pack, fires a scripted wrecking ball across a grid
 * of {strength scale, ball mass, ball speed}, and reports the destruction response
 * (bonds broken, peak rigid bodies, bond survival %, COM drop) as JSON — the JS
 * counterpart to the Rust `high_rise_sweep` example, exercising real contacts +
 * splash damage rather than a synthetic impulse.
 *
 *   npm run build        # produces dist/ incl. WASM and high-rise.json
 *   node scripts/sweep-high-rise.mjs
 *
 * Requires dist/stress_solver.wasm (run `npm run build` first).
 */
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, readFileSync } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, '../dist');

if (!existsSync(resolve(distDir, 'stress_solver.wasm'))) {
  console.error('ERROR: dist/stress_solver.wasm not found. Run: npm run build');
  process.exit(1);
}
const scenePath = resolve(distDir, 'high-rise.json');
if (!existsSync(scenePath)) {
  console.error('ERROR: dist/high-rise.json not found. Run: npm run build:ts');
  process.exit(1);
}

const { buildDestructibleCore, parseScenePackJson, createBondBreakRecorder } = await import(
  resolve(distDir, 'rapier.js')
);

const pack = parseScenePackJson(readFileSync(scenePath, 'utf8'));
const { scenario, defaults, nodeTypes } = pack;

// Pick a target node nearest a point, optionally filtered by structural type.
function nearestNode(point, want) {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < scenario.nodes.length; i++) {
    if (want && nodeTypes[i] !== want) continue;
    const c = scenario.nodes[i].centroid;
    const d = (c.x - point.x) ** 2 + (c.y - point.y) ** 2 + (c.z - point.z) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// Bounds.
let lo = { x: Infinity, y: Infinity, z: Infinity };
let hi = { x: -Infinity, y: -Infinity, z: -Infinity };
for (const n of scenario.nodes) {
  lo = { x: Math.min(lo.x, n.centroid.x), y: Math.min(lo.y, n.centroid.y), z: Math.min(lo.z, n.centroid.z) };
  hi = { x: Math.max(hi.x, n.centroid.x), y: Math.max(hi.y, n.centroid.y), z: Math.max(hi.z, n.centroid.z) };
}
const midY = (lo.y + hi.y) * 0.5;

const targets = {
  infill_mid: nearestNode({ x: 0, y: midY, z: lo.z }, 'infill'),
  column_low: nearestNode({ x: 0, y: lo.y + 4, z: lo.z }, 'column'),
};

const strengthScales = [0.5, 1.0, 2.0];
const masses = [800, 2500, 4000];
const speeds = [12, 18, 25];
const FRAMES = 240;
const DT = 1 / 60;

function scaledSettings(scale) {
  const s = defaults.solverSettings;
  if (!s) return undefined;
  return Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v * scale]));
}

async function runCell(targetName, node, strengthScale, mass, speed) {
  const core = await buildDestructibleCore({
    scenario,
    gravity: defaults.gravity,
    materialScale: defaults.materialScale,
    solverSettings: scaledSettings(strengthScale),
    friction: defaults.physics.friction,
    restitution: defaults.physics.restitution,
    contactForceScale: defaults.physics.contactForceScale,
    damage: { enabled: false },
  });
  const recorder = createBondBreakRecorder(core);
  const initialBonds = core.getActiveBondsCount();

  // Settle one frame, then fire the ball from outside the -Z face toward the target.
  core.step(DT);
  const t = scenario.nodes[node].centroid;
  const start = { x: t.x, y: t.y, z: t.z - 8 };
  core.enqueueProjectile({
    position: start,
    velocity: { x: 0, y: 0, z: speed },
    radius: defaults.projectile.radius,
    mass,
    ttl: defaults.projectile.ttlMs,
  });

  for (let f = 0; f < FRAMES; f++) {
    core.step(DT);
    recorder.sample(f);
  }
  const finalBonds = core.getActiveBondsCount();
  const result = {
    target: targetName,
    strengthScale,
    mass,
    speed,
    bondsBroken: recorder.totalBondsBroken(),
    bondSurvivalPct: +((finalBonds / Math.max(1, initialBonds)) * 100).toFixed(1),
    peakRigidBodies: recorder.peakRigidBodies(),
    comDrop: +recorder.comDrop().toFixed(2),
  };
  core.dispose();
  return result;
}

const rows = [];
for (const [targetName, node] of Object.entries(targets)) {
  for (const strengthScale of strengthScales) {
    for (const mass of masses) {
      for (const speed of speeds) {
        rows.push(await runCell(targetName, node, strengthScale, mass, speed));
      }
    }
  }
}

console.log(
  JSON.stringify(
    { scene: 'high-rise', nodes: scenario.nodes.length, bonds: scenario.bonds.length, rows },
    null,
    2,
  ),
);

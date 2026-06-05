#!/usr/bin/env node
/**
 * Destruction-driven CGNR benchmark.
 *
 * Loads a session-recording's scenario (e.g. mini-city, 8152 nodes / 21769
 * bonds) and drives the solver through actual progressive destruction:
 * gravity + periodic impulse "projectile" impacts on random surface nodes
 * + applyFractureCommands.  Reports solver-only timing per frame so the
 * kernel-path SIMD comparison (scalar autovec / EMCC_USE_SIMD=1 AVX
 * intrinsics / EMCC_USE_WASM_SIMD=1 direct v128) is measured on the regime
 * where it matters: heavy graphs, real island changes, real fracture work.
 *
 * Usage:
 *   node scripts/bench-destroy.mjs <recording.json[.gz]>
 *     [--frames=N] [--impacts=N] [--impact-every=N] [--impact-force=F] [--seed=S]
 *
 * Defaults: 240 frames, an impact every 4 frames, impulse 8000, seed=1.
 *
 * Reports:
 *   - solverUpdate per-frame stats (gravity inject + addForce + WASM solve +
 *     stress error)
 *   - WASM solverSolve isolated (timing around the C++ ext_stress_solver_update)
 *   - bonds remaining / actor count over time
 *
 * Picks up the current dist/, so switch builds between runs to A/B kernels.
 */
import { existsSync, createReadStream, readFileSync } from 'node:fs';
import { createGunzip } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, '../dist');

const args = process.argv.slice(2);
const recordingPath = args.find((a) => !a.startsWith('--'));
const getArg = (k, fallback) => {
  const a = args.find((x) => x.startsWith(`--${k}=`));
  return a ? a.split('=')[1] : fallback;
};
const FRAMES = parseInt(getArg('frames', '240'), 10);
const IMPACT_EVERY = parseInt(getArg('impact-every', '4'), 10);
const IMPACT_FORCE = parseFloat(getArg('impact-force', '8000'));
const IMPACTS_MAX = parseInt(getArg('impacts', '60'), 10);
const SEED = parseInt(getArg('seed', '1'), 10);

if (!recordingPath) {
  console.error('Usage: node scripts/bench-destroy.mjs <recording.json[.gz]> [--frames=N] [--impacts=N] [--impact-every=N] [--impact-force=F] [--seed=S]');
  process.exit(1);
}
if (!existsSync(resolve(distDir, 'stress_solver.wasm'))) {
  console.error('ERROR: dist/stress_solver.wasm not found. Run: npm run build');
  process.exit(1);
}

async function readRecording(p) {
  if (p.endsWith('.gz')) {
    return await new Promise((res, rej) => {
      const chunks = [];
      createReadStream(p).pipe(createGunzip())
        .on('data', (c) => chunks.push(c))
        .on('end', () => res(JSON.parse(Buffer.concat(chunks).toString('utf8'))))
        .on('error', rej);
    });
  }
  return JSON.parse(readFileSync(p, 'utf8'));
}

// Tiny deterministic PRNG so the impact pattern is reproducible across kernel
// paths (so A/B numbers compare the same trajectory of bond breaks).
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const recording = await readRecording(recordingPath);
const scenario = recording.scenario;
console.log(`Destruction bench`);
console.log(`  recording: ${recordingPath}`);
console.log(`  scenario:  ${scenario.nodes.length} nodes, ${scenario.bonds.length} bonds`);
console.log(`  config:    ${FRAMES} frames, impact every ${IMPACT_EVERY} (max ${IMPACTS_MAX}), force ${IMPACT_FORCE}, seed ${SEED}`);

const origDebug = console.debug, origWarn = console.warn;
console.debug = () => {};
console.warn = () => {};

const { loadStressSolver } = await import(resolve(distDir, 'index.js'));
const rt = await loadStressSolver();
console.debug = origDebug;
console.warn = origWarn;

const usingSimd = rt.module.ccall('stress_processor_using_simd', 'number', [], []);
console.log(`  WASM:      s_use_simd=${Boolean(usingSimd)}`);

// Sensible defaults if the recording's parameters don't carry settings.
const settings = scenario.parameters?.settings ?? {
  maxSolverIterationsPerFrame: 25,
  graphReductionLevel: 0,
  compressionElasticLimit: 1,
  compressionFatalLimit: 2,
  tensionElasticLimit: -1,
  tensionFatalLimit: -1,
  shearElasticLimit: -1,
  shearFatalLimit: -1
};

const solver = rt.createExtSolver({
  nodes: scenario.nodes,
  bonds: scenario.bonds,
  settings
});
solver.setIslandAware(true);
solver.setSkipSettled(true);

const gravity = { x: 0, y: recording.meta?.config?.solver?.gravity ?? -9.81, z: 0 };

// Build a list of candidate impact targets: nodes whose mass is non-zero
// (skip foundation/support nodes) and which sit in the upper half of the
// city, where projectiles would actually land.
const ys = scenario.nodes.map((n) => n.centroid?.y ?? 0);
const yMin = Math.min(...ys);
const yMax = Math.max(...ys);
const yMid = yMin + (yMax - yMin) * 0.3;
const candidates = [];
for (let i = 0; i < scenario.nodes.length; i++) {
  const n = scenario.nodes[i];
  if ((n.mass ?? 0) > 0 && (n.centroid?.y ?? 0) >= yMid) candidates.push(i);
}
console.log(`  impact targets: ${candidates.length} (mass>0, y>=${yMid.toFixed(1)})`);

const rng = mulberry32(SEED);

// Warmup — 3 gravity-only frames so warm-start state is established.
for (let i = 0; i < 3; i++) {
  solver.addGravity(gravity);
  solver.update();
}

const solveSamples = [];
let impactsApplied = 0;
let totalFractured = 0;

for (let f = 0; f < FRAMES; f++) {
  // Gravity every frame.
  solver.addGravity(gravity);

  // Periodic projectile-style impulses.
  if (f % IMPACT_EVERY === 0 && impactsApplied < IMPACTS_MAX) {
    const pick = candidates[Math.floor(rng() * candidates.length)];
    // Force pointing down + slightly sideways, applied at the node centroid.
    // (No torque — the solver injects it as a centroidal force.)
    const force = { x: (rng() - 0.5) * IMPACT_FORCE, y: -IMPACT_FORCE, z: (rng() - 0.5) * IMPACT_FORCE };
    solver.addForce(pick, { x: 0, y: 0, z: 0 }, force, 0);
    impactsApplied++;
  }

  // Time the WASM solve.
  const t0 = performance.now();
  solver.update();
  const t1 = performance.now();
  solveSamples.push(t1 - t0);

  // Apply fractures (this is what makes the graph change between frames).
  const fractures = solver.generateFractureCommandsPerActor();
  if (fractures.length > 0) {
    const n = solver.applyFractureCommands(fractures);
    let cnt = 0;
    for (const ev of n) cnt += ev.children?.length ?? 0;
    totalFractured += cnt;
  }
}

solver.destroy();

const sorted = [...solveSamples].sort((a, b) => a - b);
const mean = solveSamples.reduce((a, b) => a + b, 0) / solveSamples.length;
const p50 = sorted[Math.floor(sorted.length * 0.5)];
const p95 = sorted[Math.floor(sorted.length * 0.95)];
const max = sorted[sorted.length - 1];
const sum = solveSamples.reduce((a, b) => a + b, 0);

console.log(`\nsolver.update() over ${FRAMES} frames (with ${impactsApplied} projectile impacts, ${totalFractured} split events):`);
console.log(`  mean: ${mean.toFixed(3)} ms`);
console.log(`  p50:  ${p50.toFixed(3)} ms`);
console.log(`  p95:  ${p95.toFixed(3)} ms`);
console.log(`  max:  ${max.toFixed(3)} ms`);
console.log(`  total: ${sum.toFixed(1)} ms`);

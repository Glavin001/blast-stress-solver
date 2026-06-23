#!/usr/bin/env node
/**
 * Replay the CGNR solve from a session recording — isolated, no Rapier, no JS-side
 * contact-inject, no fracture loop.  Build an ExtStressSolver from the recording's
 * scenario, run gravity-only `update()` for N frames, report the per-frame solverSolve
 * time mean / p50 / p95 / max.
 *
 * Usage:
 *   node scripts/bench-replay.mjs <recording.json[.gz]> [--frames N]
 *
 * Picks up the current dist/ — so switch builds between runs (default / EMCC_USE_SIMD=1
 * / future EMCC_USE_WASM_SIMD=1) to A/B the solver hot loop directly against the
 * mini-city scenario.
 */
import { existsSync, readFileSync, createReadStream } from 'node:fs';
import { createGunzip } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, '../dist');

const args = process.argv.slice(2);
const recordingPath = args.find((a) => !a.startsWith('--'));
const framesArg = args.find((a) => a.startsWith('--frames='));
const FRAMES = framesArg ? parseInt(framesArg.split('=')[1], 10) : 60;

if (!recordingPath) {
  console.error('Usage: node scripts/bench-replay.mjs <recording.json[.gz]> [--frames N]');
  process.exit(1);
}

if (!existsSync(resolve(distDir, 'stress_solver.wasm'))) {
  console.error('ERROR: dist/stress_solver.wasm not found. Run: npm run build');
  process.exit(1);
}

async function readRecording(path) {
  if (path.endsWith('.gz')) {
    return await new Promise((res, rej) => {
      const chunks = [];
      createReadStream(path).pipe(createGunzip())
        .on('data', (c) => chunks.push(c))
        .on('end', () => res(JSON.parse(Buffer.concat(chunks).toString('utf8'))))
        .on('error', rej);
    });
  }
  return JSON.parse(readFileSync(path, 'utf8'));
}

const recording = await readRecording(recordingPath);
const scenario = recording.scenario;
console.log(`Recording: ${recordingPath}`);
console.log(`  demo: ${recording.meta?.demo ?? 'unknown'}`);
console.log(`  nodes: ${scenario.nodes.length}, bonds: ${scenario.bonds.length}`);

// Suppress noisy stress-solver logs.
const origDebug = console.debug;
const origWarn = console.warn;
console.debug = () => {};
console.warn = () => {};

const { loadStressSolver } = await import(resolve(distDir, 'index.js'));
const rt = await loadStressSolver();
console.debug = origDebug;
console.warn = origWarn;

// Probe which kernel path the WASM was compiled with (the
// `stress_processor_using_simd` export reflects `StressProcessor::s_use_simd`).
const usingSimd = rt.module.ccall('stress_processor_using_simd', 'number', [], []);
console.log(`  WASM s_use_simd: ${Boolean(usingSimd)}`);

// Build the same ExtStressSolver the recording would have used. The recorded
// settings ride on scenario.parameters; fall back to the demo's defaults if not.
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

// Recording metadata stores its world gravity here.
const g = recording.meta?.config?.solver?.gravity ?? -9.81;
const gravity = { x: 0, y: g, z: 0 };

// Warmup — first solve is cold (cache cold, no warm-start state).
for (let i = 0; i < 3; i++) {
  solver.addGravity(gravity);
  solver.update();
}

const samples = [];
for (let f = 0; f < FRAMES; f++) {
  solver.addGravity(gravity);
  const t0 = performance.now();
  solver.update();
  const t1 = performance.now();
  samples.push(t1 - t0);
}

solver.destroy();

const sorted = [...samples].sort((a, b) => a - b);
const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
const p50 = sorted[Math.floor(sorted.length * 0.5)];
const p95 = sorted[Math.floor(sorted.length * 0.95)];
const max = sorted[sorted.length - 1];

console.log(`\nsolver.update() over ${FRAMES} frames:`);
console.log(`  mean: ${mean.toFixed(3)} ms`);
console.log(`  p50:  ${p50.toFixed(3)} ms`);
console.log(`  p95:  ${p95.toFixed(3)} ms`);
console.log(`  max:  ${max.toFixed(3)} ms`);

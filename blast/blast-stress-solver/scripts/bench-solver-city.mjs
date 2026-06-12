// Pure WASM stress-solver micro-benchmark at city scale (no Rapier, no rendering).
//
// Builds a grid-of-towers bond graph (~mini-city scale: default 36 towers ×
// 4×4×16 fragments = 9216 nodes / 22464 bonds), then times `solver.update()`
// under the two regimes that matter for a real-time game:
//
//   steady  — converged structure, gravity refreshed every frame (idle city)
//   excited — one tower poked with an impulse every frame (localized action)
//
// and A/Bs whole-graph vs island-aware + skip-settled solving. This isolates
// the question "what does ONE localized impact cost the stress solve at city
// scale?" from Rapier, contact drain, and rendering.
//
// Usage:
//   npm run build                  # WASM + TS first
//   node scripts/bench-solver-city.mjs
//   TOWERS=64 node scripts/bench-solver-city.mjs   # bigger city
//
// Representative result (Node 22, x64 container, default scalar WASM build):
//   whole-graph:   steady ≈ 1.4 ms   excited ≈ 13.1 ms (p95 ≈ 16 ms)
//   island-aware:  steady ≈ 1.0 ms   excited ≈ 1.35 ms (35/36 islands skipped)
// i.e. island-aware solving is ~10× cheaper the moment anything happens,
// because only the struck tower's island re-solves.
import { loadStressSolver } from '../dist/index.js';

const TOWERS = Number(process.env.TOWERS ?? 36);
const W = 4, D = 4, H = 16; // fragments per tower: W*D*H
const FRAMES = Number(process.env.FRAMES ?? 120);
const label = process.env.LABEL ?? 'solver-city';

function buildCity() {
  const nodes = [];
  const bonds = [];
  const side = Math.ceil(Math.sqrt(TOWERS));
  for (let t = 0; t < TOWERS; t++) {
    const ox = (t % side) * 20, oz = Math.floor(t / side) * 20;
    const base = nodes.length;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        for (let z = 0; z < D; z++) {
          nodes.push({
            centroid: { x: ox + x, y: y + 0.5, z: oz + z },
            mass: y === 0 ? 0 : 100, // ground row = supports (mass 0)
            volume: 1,
          });
        }
    const idx = (x, y, z) => base + y * W * D + x * D + z;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        for (let z = 0; z < D; z++) {
          if (x + 1 < W) bonds.push({ node0: idx(x, y, z), node1: idx(x + 1, y, z), centroid: { x: ox + x + 0.5, y: y + 0.5, z: oz + z }, normal: { x: 1, y: 0, z: 0 }, area: 1 });
          if (z + 1 < D) bonds.push({ node0: idx(x, y, z), node1: idx(x, y, z + 1), centroid: { x: ox + x, y: y + 0.5, z: oz + z + 0.5 }, normal: { x: 0, y: 0, z: 1 }, area: 1 });
          if (y + 1 < H) bonds.push({ node0: idx(x, y, z), node1: idx(x, y + 1, z), centroid: { x: ox + x, y: y + 1, z: oz + z }, normal: { x: 0, y: 1, z: 0 }, area: 1 });
        }
  }
  return { nodes, bonds };
}

function stats(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return { mean, p50: s[Math.floor(s.length * 0.5)], p95: s[Math.floor(s.length * 0.95)], max: s[s.length - 1] };
}
const round = (v) => +v.toFixed(3);

const { nodes, bonds } = buildCity();
const runtime = await loadStressSolver();
const settings = runtime.defaultExtSettings();
settings.maxSolverIterationsPerFrame = 24;
settings.graphReductionLevel = 0;
// Generous limits so nothing fractures during the timing loop — the bench
// isolates solve cost, not fracture handling.
settings.compressionFatalLimit = 1e9; settings.tensionFatalLimit = 1e9; settings.shearFatalLimit = 1e9;
settings.compressionElasticLimit = 1e8; settings.tensionElasticLimit = 1e8; settings.shearElasticLimit = 1e8;

for (const islandAware of [false, true]) {
  const solver = runtime.createExtSolver({ nodes, bonds, settings });
  solver.setIslandAware?.(islandAware);
  solver.setSkipSettled?.(islandAware);

  const g = { x: 0, y: -9.81, z: 0 };
  for (let i = 0; i < 50; i++) { solver.addGravity(g); solver.update(); } // converge

  // (a) steady: gravity refresh + update each frame (settled city)
  let t = [];
  for (let i = 0; i < FRAMES; i++) {
    solver.addGravity(g);
    const t0 = performance.now();
    solver.update();
    t.push(performance.now() - t0);
  }
  const steady = stats(t);

  // (b) excited: poke one tower with an impulse every frame (localized action)
  t = [];
  const pokeNode = Math.floor(nodes.length / 2) + 7;
  for (let i = 0; i < FRAMES; i++) {
    solver.addGravity(g);
    solver.addForce(pokeNode, { x: 0, y: 8, z: 0 }, { x: 4000 * (i % 3 === 0 ? 1 : -1), y: 0, z: 0 }, 'impulse');
    const t0 = performance.now();
    solver.update();
    t.push(performance.now() - t0);
  }
  const excited = stats(t);

  console.log(JSON.stringify({
    label, islandAware,
    nodes: nodes.length, bonds: bonds.length,
    steady: { mean: round(steady.mean), p95: round(steady.p95), max: round(steady.max) },
    excited: { mean: round(excited.mean), p95: round(excited.p95), max: round(excited.max) },
    islandsSkipped: solver.islandsSkipped?.() ?? null,
    islandsTotal: solver.islandsTotal?.() ?? null,
  }));
  solver.destroy();
}

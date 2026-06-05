/**
 * Idle big-city benchmark (headless).
 *
 * Reproduces the mini-city demo's *idle* regime — a large grid of intact,
 * bonded buildings sitting under gravity with NO destruction — and reports
 * where each frame's time goes. The point is to answer: "why does a big city
 * struggle even when nothing is being destroyed and every building is a single
 * intact (fixed) body?"
 *
 * It builds `grid×grid` anchored towers, merges them into ONE scenario exactly
 * like mini-city.ts does (offset centroids + bond indices; disconnected
 * components => one stress island per building), builds the destructible core,
 * lets it settle, then steps it idle while reading the core profiler.
 *
 * Usage: node scripts/idle-city-bench.mjs [grid] [floors] [island:on|off]
 */
import { buildDestructibleCore } from '../dist/rapier.js';
import { buildTowerScenario } from '../dist/scenarios.js';

const GRID = Number(process.argv[2] ?? 10);
const FLOORS = Number(process.argv[3] ?? 3);
const ISLAND = (process.argv[4] ?? 'on') !== 'off';
const LAZY = (process.argv[5] ?? 'off') !== 'off';
const STREET = 9;
const WIDTH = 8;

function mergeScenarios(parts) {
  const nodes = [];
  const bonds = [];
  let base = 0;
  for (const { scenario, offset } of parts) {
    for (const n of scenario.nodes) {
      nodes.push({
        centroid: { x: n.centroid.x + offset.x, y: n.centroid.y + offset.y, z: n.centroid.z + offset.z },
        mass: n.mass,
        volume: n.volume,
      });
    }
    for (const b of scenario.bonds) {
      bonds.push({
        node0: b.node0 + base,
        node1: b.node1 + base,
        centroid: { x: b.centroid.x + offset.x, y: b.centroid.y + offset.y, z: b.centroid.z + offset.z },
        normal: { ...b.normal },
        area: b.area,
      });
    }
    base += scenario.nodes.length;
  }
  return { nodes, bonds, parameters: {} };
}

async function buildCity() {
  // One template per floor count (mini-city caches templates too).
  const template = await buildTowerScenario({ width: WIDTH, depth: WIDTH, floorCount: FLOORS, floorHeight: 3 });
  const cell = WIDTH + STREET;
  const span = (GRID - 1) * cell;
  const half = span / 2;
  const parts = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      parts.push({ scenario: template, offset: { x: -half + c * cell, y: 0, z: -half + r * cell } });
    }
  }
  return { merged: mergeScenarios(parts), perBuildingNodes: template.nodes.length, perBuildingBonds: template.bonds.length };
}

function mean(a) { return a.reduce((s, v) => s + v, 0) / Math.max(1, a.length); }

async function main() {
  const { merged, perBuildingNodes, perBuildingBonds } = await buildCity();
  const buildings = GRID * GRID;
  console.log(`\n=== Idle city bench: ${GRID}×${GRID} = ${buildings} buildings, ${FLOORS} floors, island=${ISLAND ? 'ON' : 'OFF'}, lazy=${LAZY ? 'ON' : 'OFF'} ===`);
  console.log(`scenario: ${merged.nodes.length} nodes, ${merged.bonds.length} bonds (${perBuildingNodes} nodes / ${perBuildingBonds} bonds per building)`);

  // Profiler capture
  let sample = null;
  const core = await buildDestructibleCore({
    scenario: merged,
    gravity: -9.81,
    materialScale: 1e10,
    friction: 0.25,
    restitution: 0,
    contactForceScale: 30,
    debrisCollisionMode: 'all',
    damage: { enabled: false },
    smallBodyDamping: { mode: 'off' },
    debrisCleanup: { mode: 'afterGroundCollision', debrisTtlMs: 8000, maxCollidersForDebris: 2 },
    lazyIntactColliders: LAZY,
  });
  core.setIslandSolver?.({ enabled: ISLAND });
  core.setProfiler?.({ enabled: true, onSample: (s) => { sample = s; } });

  const bodies = core.getRigidBodyCount();
  const colliderCount = (() => { try { return core.world.colliders.len(); } catch { return -1; } })();
  const lz = core.getLazyColliderStats?.();
  console.log(`rapier: ${bodies} rigid bodies, ${colliderCount} world colliders` +
    (lz ? `  | lazy=${lz.enabled} buildings=${lz.buildingCount} dormant=${lz.dormantCount} exploded=${lz.explodedCount}` : ''));

  const dt = 1 / 60;
  // Settle
  for (let f = 0; f < 40; f++) core.step(dt);

  // Measure steady idle: wall-clock of core.step + profiler phase sums.
  const N = 80;
  const stepWall = [];
  const phase = {};
  const PHASES = ['rapierStepMs', 'solverUpdateMs', 'solverGravityInjectMs', 'solverSolveMs', 'contactDrainMs', 'snapshotCaptureMs', 'snapshotRestoreMs', 'fractureMs', 'preStepSweepMs', 'totalMs'];
  for (const p of PHASES) phase[p] = [];
  let islandsSkipped = 0, islandCount = 0, convergedTrue = 0, solveRan = 0;

  for (let f = 0; f < N; f++) {
    const t0 = performance.now();
    core.step(dt);
    stepWall.push(performance.now() - t0);
    if (sample) {
      for (const p of PHASES) phase[p].push(sample[p] ?? 0);
      if ((sample.solverSolveMs ?? 0) > 0) solveRan++;
    }
    const isl = core.getIslandSolverStats?.();
    if (isl) { islandsSkipped += isl.islandsSkipped ?? 0; islandCount = isl.islandCount ?? 0; }
  }

  const wall = mean(stepWall);
  const profiled = PHASES.filter(p => p !== 'totalMs').reduce((s, p) => s + mean(phase[p]), 0);
  // The chunk-transform loop + countRigidBody/stats etc. are NOT separate profiler
  // fields; they live in (totalMs - sum of named phases). Report it explicitly.
  const totalMs = mean(phase.totalMs);
  const unprofiled = Math.max(0, totalMs - profiled);

  console.log(`\n  islands: ${islandCount} (avg skipped/frame: ${(islandsSkipped / N).toFixed(0)}/${islandCount})  solverSolve ran on ${solveRan}/${N} frames`);
  console.log(`\n  per-frame idle cost (mean over ${N} steady frames):`);
  console.log(`    core.step() wall .............. ${wall.toFixed(3)} ms`);
  console.log(`    profiler totalMs .............. ${totalMs.toFixed(3)} ms`);
  console.log(`    ├─ rapierStepMs (world.step) .. ${mean(phase.rapierStepMs).toFixed(3)} ms`);
  console.log(`    ├─ solverUpdateMs (stress) .... ${mean(phase.solverUpdateMs).toFixed(3)} ms`);
  console.log(`    │    ├─ gravityInject ......... ${mean(phase.solverGravityInjectMs).toFixed(3)} ms`);
  console.log(`    │    └─ CGNR solve ............ ${mean(phase.solverSolveMs).toFixed(3)} ms`);
  console.log(`    ├─ contactDrainMs ............. ${mean(phase.contactDrainMs).toFixed(3)} ms`);
  console.log(`    ├─ snapshotCaptureMs .......... ${mean(phase.snapshotCaptureMs).toFixed(3)} ms`);
  console.log(`    ├─ fractureMs ................. ${mean(phase.fractureMs).toFixed(3)} ms`);
  console.log(`    ├─ preStepSweepMs ............. ${mean(phase.preStepSweepMs).toFixed(3)} ms`);
  console.log(`    └─ UNPROFILED (chunk xform loop + stats) ~ ${unprofiled.toFixed(3)} ms`);

  core.dispose?.();
}

main().catch((e) => { console.error(e); process.exit(1); });

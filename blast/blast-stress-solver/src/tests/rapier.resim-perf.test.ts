/**
 * **Resim / rapier perf levers + output-equivalence harness** (requires the full
 * WASM + TS build; skips gracefully if `dist/stress_solver.wasm` is absent).
 *
 * The mini-city recording analysis (see `recording.minicity.regression.test.ts`)
 * showed that on a large, complex scene the real-time cost is dominated by two
 * rapier-side levers:
 *
 *   1. `world.step()` — broad/narrow phase + island constraint solve over *all*
 *      accumulated debris. Grows with rigid-body count; the single biggest phase.
 *   2. **Resimulation** — on a fracture the world is rolled back and stepped a
 *      *second* time with the new topology, ≈doubling that frame.
 *
 * The goal is to make those faster **without changing the simulation output**.
 * That is a two-part problem and this file builds the tooling for both:
 *
 *   • A reproducible **measurement** of each lever (resim marginal cost; rapierStep
 *     scaling with debris count) so a change can be shown to actually help.
 *   • An **output-equivalence harness** (`captureTrajectory` / `compareTrajectories`)
 *     so a change can be *proven faithful* — identical body trajectories — rather
 *     than merely fast. The harness is validated here with a positive control
 *     (determinism → zero divergence) and a negative control (a different impact →
 *     large divergence), so it can be trusted as the safety net for future work.
 *
 * Run: npm run build && npx vitest run src/tests/rapier.resim-perf.test.ts
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeStats, type Stats } from '../rapier/recordingAnalysis';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, '../../dist/stress_solver.wasm');
const runtimeAvailable = existsSync(wasmPath);

// Lazy dist imports — only when the WASM runtime is present. Typed as
// always-defined (matching `rapier.perf.test.ts`); `loadModules()` is awaited
// before any use inside each test.
let buildDestructibleCore: (opts: any) => Promise<any>;
let buildTowerScenario: (opts?: any) => any;
let modulesLoaded = false;

async function loadModules() {
  if (modulesLoaded) return;
  const rapier = await import('../../dist/rapier.js');
  const scenarios = await import('../../dist/scenarios.js');
  buildDestructibleCore = rapier.buildDestructibleCore;
  buildTowerScenario = scenarios.buildTowerScenario;
  modulesLoaded = true;
}

// ── Types ─────────────────────────────────────────────────────────────────────

type ProjectileSpec = {
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  radius?: number;
  mass?: number;
};
type ImpactPlan = { frame: number; projectiles: ProjectileSpec[] };

/** One frame of body transforms, keyed by node index (stable across runs of an
 *  identical scenario). 7 floats per active node: px,py,pz, qx,qy,qz,qw. */
type FrameSnapshot = { nodes: number[]; data: Float64Array };
type Trajectory = FrameSnapshot[];

type RunResult = {
  trajectory: Trajectory;
  samples: any[]; // profiler samples (one per frame)
  setupMs: number;
};

// ── Output-equivalence harness ────────────────────────────────────────────────

/**
 * Drive a freshly-built core for `frames` steps (firing `plan` projectiles), and
 * record every active chunk's world transform each frame plus the profiler sample.
 *
 * Determinism notes: debris cleanup is forced `off` and projectile TTL is huge, so
 * no wall-clock (`Date.now()` / `performance.now()`) path can perturb the
 * simulation state between runs. The core is otherwise built from `coreOpts`.
 */
async function runScenario(
  scenario: any,
  coreOpts: Record<string, any>,
  drive: { frames: number; dt?: number; plan?: ImpactPlan[] },
): Promise<RunResult> {
  const { frames, dt = 1 / 60, plan = [] } = drive;
  const t0 = performance.now();
  const core = await buildDestructibleCore({
    scenario,
    gravity: -9.81,
    materialScale: 1e8,
    resimulateOnFracture: true,
    maxResimulationPasses: 1,
    snapshotMode: 'perBody',
    // Deterministic: never run the Date.now()-based debris eviction.
    debrisCleanup: { mode: 'off' },
    ...coreOpts,
  });
  const setupMs = performance.now() - t0;

  const samples: any[] = [];
  core.setProfiler({ enabled: true, onSample: (s: any) => samples.push({ ...s }) });

  const trajectory: Trajectory = [];
  for (let i = 0; i < frames; i += 1) {
    for (const p of plan) {
      if (p.frame === i) {
        for (const proj of p.projectiles) {
          core.enqueueProjectile({ ttl: 1e9, radius: 0.3, mass: 15000, ...proj });
        }
      }
    }
    core.step(dt);
    trajectory.push(snapshotFrame(core));
  }

  try { core.dispose(); } catch { /* WASM may be mid-teardown */ }
  return { trajectory, samples, setupMs };
}

function snapshotFrame(core: any): FrameSnapshot {
  const rows: Array<[number, number, number, number, number, number, number, number]> = [];
  for (const c of core.chunks) {
    if (!c.active || !c.worldPosition || !c.worldQuaternion) continue;
    const p = c.worldPosition;
    const q = c.worldQuaternion;
    rows.push([c.nodeIndex, p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
  }
  rows.sort((a, b) => a[0] - b[0]); // stable order by node index
  const nodes = rows.map((r) => r[0]);
  const data = new Float64Array(rows.length * 7);
  for (let i = 0; i < rows.length; i += 1) {
    data.set(rows[i].slice(1) as number[], i * 7);
  }
  return { nodes, data };
}

type Divergence = {
  /** Max |Δposition| (metres) over every body/frame the two runs share. */
  maxPosDelta: number;
  /** Max |Δquaternion-component| over every body/frame. */
  maxQuatDelta: number;
  /** First frame at which the active-node *set* differed, or -1 if never. */
  topologyDivergeFrame: number;
  /** First frame with any transform delta above `eps`, or -1 if none. */
  firstDivergeFrame: number;
  framesCompared: number;
};

/** Compare two trajectories body-by-body (matched on node index per frame). */
function compareTrajectories(a: Trajectory, b: Trajectory, eps = 1e-9): Divergence {
  const n = Math.min(a.length, b.length);
  let maxPos = 0, maxQuat = 0, topoDiverge = -1, firstDiverge = -1;
  for (let f = 0; f < n; f += 1) {
    const fa = a[f], fb = b[f];
    // Build node→row maps (robust even if ordering ever differs).
    const mb = new Map<number, number>();
    for (let i = 0; i < fb.nodes.length; i += 1) mb.set(fb.nodes[i], i);
    if (fa.nodes.length !== fb.nodes.length && topoDiverge < 0) topoDiverge = f;
    for (let i = 0; i < fa.nodes.length; i += 1) {
      const j = mb.get(fa.nodes[i]);
      if (j === undefined) { if (topoDiverge < 0) topoDiverge = f; continue; }
      const oa = i * 7, ob = j * 7;
      for (let k = 0; k < 3; k += 1) {
        const d = Math.abs(fa.data[oa + k] - fb.data[ob + k]);
        if (d > maxPos) maxPos = d;
      }
      for (let k = 3; k < 7; k += 1) {
        const d = Math.abs(fa.data[oa + k] - fb.data[ob + k]);
        if (d > maxQuat) maxQuat = d;
      }
      if (firstDiverge < 0 && (maxPos > eps || maxQuat > eps)) firstDiverge = f;
    }
  }
  return {
    maxPosDelta: maxPos,
    maxQuatDelta: maxQuat,
    topologyDivergeFrame: topoDiverge,
    firstDivergeFrame: firstDiverge,
    framesCompared: n,
  };
}

// ── Per-phase timing extraction from live profiler samples ────────────────────

function phaseStats(samples: any[], field: string): Stats {
  return computeStats(samples.map((s) => (s as any)[field] ?? 0));
}
/** mean over frames that actually ran ≥1 resim pass. */
function meanOnResimFrames(samples: any[], field: string): number {
  const xs = samples.filter((s) => (s.resimPasses ?? 0) > 0).map((s) => s[field] ?? 0);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

// ── Scenario presets ──────────────────────────────────────────────────────────

const centerHit = (height: number, mass = 30000): ImpactPlan[] => [{
  frame: 2,
  projectiles: [{
    position: { x: 0, y: height * 0.5, z: 5 },
    velocity: { x: 0, y: 0, z: -45 },
    radius: 0.5,
    mass,
  }],
}];

/** A staggered barrage — like the mini-city meteor storm, it keeps fracturing the
 *  structure across many frames so resimulation fires repeatedly (not just once). */
const bombardment = (height: number, count = 6, interval = 22): ImpactPlan[] =>
  Array.from({ length: count }, (_, i) => ({
    frame: 4 + i * interval,
    projectiles: [{
      position: {
        x: 3 * Math.cos((i / count) * Math.PI * 2),
        y: height * (0.35 + 0.4 * ((i % 3) / 3)),
        z: 4 * Math.sin((i / count) * Math.PI * 2) + 4,
      },
      velocity: {
        x: -18 * Math.cos((i / count) * Math.PI * 2),
        y: -4,
        z: -34,
      },
      radius: 0.45,
      mass: 26000,
    }],
  }));

// ──────────────────────────────────────────────────────────────────────────────

describe.skipIf(!runtimeAvailable)('Resim / rapier perf levers (requires WASM build)', () => {
  // ── A. The output-equivalence harness itself (positive + negative controls) ──
  describe('A. Output-equivalence harness — trusted safety net', () => {
    it('positive control: identical scenario + inputs → ZERO divergence (determinism)', async () => {
      await loadModules();
      const mk = () => buildTowerScenario({ side: 5, stories: 10, totalMass: 4000 });
      const plan = centerHit(5);
      const a = await runScenario(mk(), {}, { frames: 90, plan });
      const b = await runScenario(mk(), {}, { frames: 90, plan });
      const d = compareTrajectories(a.trajectory, b.trajectory);
      // The whole premise of an output-equivalence guard: the same code on the
      // same inputs must reproduce bit-for-bit, so any future delta is signal.
      expect(d.maxPosDelta).toBe(0);
      expect(d.maxQuatDelta).toBe(0);
      expect(d.topologyDivergeFrame).toBe(-1);
      expect(d.framesCompared).toBe(90);
      // Sanity: the scenario actually fractured (else the test proves nothing).
      const lastBodies = a.samples[a.samples.length - 1].rigidBodies;
      expect(lastBodies).toBeGreaterThan(10);
    }, 60_000);

    it('negative control: a different impact → LARGE divergence (harness is sensitive)', async () => {
      await loadModules();
      const mk = () => buildTowerScenario({ side: 5, stories: 10, totalMass: 4000 });
      const a = await runScenario(mk(), {}, { frames: 90, plan: centerHit(5, 30000) });
      const b = await runScenario(mk(), {}, { frames: 90, plan: centerHit(5, 60000) });
      const d = compareTrajectories(a.trajectory, b.trajectory);
      // A heavier projectile must visibly change where the rubble ends up.
      expect(d.maxPosDelta).toBeGreaterThan(0.05);
      expect(d.firstDivergeFrame).toBeGreaterThanOrEqual(0);
    }, 60_000);

    it('profiler attachment does not perturb the simulation output', async () => {
      await loadModules();
      const mk = () => buildTowerScenario({ side: 5, stories: 10, totalMass: 4000 });
      const plan = centerHit(5);
      // With profiler (runScenario always attaches one).
      const withProf = await runScenario(mk(), {}, { frames: 60, plan });
      // Without profiler: drive a core manually, no setProfiler.
      const core = await buildDestructibleCore({
        scenario: mk(), gravity: -9.81, materialScale: 1e8,
        resimulateOnFracture: true, maxResimulationPasses: 1,
        snapshotMode: 'perBody', debrisCleanup: { mode: 'off' },
      });
      const traj: Trajectory = [];
      for (let i = 0; i < 60; i += 1) {
        for (const p of plan) if (p.frame === i) for (const proj of p.projectiles) core.enqueueProjectile({ ttl: 1e9, ...proj });
        core.step(1 / 60);
        traj.push(snapshotFrame(core));
      }
      try { core.dispose(); } catch { /* ignore */ }
      const d = compareTrajectories(withProf.trajectory, traj);
      // Measurement must be free: identical output with and without the profiler.
      expect(d.maxPosDelta).toBe(0);
      expect(d.maxQuatDelta).toBe(0);
    }, 60_000);

    it('deferRedundantResimSnapshot optimization is output-neutral AND cheaper', async () => {
      await loadModules();
      // Lever #2: at maxResimulationPasses=1 (the production resim path) the snapshot
      // re-capture between the rollback and the resim world.step() is never restored,
      // so skipping it must change nothing about the simulation — only the cost.
      const mk = () => buildTowerScenario({ side: 6, stories: 12, totalMass: 5000 });
      const plan = centerHit(6, 45000);
      const frames = 120;

      const optimized = await runScenario(mk(), { deferRedundantResimSnapshot: true }, { frames, plan });
      const legacy = await runScenario(mk(), { deferRedundantResimSnapshot: false }, { frames, plan });

      // ── Faithful: byte-identical trajectories (this is the whole point). ──
      const d = compareTrajectories(optimized.trajectory, legacy.trajectory);
      expect(d.topologyDivergeFrame).toBe(-1);
      expect(d.maxPosDelta).toBe(0);
      expect(d.maxQuatDelta).toBe(0);

      // The optimization must have actually engaged: a fracture → a resim frame.
      const resimFrames = optimized.samples.filter((s) => (s.resimPasses ?? 0) > 0).length;
      expect(resimFrames).toBeGreaterThan(0);

      // ── Cheaper: fewer/cheaper snapshot captures on resim frames. ──
      // Legacy captures twice per resim frame (pre-step + dead re-capture);
      // optimized captures once. Compare snapshot-capture wall-clock on resim frames.
      const capOn = (run: RunResult) =>
        run.samples.filter((s) => (s.resimPasses ?? 0) > 0)
          .reduce((a, s) => a + (s.snapshotCaptureMs ?? 0), 0);
      const optCap = capOn(optimized);
      const legCap = capOn(legacy);
      console.log('\n  [Lever #2: defer redundant resim snapshot]');
      console.log(`    resim frames: ${resimFrames}   trajectory divergence: ${d.maxPosDelta} m (identical)`);
      console.log(`    snapshotCapture on resim frames: optimized ${optCap.toFixed(2)}ms vs legacy ${legCap.toFixed(2)}ms ` +
        `(${legCap > 0 ? ((1 - optCap / legCap) * 100).toFixed(0) : '0'}% less)`);
      // Optimized does strictly less snapshot work (allow equality only if the
      // capture timer is below the clock's resolution on a tiny scene).
      expect(optCap).toBeLessThanOrEqual(legCap);
    }, 90_000);
  });

  // ── B. Resim cost lever (quantify the "second physics pass") ────────────────
  describe('B. Resimulation cost — the systemic second-physics-pass lever', () => {
    it('quantifies resim cost intrinsically (resimMs + rollback) under sustained fracturing', async () => {
      await loadModules();
      // Sustained bombardment keeps fracturing the tower so resim fires on many
      // frames — the regime the mini-city recording captured (≈40% of frames).
      const scenario = buildTowerScenario({ side: 6, stories: 14, totalMass: 6000 });
      const plan = bombardment(7);
      const frames = 160;

      const run = await runScenario(scenario, { maxResimulationPasses: 1 }, { frames, plan });
      const s = run.samples;

      const total = phaseStats(s, 'totalMs');
      const resimFrames = s.filter((x) => (x.resimPasses ?? 0) > 0).length;
      const resimMs = phaseStats(s, 'resimMs');
      const snapCap = phaseStats(s, 'snapshotCaptureMs');
      const snapRes = phaseStats(s, 'snapshotRestoreMs');
      const resimWallShare = (resimMs.sum / total.sum) * 100;
      const rollbackWallShare = ((snapCap.sum + snapRes.sum) / total.sum) * 100;
      const meanResimFrame = meanOnResimFrames(s, 'totalMs');
      const nonResim = s.filter((x) => (x.resimPasses ?? 0) === 0);
      const meanNonResimFrame = nonResim.reduce((a, x) => a + x.totalMs, 0) / Math.max(1, nonResim.length);

      console.log('\n  [Resim lever] medium tower under bombardment, 160 frames');
      console.log(`    frame: mean ${total.mean.toFixed(2)}ms  p95 ${total.p95.toFixed(2)}ms  max ${total.max.toFixed(2)}ms`);
      console.log(`    resim fired on ${resimFrames}/${frames} frames (${((resimFrames / frames) * 100).toFixed(0)}%)`);
      console.log(`    resim pass (resimMs):    sum ${resimMs.sum.toFixed(0)}ms  share ${resimWallShare.toFixed(1)}%  ` +
        `mean/resim-frame ${(resimMs.sum / Math.max(1, resimFrames)).toFixed(2)}ms`);
      console.log(`    rollback (snap cap+res): share ${rollbackWallShare.toFixed(1)}%  ` +
        `(capture ${snapCap.mean.toFixed(3)}ms, restore ${snapRes.mean.toFixed(3)}ms /frame)`);
      console.log(`    resim frame ${meanResimFrame.toFixed(2)}ms vs non-resim ${meanNonResimFrame.toFixed(2)}ms ` +
        `(${(meanResimFrame / Math.max(1e-6, meanNonResimFrame)).toFixed(2)}×)`);

      // The lever fires on fracture. (How *often* it fires is scene-dependent —
      // the mini-city's ≈40%-of-frames frequency is locked by the recording
      // regression test; here we measure the per-event cost, which is universal.)
      expect(resimFrames).toBeGreaterThanOrEqual(1);
      // The second pass is intrinsically a material slice of wall-clock time —
      // measured by resimMs itself, with no output-changing on/off comparison.
      expect(resimWallShare).toBeGreaterThan(3);
      // A resim frame is markedly heavier than a comparable non-resim frame
      // (it re-runs the physics step on top of the initial pass + fracture work).
      expect(meanResimFrame).toBeGreaterThan(meanNonResimFrame * 1.4);
      // The rollback machinery (snapshot capture+restore) runs every fracture
      // frame and is non-trivial overhead even before the second world.step — a
      // concrete, output-safe optimisation target (scope/skip what it snapshots).
      expect(snapCap.sum + snapRes.sum).toBeGreaterThan(0);
    }, 120_000);
  });

  // ── C. rapierStep scaling with debris count (the dominant phase) ────────────
  describe('C. rapierStep dominates and scales with debris count', () => {
    it('rapierStep grows with accumulated rigid bodies and leads the phase ranking', async () => {
      await loadModules();
      const sizes = [
        { label: 'small  4x8 ', side: 4, stories: 8, mass: 2000, h: 4 },
        { label: 'medium 6x12', side: 6, stories: 12, mass: 5000, h: 6 },
        { label: 'large  8x16', side: 8, stories: 16, mass: 9000, h: 8 },
      ];
      console.log('\n  [rapierStep scaling] steady-state after impact');
      const points: Array<{ bodies: number; rapier: number }> = [];
      let dominantHits = 0;
      for (const s of sizes) {
        const scenario = buildTowerScenario({ side: s.side, stories: s.stories, totalMass: s.mass });
        const res = await runScenario(scenario, {}, { frames: 150, plan: centerHit(s.h, 40000) });
        // Steady-state window: last third of the run, after the cascade settles.
        const tail = res.samples.slice(Math.floor(res.samples.length * 0.66));
        const rapier = computeStats(tail.map((x) => x.rapierStepMs ?? 0));
        const solver = computeStats(tail.map((x) => x.solverUpdateMs ?? 0));
        const bodies = computeStats(tail.map((x) => x.rigidBodies ?? 0)).mean;
        points.push({ bodies, rapier: rapier.mean });
        if (rapier.mean >= solver.mean) dominantHits += 1;
        console.log(`    ${s.label}: ~${bodies.toFixed(0)} bodies  rapierStep ${rapier.mean.toFixed(2)}ms  solver ${solver.mean.toFixed(2)}ms`);
      }
      // More debris ⇒ more rapierStep time (the lever the recording flagged).
      expect(points[points.length - 1].bodies).toBeGreaterThan(points[0].bodies);
      expect(points[points.length - 1].rapier).toBeGreaterThan(points[0].rapier);
      // rapierStep is the dominant leaf at the larger scales (≥ the stress solve).
      expect(dominantHits).toBeGreaterThanOrEqual(2);
    }, 180_000);
  });
});

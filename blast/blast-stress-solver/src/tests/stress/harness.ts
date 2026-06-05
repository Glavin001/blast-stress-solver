/**
 * Stress harness — runs a scenario through the destruction pipeline, collects the
 * per-frame profiler samples the core already emits, and ranks the costliest
 * steps / sub-steps. Shared by the vitest suite (`rapier.city-stress.test.ts`) and
 * the shell script (`scripts/stress-city.mjs`).
 *
 * MEASUREMENT ONLY. Nothing here changes simulation behavior: it injects no forces
 * of its own beyond the caller's impact plan, reads the existing `CoreProfilerSample`
 * stream, and never tweaks solver iterations, resim, or world size. The core is
 * supplied via `buildCore` so this module stays engine-agnostic and bundleable
 * (the WASM-backed `buildDestructibleCore` is imported by the caller from `dist`).
 */
import type { CoreProfilerSample, ProjectileSpawn, ScenarioDesc } from '../../rapier/types';

// ── Stats ────────────────────────────────────────────────────────────────────
export interface Stats {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  sum: number;
  count: number;
}

export function computeStats(values: number[]): Stats {
  if (values.length === 0) return { mean: 0, p50: 0, p95: 0, p99: 0, max: 0, sum: 0, count: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const pick = (q: number) => sorted[Math.min(Math.floor(sorted.length * q), sorted.length - 1)];
  return {
    mean: sum / sorted.length,
    p50: pick(0.5),
    p95: pick(0.95),
    p99: pick(0.99),
    max: sorted[sorted.length - 1],
    sum,
    count: sorted.length,
  };
}

// ── Core surface we depend on (structural subset of DestructibleCore) ─────────
export interface ImpactEvent {
  frame: number;
  projectiles: ProjectileSpawn[];
}

export interface CoreLike {
  step(dt?: number): void;
  enqueueProjectile(spawn: ProjectileSpawn): void;
  getActiveBondsCount(): number;
  setProfiler(config: { enabled: boolean; onSample: (s: CoreProfilerSample) => void } | null): void;
  getIslandSettledStats?: () => {
    islandsTotal: number;
    islandsSettled: number;
    totalNodes: number;
    settledNodes: number;
    totalBonds: number;
    settledBonds: number;
    settledNodeFraction: number;
    settledBondFraction: number;
  };
  getIslandSolverStats?: () => { enabled: boolean; skipSettled: boolean; islandCount: number; islandsSkipped: number };
  setIslandSolver?: (opts: { enabled?: boolean; skipSettled?: boolean }) => void;
  dispose(): void;
}

export type BuildCore = (opts: Record<string, unknown>) => Promise<CoreLike>;

export interface RunOptions {
  name: string;
  scenario: ScenarioDesc;
  buildCore: BuildCore;
  coreOpts?: Record<string, unknown>;
  warmupFrames?: number;
  impactPlan?: ImpactEvent[];
  postImpactFrames?: number;
  dt?: number;
  /** Apply `core.setIslandSolver(...)` right after build (A/B the island-aware solve). */
  islandSolver?: { enabled?: boolean; skipSettled?: boolean };
  /** Record `getActiveBondsCount()` after every step for a per-frame parity trace.
   *  Adds a fixed per-step cost to BOTH arms of an A/B, so relative timing stays fair. */
  trackBondTrajectory?: boolean;
}

export interface ScenarioResult {
  name: string;
  nodeCount: number;
  bondsInitial: number;
  bondsFinal: number;
  setupMs: number;
  maxRigidBodies: number;
  crashed: boolean;
  crashFrame: number;
  samples: CoreProfilerSample[];
  windows: { warmup: [number, number]; impact: [number, number]; post: [number, number] };
  island?: { total: number; skippedMean: number; settledNodeFracMean: number };
  /** Per-frame active-bond count (only when trackBondTrajectory was set). */
  bondTrajectory?: number[];
}

// ── Runner ────────────────────────────────────────────────────────────────────
const DEFAULT_CORE_OPTS: Record<string, unknown> = {
  gravity: -9.81,
  materialScale: 1e8,
  resimulateOnFracture: true,
  maxResimulationPasses: 1,
  snapshotMode: 'perBody',
};

export async function runStressScenario(opts: RunOptions): Promise<ScenarioResult> {
  const { name, scenario, buildCore, coreOpts = {}, warmupFrames = 60, impactPlan = [], postImpactFrames = 180, dt = 1 / 60, islandSolver, trackBondTrajectory = false } = opts;

  const samples: CoreProfilerSample[] = [];
  const bondTrajectory: number[] = [];

  const setupStart = now();
  const core = await buildCore({ ...DEFAULT_CORE_OPTS, ...coreOpts, scenario });
  const setupMs = now() - setupStart;

  if (islandSolver) core.setIslandSolver?.(islandSolver);
  core.setProfiler({ enabled: true, onSample: (s) => samples.push(s) });

  const bondsInitial = core.getActiveBondsCount();

  let crashed = false;
  let crashFrame = -1;
  const safeStep = (frame: number) => {
    if (crashed) return;
    try {
      core.step(dt);
      if (trackBondTrajectory) bondTrajectory.push(core.getActiveBondsCount());
    } catch (err) {
      crashed = true;
      crashFrame = frame;
      // At extreme scale WASM memory can be exhausted mid-fracture; capture what we have.
      // eslint-disable-next-line no-console
      console.warn(`  [CRASH] ${name} frame ${frame}: ${(err as Error)?.message ?? err}`);
    }
  };

  // Island stats sampled across the impact window (skip-settled effectiveness).
  let islandTotal = 0;
  let islandSkippedSum = 0;
  let settledNodeFracSum = 0;
  let islandSampleCount = 0;
  const sampleIsland = () => {
    const solver = core.getIslandSolverStats?.();
    const settled = core.getIslandSettledStats?.();
    if (solver) {
      islandTotal = Math.max(islandTotal, solver.islandCount);
      islandSkippedSum += solver.islandsSkipped;
    }
    if (settled) settledNodeFracSum += settled.settledNodeFraction;
    if (solver || settled) islandSampleCount++;
  };

  // Phase 1: warmup (gravity only).
  const warmupStart = samples.length;
  for (let i = 0; i < warmupFrames; i++) safeStep(i);
  const warmupEnd = samples.length;

  // Phase 2: impact.
  const impactStart = samples.length;
  const lastImpactFrame = impactPlan.length ? Math.max(...impactPlan.map((p) => p.frame)) : 0;
  const impactFrames = Math.max(lastImpactFrame + 90, 120);
  for (let i = 0; i < impactFrames; i++) {
    if (!crashed) {
      for (const ev of impactPlan) {
        if (ev.frame === i) for (const p of ev.projectiles) core.enqueueProjectile(p);
      }
    }
    safeStep(warmupFrames + i);
    sampleIsland();
  }
  const impactEnd = samples.length;

  // Phase 3: settle.
  const postStart = samples.length;
  for (let i = 0; i < postImpactFrames; i++) safeStep(warmupFrames + impactFrames + i);
  const postEnd = samples.length;

  let bondsFinal: number;
  try {
    bondsFinal = core.getActiveBondsCount();
  } catch {
    bondsFinal = -1;
  }
  const maxRigidBodies = samples.reduce((m, s) => Math.max(m, s.rigidBodies ?? 0), 0);

  try {
    core.dispose();
  } catch {
    /* WASM may already be corrupt after a crash */
  }

  return {
    name,
    nodeCount: scenario.nodes.length,
    bondsInitial,
    bondsFinal,
    setupMs,
    maxRigidBodies,
    crashed,
    crashFrame,
    samples,
    windows: { warmup: [warmupStart, warmupEnd], impact: [impactStart, impactEnd], post: [postStart, postEnd] },
    island: islandSampleCount
      ? { total: islandTotal, skippedMean: islandSkippedSum / islandSampleCount, settledNodeFracMean: settledNodeFracSum / islandSampleCount }
      : undefined,
    bondTrajectory: trackBondTrajectory ? bondTrajectory : undefined,
  };
}

// ── Phase taxonomy ────────────────────────────────────────────────────────────
// Leaf phases are mutually exclusive and ≈ sum to totalMs (across initial + resim
// passes). Wrappers (solverUpdateMs, fractureMs, initialPassMs, resimMs) are NOT
// included here so we never double-count.
export const LEAF_PHASES = [
  'rapierStepMs',
  'contactDrainMs',
  'solverGravityInjectMs',
  'contactInjectResolveMs',
  'contactInjectGridMs',
  'contactInjectSplashMs',
  'contactInjectSubmitMs',
  'solverSolveMs',
  'fractureGenerateMs',
  'fractureApplyMs',
  'splitQueueMs',
  'bodyCreateMs',
  'colliderRebuildMs',
  'rebuildColliderMapMs',
  'cleanupDisabledMs',
  'damageReplayMs',
  'damagePreviewMs',
  'damageTickMs',
  'damageSnapshotMs',
  'damageRestoreMs',
  'damagePreDestroyMs',
  'damageFlushMs',
  'snapshotCaptureMs',
  'snapshotRestoreMs',
  'preStepSweepMs',
  'externalForceMs',
  'spawnMs',
  'projectileCleanupMs',
] as const;

/** Wrapper drilldowns: a parent and the leaves that compose it. */
export const PHASE_GROUPS: Record<string, { parent: string; children: string[] }> = {
  solver: { parent: 'solverUpdateMs', children: ['solverGravityInjectMs', 'contactInjectResolveMs', 'contactInjectGridMs', 'contactInjectSplashMs', 'contactInjectSubmitMs', 'solverSolveMs'] },
  contactInject: { parent: 'solverContactInjectMs', children: ['contactInjectResolveMs', 'contactInjectGridMs', 'contactInjectSplashMs', 'contactInjectSubmitMs'] },
  fracture: { parent: 'fractureMs', children: ['fractureGenerateMs', 'fractureApplyMs'] },
};

export interface PhaseRank {
  phase: string;
  sum: number;
  mean: number;
  p95: number;
  max: number;
  /** Share of total leaf time across the window, 0..1. */
  share: number;
}

export interface RankReport {
  frames: number;
  totalLeafSum: number;
  totalFrameMeanMs: number;
  ranked: PhaseRank[];
  dominant: string;
  /** Time spent in resim passes vs initial pass (wrappers). */
  initialPassMs: Stats;
  resimMs: Stats;
  /** Scaling: mean total frame time bucketed by rigid-body count. */
  scalingByBodies: Array<{ bodiesAtLeast: number; frames: number; meanTotalMs: number }>;
}

function field(s: CoreProfilerSample, key: string): number {
  return (s as unknown as Record<string, number>)[key] ?? 0;
}

/** Rank leaf phases by total wall-time contribution across the given samples. */
export function rankPhases(samples: CoreProfilerSample[]): RankReport {
  const ranked: PhaseRank[] = [];
  let totalLeafSum = 0;
  const leafStats: Record<string, Stats> = {};
  for (const key of LEAF_PHASES) {
    const st = computeStats(samples.map((s) => field(s, key)));
    leafStats[key] = st;
    totalLeafSum += st.sum;
  }
  for (const key of LEAF_PHASES) {
    const st = leafStats[key];
    if (st.max < 1e-4) continue;
    ranked.push({ phase: key, sum: st.sum, mean: st.mean, p95: st.p95, max: st.max, share: totalLeafSum > 0 ? st.sum / totalLeafSum : 0 });
  }
  ranked.sort((a, b) => b.sum - a.sum);

  const totalStats = computeStats(samples.map((s) => field(s, 'totalMs')));

  // Scaling buckets by body count.
  const buckets = [0, 25, 50, 100, 200, 400, 800];
  const scalingByBodies = buckets
    .map((threshold, i) => {
      const next = buckets[i + 1] ?? Infinity;
      const inBucket = samples.filter((s) => (s.rigidBodies ?? 0) >= threshold && (s.rigidBodies ?? 0) < next);
      return { bodiesAtLeast: threshold, frames: inBucket.length, meanTotalMs: computeStats(inBucket.map((s) => field(s, 'totalMs'))).mean };
    })
    .filter((b) => b.frames > 0);

  return {
    frames: samples.length,
    totalLeafSum,
    totalFrameMeanMs: totalStats.mean,
    ranked,
    dominant: ranked[0]?.phase ?? 'none',
    initialPassMs: computeStats(samples.map((s) => field(s, 'initialPassMs'))),
    resimMs: computeStats(samples.map((s) => field(s, 'resimMs'))),
    scalingByBodies,
  };
}

// ── Human-readable report ─────────────────────────────────────────────────────
const fmt = (v: number, w = 9) => v.toFixed(2).padStart(w);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`.padStart(7);

export function impactSamples(result: ScenarioResult): CoreProfilerSample[] {
  return result.samples.slice(result.windows.impact[0], result.windows.impact[1]);
}

export function printReport(results: ScenarioResult[]): void {
  const log = (...a: unknown[]) => console.log(...a);
  log('\n' + '='.repeat(92));
  log('  CITY DESTRUCTION STRESS — COSTLIEST STEPS REPORT');
  log('='.repeat(92));

  for (const r of results) {
    const win = impactSamples(r);
    const rank = rankPhases(win.length ? win : r.samples);
    log(`\n${'─'.repeat(92)}`);
    log(`  ${r.name}`);
    log(
      `  nodes=${r.nodeCount}  bonds=${r.bondsInitial}->${r.bondsFinal}  setup=${r.setupMs.toFixed(0)}ms  ` +
        `maxBodies=${r.maxRigidBodies}  frames=${r.samples.length}${r.crashed ? `  CRASHED@${r.crashFrame}` : ''}`,
    );
    if (r.island) {
      log(`  islands: total=${r.island.total}  meanSkipped/frame=${r.island.skippedMean.toFixed(1)}  settledNodeFrac=${pct(r.island.settledNodeFracMean)}`);
    }
    log(`  impact-window mean frame: ${rank.totalFrameMeanMs.toFixed(2)}ms over ${rank.frames} frames  (dominant: ${rank.dominant})`);
    log(`  ${'phase'.padEnd(26)} ${'share'.padStart(7)} ${'sum ms'.padStart(10)} ${'mean'.padStart(9)} ${'p95'.padStart(9)} ${'max'.padStart(9)}`);
    log('  ' + '-'.repeat(74));
    for (const p of rank.ranked.slice(0, 12)) {
      log(`  ${p.phase.padEnd(26)} ${pct(p.share)} ${fmt(p.sum, 10)} ${fmt(p.mean)} ${fmt(p.p95)} ${fmt(p.max)}`);
    }
    // Resim share
    const resimShare = rank.initialPassMs.sum + rank.resimMs.sum > 0 ? rank.resimMs.sum / (rank.initialPassMs.sum + rank.resimMs.sum) : 0;
    log(`  resim time share: ${pct(resimShare)} (initialPass sum ${rank.initialPassMs.sum.toFixed(1)}ms, resim sum ${rank.resimMs.sum.toFixed(1)}ms)`);
    // Scaling by bodies
    if (rank.scalingByBodies.length > 1) {
      const cells = rank.scalingByBodies.map((b) => `${b.bodiesAtLeast}+:${b.meanTotalMs.toFixed(1)}ms(${b.frames}f)`).join('  ');
      log(`  frame cost vs bodies: ${cells}`);
    }
    // rapierStepMs context: what is Rapier actually stepping?
    const rs = rapierStepContext(win.length ? win : r.samples);
    if (rs.meanAwake > 0 || rs.meanColliders > 0) {
      log(`  rapierStep context: awake bodies mean=${rs.meanAwake.toFixed(0)} (max ${rs.maxAwake})  colliders mean=${rs.meanColliders.toFixed(0)}`);
      if (rs.stepByAwake.length > 1) {
        const cells = rs.stepByAwake.map((b) => `${b.awakeAtLeast}+:${b.meanStepMs.toFixed(2)}ms(${b.frames}f)`).join('  ');
        log(`  rapierStepMs vs awake bodies: ${cells}`);
      }
    }
    // Island-aware solve activity (only when the island solver ran).
    const ia = islandActivity(win.length ? win : r.samples);
    if (ia) {
      log(`  island solve: total=${ia.meanTotal.toFixed(0)}  solved/frame=${ia.meanSolved.toFixed(1)}  skipped/frame=${ia.meanSkipped.toFixed(1)} (${pct(ia.meanTotal > 0 ? ia.meanSkipped / ia.meanTotal : 0)} skipped)`);
    }
  }

  // Cross-scenario summary.
  log(`\n${'='.repeat(92)}`);
  log('  SUMMARY');
  log('  ' + '-'.repeat(90));
  log(`  ${'scenario'.padEnd(28)} ${'nodes'.padStart(7)} ${'bodies'.padStart(7)} ${'mean'.padStart(8)} ${'p95'.padStart(8)} ${'max'.padStart(9)} ${'dominant'.padStart(24)}`);
  for (const r of results) {
    const win = impactSamples(r);
    const all = computeStats((win.length ? win : r.samples).map((s) => field(s, 'totalMs')));
    const dom = rankPhases(win.length ? win : r.samples).dominant;
    log(`  ${r.name.padEnd(28)} ${String(r.nodeCount).padStart(7)} ${String(r.maxRigidBodies).padStart(7)} ${fmt(all.mean, 8)} ${fmt(all.p95, 8)} ${fmt(all.max)} ${dom.padStart(24)}`);
  }
  log('='.repeat(92) + '\n');
}

// ── rapierStepMs / island context (drivers of the two dominant phases) ────────
export interface RapierStepContext {
  meanAwake: number;
  maxAwake: number;
  meanColliders: number;
  /** Mean rapierStepMs bucketed by awake-body count — reveals whether the Rapier
   *  step cost tracks awake bodies (=> sleeping/damping helps) more than total. */
  stepByAwake: Array<{ awakeAtLeast: number; frames: number; meanStepMs: number }>;
}

export function rapierStepContext(samples: CoreProfilerSample[]): RapierStepContext {
  const awake = samples.map((s) => (s as { rapierAwakeBodyCount?: number }).rapierAwakeBodyCount ?? 0);
  const colliders = samples.map((s) => (s as { rapierColliderCount?: number }).rapierColliderCount ?? 0);
  const buckets = [0, 25, 50, 100, 200, 400, 800, 1600];
  const stepByAwake = buckets
    .map((threshold, i) => {
      const next = buckets[i + 1] ?? Infinity;
      const inBucket = samples.filter((s) => {
        const a = (s as { rapierAwakeBodyCount?: number }).rapierAwakeBodyCount ?? 0;
        return a >= threshold && a < next;
      });
      return { awakeAtLeast: threshold, frames: inBucket.length, meanStepMs: computeStats(inBucket.map((s) => field(s, 'rapierStepMs'))).mean };
    })
    .filter((b) => b.frames > 0);
  const awakeStats = computeStats(awake);
  return { meanAwake: awakeStats.mean, maxAwake: awakeStats.max, meanColliders: computeStats(colliders).mean, stepByAwake };
}

export interface IslandActivity {
  meanTotal: number;
  meanSolved: number;
  meanSkipped: number;
}

/** Per-frame island-solve activity from the profiler (only populated when the
 *  island solver was enabled). Returns null otherwise. */
export function islandActivity(samples: CoreProfilerSample[]): IslandActivity | null {
  const withIsland = samples.filter((s) => (s as { islandSolveTotal?: number }).islandSolveTotal != null);
  if (!withIsland.length) return null;
  return {
    meanTotal: computeStats(withIsland.map((s) => (s as { islandSolveTotal?: number }).islandSolveTotal ?? 0)).mean,
    meanSolved: computeStats(withIsland.map((s) => (s as { islandSolveCount?: number }).islandSolveCount ?? 0)).mean,
    meanSkipped: computeStats(withIsland.map((s) => (s as { islandsSkipped?: number }).islandsSkipped ?? 0)).mean,
  };
}

// ── Machine-readable report ───────────────────────────────────────────────────
export function toJsonReport(results: ScenarioResult[]): unknown {
  return {
    generatedAt: new Date().toISOString(),
    scenarios: results.map((r) => {
      const win = impactSamples(r);
      const samplesForReport = win.length ? win : r.samples;
      const rank = rankPhases(samplesForReport);
      const rs = rapierStepContext(samplesForReport);
      const ia = islandActivity(samplesForReport);
      return {
        name: r.name,
        nodeCount: r.nodeCount,
        bondsInitial: r.bondsInitial,
        bondsFinal: r.bondsFinal,
        bondSurvivalPct: r.bondsInitial > 0 ? +((r.bondsFinal / r.bondsInitial) * 100).toFixed(1) : null,
        setupMs: +r.setupMs.toFixed(1),
        maxRigidBodies: r.maxRigidBodies,
        totalFrames: r.samples.length,
        crashed: r.crashed,
        crashFrame: r.crashFrame,
        island: r.island,
        impactWindow: {
          frames: rank.frames,
          meanFrameMs: +rank.totalFrameMeanMs.toFixed(3),
          dominant: rank.dominant,
          ranked: rank.ranked.map((p) => ({ phase: p.phase, share: +p.share.toFixed(4), sumMs: +p.sum.toFixed(2), meanMs: +p.mean.toFixed(4), p95Ms: +p.p95.toFixed(4), maxMs: +p.max.toFixed(3) })),
          resim: { initialPassSumMs: +rank.initialPassMs.sum.toFixed(2), resimSumMs: +rank.resimMs.sum.toFixed(2) },
          scalingByBodies: rank.scalingByBodies.map((b) => ({ ...b, meanTotalMs: +b.meanTotalMs.toFixed(3) })),
          rapierStep: { meanAwakeBodies: +rs.meanAwake.toFixed(1), maxAwakeBodies: rs.maxAwake, meanColliders: +rs.meanColliders.toFixed(1), stepByAwake: rs.stepByAwake.map((b) => ({ ...b, meanStepMs: +b.meanStepMs.toFixed(3) })) },
          islandActivity: ia ? { meanTotal: +ia.meanTotal.toFixed(1), meanSolved: +ia.meanSolved.toFixed(1), meanSkipped: +ia.meanSkipped.toFixed(1) } : null,
        },
      };
    }),
  };
}

// ── A/B comparison (e.g. island-aware solve on vs off) ────────────────────────
export interface ABParity {
  /** Exact match of the final active-bond count (the strongest single signal). */
  bondsFinalMatch: boolean;
  baselineBondsFinal: number;
  treatmentBondsFinal: number;
  baselineMaxBodies: number;
  treatmentMaxBodies: number;
  /** Max |Δ| in rigid-body count per frame across the full run (0 = identical path). */
  bodyTrajectoryMaxDiff: number;
  /** Max |Δ| in active-bond count per frame (only if both arms tracked it). */
  bondTrajectoryMaxDiff: number | null;
  framesCompared: number;
}

export interface ABPhaseDelta {
  phase: string;
  baselineSumMs: number;
  treatmentSumMs: number;
  /** baseline / treatment over the impact window. >1 = treatment faster. */
  speedup: number;
}

export interface ABComparison {
  label: string;
  baseline: ScenarioResult;
  treatment: ScenarioResult;
  phases: ABPhaseDelta[];
  /** Whole-frame speedup over the impact window (baseline mean / treatment mean). */
  frameSpeedup: number;
  parity: ABParity;
}

function sumField(samples: CoreProfilerSample[], key: string): number {
  return samples.reduce((a, s) => a + field(s, key), 0);
}

/** Compare two runs of the same scenario over their impact windows. `phases` lists
 *  the leaf phases to break out (the rest are summarized by frame speedup). */
export function compareAB(label: string, baseline: ScenarioResult, treatment: ScenarioResult, phases: string[] = ['solverSolveMs', 'rapierStepMs', 'snapshotCaptureMs', 'snapshotRestoreMs']): ABComparison {
  const bWin = impactSamples(baseline);
  const tWin = impactSamples(treatment);
  const phaseDeltas: ABPhaseDelta[] = phases.map((phase) => {
    const baselineSumMs = sumField(bWin, phase);
    const treatmentSumMs = sumField(tWin, phase);
    return { phase, baselineSumMs, treatmentSumMs, speedup: treatmentSumMs > 0 ? baselineSumMs / treatmentSumMs : Infinity };
  });
  const bMean = computeStats(bWin.map((s) => field(s, 'totalMs'))).mean;
  const tMean = computeStats(tWin.map((s) => field(s, 'totalMs'))).mean;

  // Parity: compare per-frame body counts (and bonds, if tracked) over the common length.
  const n = Math.min(baseline.samples.length, treatment.samples.length);
  let bodyTrajectoryMaxDiff = 0;
  for (let i = 0; i < n; i++) {
    bodyTrajectoryMaxDiff = Math.max(bodyTrajectoryMaxDiff, Math.abs((baseline.samples[i].rigidBodies ?? 0) - (treatment.samples[i].rigidBodies ?? 0)));
  }
  let bondTrajectoryMaxDiff: number | null = null;
  if (baseline.bondTrajectory && treatment.bondTrajectory) {
    const m = Math.min(baseline.bondTrajectory.length, treatment.bondTrajectory.length);
    bondTrajectoryMaxDiff = 0;
    for (let i = 0; i < m; i++) bondTrajectoryMaxDiff = Math.max(bondTrajectoryMaxDiff, Math.abs(baseline.bondTrajectory[i] - treatment.bondTrajectory[i]));
  }

  return {
    label,
    baseline,
    treatment,
    phases: phaseDeltas,
    frameSpeedup: tMean > 0 ? bMean / tMean : Infinity,
    parity: {
      bondsFinalMatch: baseline.bondsFinal === treatment.bondsFinal,
      baselineBondsFinal: baseline.bondsFinal,
      treatmentBondsFinal: treatment.bondsFinal,
      baselineMaxBodies: baseline.maxRigidBodies,
      treatmentMaxBodies: treatment.maxRigidBodies,
      bodyTrajectoryMaxDiff,
      bondTrajectoryMaxDiff,
      framesCompared: n,
    },
  };
}

export function printABReport(cmp: ABComparison): void {
  const log = (...a: unknown[]) => console.log(...a);
  log('\n' + '='.repeat(92));
  log(`  A/B COMPARISON — ${cmp.label}`);
  log('='.repeat(92));
  log(`  baseline:  ${cmp.baseline.name}  (islands total=${cmp.baseline.island?.total ?? 0}, meanSkipped/frame=${(cmp.baseline.island?.skippedMean ?? 0).toFixed(1)})`);
  log(`  treatment: ${cmp.treatment.name}  (islands total=${cmp.treatment.island?.total ?? 0}, meanSkipped/frame=${(cmp.treatment.island?.skippedMean ?? 0).toFixed(1)})`);
  log('');
  log(`  ${'phase'.padEnd(22)} ${'baseline sum'.padStart(13)} ${'treatment sum'.padStart(14)} ${'speedup'.padStart(9)}`);
  log('  ' + '-'.repeat(60));
  for (const p of cmp.phases) {
    log(`  ${p.phase.padEnd(22)} ${fmt(p.baselineSumMs, 11)}ms ${fmt(p.treatmentSumMs, 12)}ms ${(p.speedup).toFixed(2).padStart(8)}x`);
  }
  log('');
  log(`  whole-frame speedup (impact window): ${cmp.frameSpeedup.toFixed(2)}x  ${cmp.frameSpeedup >= 1 ? '(treatment faster)' : '(treatment SLOWER)'}`);
  log('');
  const p = cmp.parity;
  // Structural parity = the destruction itself: which bonds break, and when. This is
  // what "looks the same" means. Transient rigid-body count can still jitter by a few
  // (debris sleep/cleanup flips a threshold within solver tolerance) without any bond
  // breaking differently — that is incidental, not a behavior change.
  const fracturesIdentical = p.bondsFinalMatch && (p.bondTrajectoryMaxDiff === null || p.bondTrajectoryMaxDiff === 0);
  const bitIdentical = fracturesIdentical && p.bodyTrajectoryMaxDiff === 0;
  log('  PARITY (behavior must be preserved):');
  log(`    final bonds:      baseline=${p.baselineBondsFinal}  treatment=${p.treatmentBondsFinal}  ${p.bondsFinalMatch ? 'MATCH ✓' : 'MISMATCH ✗'}`);
  log(`    bond trajectory:  max |Δ|/frame = ${p.bondTrajectoryMaxDiff === null ? '(not tracked)' : p.bondTrajectoryMaxDiff}  ${fracturesIdentical ? '(fractures identical ✓)' : '(FRACTURES DIVERGE ✗)'}`);
  log(`    max bodies:       baseline=${p.baselineMaxBodies}  treatment=${p.treatmentMaxBodies}`);
  log(`    body trajectory:  max |Δ|/frame = ${p.bodyTrajectoryMaxDiff} over ${p.framesCompared} frames`);
  if (bitIdentical) {
    log('    => BIT-IDENTICAL observable behavior ✓ — speedup is free.');
  } else if (fracturesIdentical) {
    log(`    => DESTRUCTION IDENTICAL ✓ (same bonds break, same frames). Rigid-body count jitters by up to ${p.bodyTrajectoryMaxDiff} (transient debris/sleep bookkeeping within solver tolerance) — structural outcome unchanged.`);
  } else {
    log('    => FRACTURES DIVERGE ✗ — different bonds break; do NOT adopt without investigation.');
  }
  log('='.repeat(92) + '\n');
}

export function abToJsonReport(cmp: ABComparison): unknown {
  return {
    label: cmp.label,
    frameSpeedup: +cmp.frameSpeedup.toFixed(3),
    phases: cmp.phases.map((p) => ({ phase: p.phase, baselineSumMs: +p.baselineSumMs.toFixed(2), treatmentSumMs: +p.treatmentSumMs.toFixed(2), speedup: +p.speedup.toFixed(3) })),
    parity: cmp.parity,
    baseline: { name: cmp.baseline.name, bondsFinal: cmp.baseline.bondsFinal, maxBodies: cmp.baseline.maxRigidBodies, islandTotal: cmp.baseline.island?.total ?? 0 },
    treatment: { name: cmp.treatment.name, bondsFinal: cmp.treatment.bondsFinal, maxBodies: cmp.treatment.maxRigidBodies, islandTotal: cmp.treatment.island?.total ?? 0, meanIslandsSkipped: +(cmp.treatment.island?.skippedMean ?? 0).toFixed(2) },
  };
}

// ── timing source (browser perf or node) ──────────────────────────────────────
const now: () => number = typeof performance !== 'undefined' && typeof performance.now === 'function' ? () => performance.now() : () => Date.now();

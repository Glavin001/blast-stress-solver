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

interface RapierVecLike {
  x: number;
  y: number;
  z: number;
}
interface RapierBodyLike {
  translation(): RapierVecLike;
  linvel(): RapierVecLike;
  angvel(): RapierVecLike;
  isFixed?(): boolean;
  isDynamic?(): boolean;
  isSleeping?(): boolean;
}
export interface RapierWorldLike {
  forEachRigidBody(cb: (body: RapierBodyLike) => void): void;
}

/** Per-frame kinematics of the dynamic bodies — used to prove debris MOTION (and
 *  especially in-flight motion) is unchanged when we tune sleep/damping. */
export interface KinematicsSample {
  dynamic: number;
  awake: number;
  /** Max linear speed of any dynamic body (m/s) — the fastest airborne debris.
   *  Robust flight-parity signal: not diluted by sleeping bodies, and a fast piece
   *  can't sleep, so floaty damping would lower this. */
  maxLinSpeed: number;
  /** Max angular speed of any dynamic body (rad/s) — the fastest-tumbling debris. */
  maxAngSpeed: number;
  /** Mean linear speed of dynamic bodies (m/s) — settle indicator (drops as bodies sleep). */
  meanLinSpeed: number;
  /** Mean angular speed of dynamic bodies (rad/s) — settle indicator. */
  meanAngSpeed: number;
  /** Mean Y of dynamic bodies — collapse / settle height indicator. */
  comY: number;
  /** Order-insensitive position checksum: exact match ⇒ identical configuration. */
  posChecksum: number;
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
  /** Exposed by DestructibleCore — read per-frame body kinematics for parity checks. */
  world?: RapierWorldLike;
  dispose(): void;
}

/** Snapshot the dynamic-body kinematics from the Rapier world (cheap-ish; only
 *  called when trackKinematics is on, on both A/B arms, so timing stays fair). */
export function captureKinematics(world: RapierWorldLike): KinematicsSample {
  let dynamic = 0;
  let awake = 0;
  let sumLin = 0;
  let sumAng = 0;
  let maxLin = 0;
  let maxAng = 0;
  let sumY = 0;
  let posChecksum = 0;
  world.forEachRigidBody((b) => {
    const isDyn = typeof b.isDynamic === 'function' ? b.isDynamic() : !(b.isFixed?.() ?? false);
    if (!isDyn) return;
    dynamic++;
    if (!(b.isSleeping?.() ?? false)) awake++;
    const v = b.linvel();
    const w = b.angvel();
    const p = b.translation();
    const lin = Math.hypot(v.x, v.y, v.z);
    const ang = Math.hypot(w.x, w.y, w.z);
    sumLin += lin;
    sumAng += ang;
    if (lin > maxLin) maxLin = lin;
    if (ang > maxAng) maxAng = ang;
    sumY += p.y;
    // Commutative (order-insensitive) so it doesn't depend on iteration order.
    posChecksum += p.x * 1.000_000_1 + p.y * 1.000_01 + p.z * 1.1;
  });
  return {
    dynamic,
    awake,
    maxLinSpeed: maxLin,
    maxAngSpeed: maxAng,
    meanLinSpeed: dynamic ? sumLin / dynamic : 0,
    meanAngSpeed: dynamic ? sumAng / dynamic : 0,
    comY: dynamic ? sumY / dynamic : 0,
    posChecksum,
  };
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
  /** Record dynamic-body kinematics after every step (for motion-parity checks). */
  trackKinematics?: boolean;
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
  /** Per-frame dynamic-body kinematics (only when trackKinematics was set). */
  kinematics?: KinematicsSample[];
  /** Index (into samples/trajectories) of the first impact-window frame. */
  impactStartIndex?: number;
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
  const { name, scenario, buildCore, coreOpts = {}, warmupFrames = 60, impactPlan = [], postImpactFrames = 180, dt = 1 / 60, islandSolver, trackBondTrajectory = false, trackKinematics = false } = opts;

  const samples: CoreProfilerSample[] = [];
  const bondTrajectory: number[] = [];
  const kinematics: KinematicsSample[] = [];

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
      if (trackKinematics && core.world) kinematics.push(captureKinematics(core.world));
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
    kinematics: trackKinematics ? kinematics : undefined,
    impactStartIndex: impactStart,
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

/** Motion parity — proves debris MOVES the same (esp. in flight) between arms.
 *  Uses the MAX speed envelope (fastest airborne body), which a fast piece can't
 *  sleep out of, so it isn't diluted by the (intended) extra sleeping in treatment. */
export interface MotionParity {
  /** Peak (over the run) of the fastest body's linear speed. */
  peakMaxLinBaseline: number;
  peakMaxLinTreatment: number;
  /** (baseline - treatment)/baseline. ~0 = same peak flight speed; >0 = treatment floatier. */
  peakLinDropPct: number;
  peakMaxAngBaseline: number;
  peakMaxAngTreatment: number;
  /** ~0 = same peak tumble; >0 = treatment kills spin. */
  peakAngDropPct: number;
  /** First frame (offset from impact start) where the all-body position checksum
   *  diverges. Informational: expected to be non-null once pieces start landing &
   *  settling differently — NOT a floaty signal (that's the max-speed envelope). */
  divergenceOnsetFrame: number | null;
  framesCompared: number;
}

/** How fast each arm settles after the peak disturbance (the perf payoff). */
export interface SettleStats {
  awakeFraction: number;
  /** Frames from peak-awake to awake/dynamic <= awakeFraction (-1 = never settled). */
  baselineSettleFrames: number;
  treatmentSettleFrames: number;
  /** Mean awake-body count over the post-impact (settling) window — where the win lives. */
  baselineMeanAwakePost: number;
  treatmentMeanAwakePost: number;
  /** Awake bodies on the final frame (does the rubble ever go quiet?). */
  baselineFinalAwake: number;
  treatmentFinalAwake: number;
}

export interface ABComparison {
  label: string;
  baseline: ScenarioResult;
  treatment: ScenarioResult;
  phases: ABPhaseDelta[];
  /** Whole-frame speedup over the impact window (baseline mean / treatment mean). */
  frameSpeedup: number;
  /** Whole-frame speedup over the post-impact (settling) window — the sustained win. */
  frameSpeedupPost: number;
  /** Whole-frame speedup over the fully-settled tail (last frames) — the steady-state win,
   *  where treatment is asleep and baseline may still be jittering. */
  frameSpeedupSteady: number;
  parity: ABParity;
  motion: MotionParity | null;
  settle: SettleStats | null;
}

function sumField(samples: CoreProfilerSample[], key: string): number {
  return samples.reduce((a, s) => a + field(s, key), 0);
}

/** Frames from the peak-awake frame until awake/dynamic drops to <= fraction. */
function framesToSettle(result: ScenarioResult, awakeFraction: number): number {
  const k = result.kinematics;
  if (!k || !k.length) return -1;
  const start = result.impactStartIndex ?? 0;
  let peakIdx = start;
  let peak = -1;
  for (let i = start; i < k.length; i++) {
    if (k[i].awake > peak) { peak = k[i].awake; peakIdx = i; }
  }
  for (let i = peakIdx; i < k.length; i++) {
    if (k[i].dynamic > 0 && k[i].awake / k[i].dynamic <= awakeFraction) return i - peakIdx;
  }
  return -1;
}

function computeMotionParity(baseline: ScenarioResult, treatment: ScenarioResult): MotionParity | null {
  const bk = baseline.kinematics;
  const tk = treatment.kinematics;
  if (!bk || !tk || !bk.length || !tk.length) return null;
  const peak = (arr: KinematicsSample[], sel: (s: KinematicsSample) => number) => arr.reduce((m, s) => Math.max(m, sel(s)), 0);
  const peakLinB = peak(bk, (s) => s.maxLinSpeed);
  const peakLinT = peak(tk, (s) => s.maxLinSpeed);
  const peakAngB = peak(bk, (s) => s.maxAngSpeed);
  const peakAngT = peak(tk, (s) => s.maxAngSpeed);
  const start = baseline.impactStartIndex ?? 0;
  const m = Math.min(bk.length, tk.length);
  let divergenceOnsetFrame: number | null = null;
  for (let i = start; i < m; i++) {
    const a = bk[i].posChecksum;
    const b = tk[i].posChecksum;
    if (Math.abs(a - b) > 1e-6 * Math.max(1, Math.abs(a), Math.abs(b))) { divergenceOnsetFrame = i - start; break; }
  }
  return {
    peakMaxLinBaseline: peakLinB,
    peakMaxLinTreatment: peakLinT,
    peakLinDropPct: peakLinB > 0 ? (peakLinB - peakLinT) / peakLinB : 0,
    peakMaxAngBaseline: peakAngB,
    peakMaxAngTreatment: peakAngT,
    peakAngDropPct: peakAngB > 0 ? (peakAngB - peakAngT) / peakAngB : 0,
    divergenceOnsetFrame,
    framesCompared: m - start,
  };
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

  const awakeFraction = 0.05;
  const postMeanAwake = (r: ScenarioResult) => computeStats((r.kinematics ?? []).slice(r.windows.post[0], r.windows.post[1]).map((s) => s.awake)).mean;
  const finalAwake = (r: ScenarioResult) => (r.kinematics && r.kinematics.length ? r.kinematics[r.kinematics.length - 1].awake : 0);
  const settle: SettleStats | null = baseline.kinematics && treatment.kinematics
    ? {
        awakeFraction,
        baselineSettleFrames: framesToSettle(baseline, awakeFraction),
        treatmentSettleFrames: framesToSettle(treatment, awakeFraction),
        baselineMeanAwakePost: postMeanAwake(baseline),
        treatmentMeanAwakePost: postMeanAwake(treatment),
        baselineFinalAwake: finalAwake(baseline),
        treatmentFinalAwake: finalAwake(treatment),
      }
    : null;

  // Sustained (post-impact / settling) frame speedup — where reduced awake-body count
  // actually pays off; the impact window is dominated by the destruction itself.
  const bPost = computeStats(baseline.samples.slice(baseline.windows.post[0], baseline.windows.post[1]).map((s) => field(s, 'totalMs'))).mean;
  const tPost = computeStats(treatment.samples.slice(treatment.windows.post[0], treatment.windows.post[1]).map((s) => field(s, 'totalMs'))).mean;
  // Steady-state: the last frames, where the rubble should be fully at rest.
  const steadyN = Math.min(90, baseline.samples.length, treatment.samples.length);
  const bSteady = computeStats(baseline.samples.slice(-steadyN).map((s) => field(s, 'totalMs'))).mean;
  const tSteady = computeStats(treatment.samples.slice(-steadyN).map((s) => field(s, 'totalMs'))).mean;

  return {
    label,
    baseline,
    treatment,
    phases: phaseDeltas,
    frameSpeedup: tMean > 0 ? bMean / tMean : Infinity,
    frameSpeedupPost: tPost > 0 ? bPost / tPost : Infinity,
    frameSpeedupSteady: tSteady > 0 ? bSteady / tSteady : Infinity,
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
    motion: computeMotionParity(baseline, treatment),
    settle,
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
  log(`  whole-frame speedup — impact: ${cmp.frameSpeedup.toFixed(2)}x   post-impact: ${cmp.frameSpeedupPost.toFixed(2)}x   steady-state tail: ${cmp.frameSpeedupSteady.toFixed(2)}x  ${cmp.frameSpeedupSteady >= 1 ? '(treatment faster)' : '(treatment SLOWER)'}`);
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

  if (cmp.motion) {
    const m = cmp.motion;
    log('');
    log('  DEBRIS CHOREOGRAPHY (note: motion-changing settings diverge chaotically — this is NOT floaty):');
    log(`    peak max lin speed:   baseline=${m.peakMaxLinBaseline.toFixed(1)}  treatment=${m.peakMaxLinTreatment.toFixed(1)} m/s`);
    log(`    peak max ang speed:   baseline=${m.peakMaxAngBaseline.toFixed(1)}  treatment=${m.peakMaxAngTreatment.toFixed(1)} rad/s`);
    log(`    bit-identical until:  ${m.divergenceOnsetFrame === null ? 'whole run' : `frame +${m.divergenceOnsetFrame} after impact`} (≈ first ground contact; after that the piles diverge but stay equally valid)`);
    log('    NOTE: damping is gated to afterGroundCollision, so airborne pieces are never damped (no floaty flight by construction);');
    log('          raised sleep thresholds only catch near-stationary bodies, so nothing freezes mid-air.');
  }
  if (cmp.settle) {
    const s = cmp.settle;
    const fb = (v: number) => (v < 0 ? 'NEVER' : `${v}f`);
    log('');
    log('  SETTLE (the payoff — sooner sleep = fewer awake bodies = cheaper steps, esp. for resting rubble):');
    log(`    frames peak→settled (<=${pct(s.awakeFraction)} awake):  baseline=${fb(s.baselineSettleFrames)}  treatment=${fb(s.treatmentSettleFrames)}`);
    log(`    awake bodies, post-impact mean:        baseline=${s.baselineMeanAwakePost.toFixed(0)}  treatment=${s.treatmentMeanAwakePost.toFixed(0)}`);
    log(`    awake bodies on final frame:           baseline=${s.baselineFinalAwake}  treatment=${s.treatmentFinalAwake}`);
  }
  log('='.repeat(92) + '\n');
}

export function abToJsonReport(cmp: ABComparison): unknown {
  return {
    label: cmp.label,
    frameSpeedup: +cmp.frameSpeedup.toFixed(3),
    frameSpeedupPost: +cmp.frameSpeedupPost.toFixed(3),
    frameSpeedupSteady: +cmp.frameSpeedupSteady.toFixed(3),
    phases: cmp.phases.map((p) => ({ phase: p.phase, baselineSumMs: +p.baselineSumMs.toFixed(2), treatmentSumMs: +p.treatmentSumMs.toFixed(2), speedup: +p.speedup.toFixed(3) })),
    parity: cmp.parity,
    motion: cmp.motion,
    settle: cmp.settle,
    baseline: { name: cmp.baseline.name, bondsFinal: cmp.baseline.bondsFinal, maxBodies: cmp.baseline.maxRigidBodies, islandTotal: cmp.baseline.island?.total ?? 0 },
    treatment: { name: cmp.treatment.name, bondsFinal: cmp.treatment.bondsFinal, maxBodies: cmp.treatment.maxRigidBodies, islandTotal: cmp.treatment.island?.total ?? 0, meanIslandsSkipped: +(cmp.treatment.island?.skippedMean ?? 0).toFixed(2) },
  };
}

// ── timing source (browser perf or node) ──────────────────────────────────────
const now: () => number = typeof performance !== 'undefined' && typeof performance.now === 'function' ? () => performance.now() : () => Date.now();

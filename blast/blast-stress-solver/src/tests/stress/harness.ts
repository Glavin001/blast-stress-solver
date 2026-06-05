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
  const { name, scenario, buildCore, coreOpts = {}, warmupFrames = 60, impactPlan = [], postImpactFrames = 180, dt = 1 / 60 } = opts;

  const samples: CoreProfilerSample[] = [];

  const setupStart = now();
  const core = await buildCore({ ...DEFAULT_CORE_OPTS, ...coreOpts, scenario });
  const setupMs = now() - setupStart;

  core.setProfiler({ enabled: true, onSample: (s) => samples.push(s) });

  const bondsInitial = core.getActiveBondsCount();

  let crashed = false;
  let crashFrame = -1;
  const safeStep = (frame: number) => {
    if (crashed) return;
    try {
      core.step(dt);
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

// ── Machine-readable report ───────────────────────────────────────────────────
export function toJsonReport(results: ScenarioResult[]): unknown {
  return {
    generatedAt: new Date().toISOString(),
    scenarios: results.map((r) => {
      const win = impactSamples(r);
      const rank = rankPhases(win.length ? win : r.samples);
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
        },
      };
    }),
  };
}

// ── timing source (browser perf or node) ──────────────────────────────────────
const now: () => number = typeof performance !== 'undefined' && typeof performance.now === 'function' ? () => performance.now() : () => Date.now();

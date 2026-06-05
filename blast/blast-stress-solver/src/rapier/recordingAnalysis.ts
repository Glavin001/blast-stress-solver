/**
 * Offline **bottleneck analysis** for `blast-sim-recording/v1` session captures.
 *
 * A recording (see {@link ./sessionRecorder}) stores, for every frame, the full
 * per-phase timing breakdown (`timing.*`) alongside the scene complexity counters
 * (`columns.bodyCount`, `rigidBodies`, `activeBonds`, `islandCount`, …). That is
 * exactly the data you need to answer *"where does real-time runtime go on a large,
 * complex scene, and what scales it?"* — without re-running the simulation.
 *
 * This module is **pure** (no DOM, no WASM, no Rapier): it consumes a
 * {@link DecodedSimRecording} and produces a structured {@link RecordingAnalysis}.
 * It is the engine behind both the `analyze-recording` CLI and the recording
 * regression tests, and it never needs the heavy per-frame body trajectory blob —
 * only the timing/columns/resim streams — so it runs anywhere.
 *
 * The goal is to **discover** levers, not change behaviour. Every number here is
 * descriptive: phase shares, body-count scaling fits, and the marginal cost of a
 * resimulation pass. Use it to decide *what* to optimise and, paired with an
 * output-equivalence trajectory check, to prove an optimisation stayed faithful.
 */
import { TIMING_FIELDS, type DecodedSimRecording } from './sessionRecorder';

// ── Timing-field taxonomy ─────────────────────────────────────────────────────
// TIMING_FIELDS mixes three kinds of column. To attribute wall-clock time without
// double counting we split them:
//
//  • TOP-LEVEL LEAVES — mutually (≈) exclusive timers that sum to ~`totalMs`.
//  • SUB-BREAKDOWNS    — children of a leaf (e.g. CGNR solve inside solverUpdate);
//                        reported for drill-down but NOT summed.
//  • WRAPPERS / COUNTS — `totalMs`, the per-pass roll-ups, and integer counters.

/** Phase timers that are (approximately) mutually exclusive and sum to ~totalMs. */
export const LEAF_TIMING_FIELDS = [
  'rapierStepMs',
  'solverUpdateMs',
  'contactDrainMs',
  'externalForceMs',
  'preStepSweepMs',
  'fractureMs',
  'splitPlannerMs',
  'splitQueueMs',
  'bodyCreateMs',
  'colliderRebuildMs',
  'rebuildColliderMapMs',
  'cleanupDisabledMs',
  'snapshotCaptureMs',
  'snapshotRestoreMs',
  'damageTickMs',
  'damageReplayMs',
  'damagePreviewMs',
  'damageRestoreMs',
  'damageSnapshotMs',
  'damagePreDestroyMs',
  'damageFlushMs',
  'spawnMs',
  'projectileCleanupMs',
] as const;

/** Children of a leaf timer — for drill-down, never summed into the leaf total. */
export const SUB_BREAKDOWN_FIELDS = [
  // children of solverUpdateMs
  'solverGravityInjectMs',
  'solverContactInjectMs',
  'solverSolveMs',
  // children of solverContactInjectMs
  'contactInjectResolveMs',
  'contactInjectGridMs',
  'contactInjectSplashMs',
  'contactInjectSubmitMs',
  // children of fractureMs
  'fractureGenerateMs',
  'fractureApplyMs',
] as const;

/** Per-pass roll-ups that *contain* leaf work (cross-check only, never summed). */
export const WRAPPER_TIMING_FIELDS = ['totalMs', 'initialPassMs', 'resimMs'] as const;

export type Stats = {
  count: number;
  min: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  sum: number;
  /** Population standard deviation. */
  stdev: number;
};

export type PhaseStat = {
  field: string;
  stats: Stats;
  /** mean(field) / mean(totalMs), as a percentage. */
  shareOfTotalPct: number;
  kind: 'leaf' | 'sub' | 'wrapper';
};

/** Ordinary-least-squares fit of y on x: y ≈ slope·x + intercept. */
export type LinearFit = {
  slope: number;
  intercept: number;
  /** Coefficient of determination (0…1); how much of the variance the fit explains. */
  r2: number;
  n: number;
};

export type ScalingFit = {
  field: string;
  /** Fit of `field` (ms) against `rigidBodies` (count). */
  fit: LinearFit;
  /** slope · 100 — milliseconds added per 100 additional rigid bodies. */
  msPer100Bodies: number;
};

export type ResimAnalysis = {
  totalFrames: number;
  resimFrames: number;
  resimFraction: number;
  /** Direct accounting: sum(resimMs) / sum(totalMs). The wall-clock spent in the
   *  second (rollback + re-step) pass, as a share of the whole session. */
  resimShareOfWallClockPct: number;
  /** mean(resimMs) over frames that actually ran a resim pass — the marginal cost. */
  meanResimPassMs: number;
  /** mean(totalMs) over resim frames vs. non-resim frames, matched within
   *  `rigidBodies` buckets so the comparison is not confounded by the fact that
   *  resim frames tend to coincide with heavier scenes. */
  matchedExtraMsPerResimFrame: number;
  /** Matched ratio totalMs(resim) / totalMs(non-resim) — ≈2.0 means "a resim frame
   *  costs about twice a comparable non-resim frame". */
  matchedResimMultiplier: number;
};

export type BudgetAnalysis = {
  budgetMs: number;
  meanFps: number;
  framesOverBudget: number;
  /** Percentage of frames that missed the 60 FPS (16.67 ms) budget. */
  pctMiss60: number;
  /** Percentage of frames that missed the 30 FPS (33.3 ms) budget. */
  pctMiss30: number;
  worstFrame: {
    index: number;
    totalMs: number;
    dominantLeaf: string;
    rigidBodies: number;
    resimPasses: number;
  };
};

export type Hitch = {
  field: string;
  frameIndex: number;
  ms: number;
  p95: number;
  /** ms / p95 — how far above the field's typical ceiling this spike sits. */
  ratio: number;
};

export type RecordingAnalysis = {
  frames: number;
  durationSeconds: number;
  meanFps: number;
  demo: string | null;
  scene: {
    bodyCount: Stats;
    rigidBodies: Stats;
    activeBonds: Stats;
    islandCount: Stats;
    islandsSkipped: Stats;
    projectiles: Stats;
  };
  total: Stats;
  /** All timing fields with non-trivial cost, sorted by mean descending. */
  phases: PhaseStat[];
  /** mean of (sum of LEAF_TIMING_FIELDS per frame). */
  leafSumMeanMs: number;
  /** mean(totalMs) − leafSumMeanMs — unattributed overhead (JS glue, GC, profiler). */
  unaccountedMeanMs: number;
  budget: BudgetAnalysis;
  scaling: ScalingFit[];
  resim: ResimAnalysis;
  hitches: Hitch[];
};

// ── Statistics helpers ────────────────────────────────────────────────────────

export function computeStats(values: ArrayLike<number>): Stats {
  const n = values.length;
  if (n === 0) {
    return { count: 0, min: 0, mean: 0, p50: 0, p95: 0, p99: 0, max: 0, sum: 0, stdev: 0 };
  }
  const sorted = Array.from(values as ArrayLike<number>).sort((a, b) => a - b);
  let sum = 0;
  for (let i = 0; i < n; i += 1) sum += sorted[i];
  const mean = sum / n;
  let varAcc = 0;
  for (let i = 0; i < n; i += 1) {
    const d = sorted[i] - mean;
    varAcc += d * d;
  }
  const pct = (p: number) => sorted[Math.min(n - 1, Math.max(0, Math.floor(p * (n - 1))))];
  return {
    count: n,
    min: sorted[0],
    mean,
    p50: pct(0.5),
    p95: pct(0.95),
    p99: pct(0.99),
    max: sorted[n - 1],
    sum,
    stdev: Math.sqrt(varAcc / n),
  };
}

/** Ordinary least squares: fit y ≈ slope·x + intercept, with R². */
export function linearFit(xs: ArrayLike<number>, ys: ArrayLike<number>): LinearFit {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return { slope: 0, intercept: n === 1 ? ys[0] : 0, r2: 0, n };
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i += 1) { sx += xs[i]; sy += ys[i]; }
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  if (sxx === 0) return { slope: 0, intercept: my, r2: 0, n };
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return { slope, intercept, r2, n };
}

// ── Column access (tolerant of older recordings missing some fields) ──────────

function timingColumn(rec: DecodedSimRecording, field: string): Float32Array {
  return rec.timing[field] ?? new Float32Array(rec.durationFrames);
}

function fieldKind(field: string): 'leaf' | 'sub' | 'wrapper' {
  if ((LEAF_TIMING_FIELDS as readonly string[]).includes(field)) return 'leaf';
  if ((SUB_BREAKDOWN_FIELDS as readonly string[]).includes(field)) return 'sub';
  return 'wrapper';
}

// ── Main analysis ─────────────────────────────────────────────────────────────

export type AnalyzeOptions = {
  /** Frame budget in ms (default 1000/60 ≈ 16.667). */
  budgetMs?: number;
  /** Drop the first N frames from steady-state stats (the build/warm-up hitch).
   *  Hitches are still reported separately. Default 0 (analyse everything). */
  warmupFrames?: number;
  /** Width of the rigid-body buckets used to de-confound the resim comparison. */
  resimBodyBucket?: number;
  /** A field is flagged as a one-time hitch when its frame value exceeds this
   *  multiple of its own p95. Default 4. */
  hitchRatio?: number;
  /** The recording's `meta` (carries `demo`, config, …). `decodeSimRecording`
   *  does not retain it, so pass it from the raw bundle when you want it labelled. */
  meta?: Record<string, unknown>;
};

export function analyzeRecording(
  rec: DecodedSimRecording,
  opts: AnalyzeOptions = {},
): RecordingAnalysis {
  const budgetMs = opts.budgetMs ?? 1000 / 60;
  const warmup = Math.max(0, opts.warmupFrames ?? 0);
  const bucketW = Math.max(1, opts.resimBodyBucket ?? 100);
  const hitchRatio = opts.hitchRatio ?? 4;

  const nAll = rec.durationFrames;
  // Steady-state window for distribution stats (hitches reported separately).
  const lo = Math.min(warmup, Math.max(0, nAll - 1));
  const sliceN = nAll - lo;

  const totalCol = timingColumn(rec, 'totalMs').subarray(lo);
  const total = computeStats(totalCol);
  const totalMean = total.mean || 1; // guard /0 for share

  // Per-phase stats + share of total.
  const phases: PhaseStat[] = [];
  for (const field of TIMING_FIELDS) {
    if (field === 'resimPasses' || field === 'rigidBodies') continue; // counts, not timers
    const col = timingColumn(rec, field).subarray(lo);
    const stats = computeStats(col);
    if (stats.max < 1e-4 && field !== 'totalMs') continue; // skip dead phases
    phases.push({
      field,
      stats,
      shareOfTotalPct: (stats.mean / totalMean) * 100,
      kind: fieldKind(field),
    });
  }
  phases.sort((a, b) => b.stats.mean - a.stats.mean);

  // Leaf sum (attributed) vs total (per frame, then mean).
  let leafSumAcc = 0;
  for (let i = lo; i < nAll; i += 1) {
    let s = 0;
    for (const f of LEAF_TIMING_FIELDS) s += timingColumn(rec, f)[i] ?? 0;
    leafSumAcc += s;
  }
  const leafSumMeanMs = sliceN > 0 ? leafSumAcc / sliceN : 0;

  // Budget / frame-rate analysis (full session — startup hitch matters here).
  const totalFull = timingColumn(rec, 'totalMs');
  let over = 0, miss30 = 0, worstIdx = 0, worstMs = -1;
  for (let i = 0; i < nAll; i += 1) {
    const ms = totalFull[i];
    if (ms > budgetMs) over += 1;
    if (ms > 1000 / 30) miss30 += 1;
    if (ms > worstMs) { worstMs = ms; worstIdx = i; }
  }
  const resimPassesCol = timingColumn(rec, 'resimPasses');
  const rigidCol = rec.columns.rigidBodies;
  const dominantLeafAt = (i: number): string => {
    let best = 'rapierStepMs', bestMs = -1;
    for (const f of LEAF_TIMING_FIELDS) {
      const v = timingColumn(rec, f)[i] ?? 0;
      if (v > bestMs) { bestMs = v; best = f; }
    }
    return best;
  };
  const budget: BudgetAnalysis = {
    budgetMs,
    meanFps: rec.durationSeconds > 0 ? nAll / rec.durationSeconds : 0,
    framesOverBudget: over,
    pctMiss60: (over / nAll) * 100,
    pctMiss30: (miss30 / nAll) * 100,
    worstFrame: {
      index: worstIdx,
      totalMs: worstMs,
      dominantLeaf: dominantLeafAt(worstIdx),
      rigidBodies: rigidCol[worstIdx] ?? 0,
      resimPasses: resimPassesCol[worstIdx] ?? 0,
    },
  };

  // Body-count scaling fits for the rapier/solver levers.
  const xs = Array.from(rigidCol);
  const scaling: ScalingFit[] = ['totalMs', 'rapierStepMs', 'solverUpdateMs', 'solverSolveMs'].map(
    (field) => {
      const fit = linearFit(xs, Array.from(timingColumn(rec, field)));
      return { field, fit, msPer100Bodies: fit.slope * 100 };
    },
  );

  // Resim analysis — direct accounting + body-bucket-matched comparison.
  const resim = analyzeResim(rec, bucketW);

  // One-time hitches: per-field spikes far above that field's own p95.
  const hitches: Hitch[] = [];
  for (const field of [...LEAF_TIMING_FIELDS, 'totalMs']) {
    const col = timingColumn(rec, field);
    const p95 = computeStats(col).p95;
    if (p95 < 1e-3) continue;
    let maxMs = -1, maxAt = -1;
    for (let i = 0; i < nAll; i += 1) if (col[i] > maxMs) { maxMs = col[i]; maxAt = i; }
    if (maxMs > p95 * hitchRatio && maxMs > budgetMs) {
      hitches.push({ field, frameIndex: maxAt, ms: maxMs, p95, ratio: maxMs / p95 });
    }
  }
  hitches.sort((a, b) => b.ms - a.ms);

  const demo =
    (opts.meta?.demo as string | undefined) ??
    (rec as unknown as { meta?: { demo?: string } }).meta?.demo ??
    null;

  return {
    frames: nAll,
    durationSeconds: rec.durationSeconds,
    meanFps: budget.meanFps,
    demo,
    scene: {
      bodyCount: computeStats(rec.columns.bodyCount),
      rigidBodies: computeStats(rec.columns.rigidBodies),
      activeBonds: computeStats(rec.columns.activeBonds),
      islandCount: computeStats(rec.columns.islandCount),
      islandsSkipped: computeStats(rec.columns.islandsSkipped),
      projectiles: computeStats(rec.columns.projectiles),
    },
    total,
    phases,
    leafSumMeanMs,
    unaccountedMeanMs: total.mean - leafSumMeanMs,
    budget,
    scaling,
    resim,
    hitches,
  };
}

function analyzeResim(rec: DecodedSimRecording, bucketW: number): ResimAnalysis {
  const n = rec.durationFrames;
  const totalCol = timingColumn(rec, 'totalMs');
  const resimMsCol = timingColumn(rec, 'resimMs');
  const passesCol = timingColumn(rec, 'resimPasses');
  const rigid = rec.columns.rigidBodies;

  let totalSum = 0, resimSum = 0, resimFrames = 0, resimPassMsSum = 0;
  for (let i = 0; i < n; i += 1) {
    totalSum += totalCol[i];
    resimSum += resimMsCol[i];
    if (passesCol[i] > 0) {
      resimFrames += 1;
      resimPassMsSum += resimMsCol[i];
    }
  }

  // Bucket frames by rigid-body count, then within each bucket compare the mean
  // totalMs of resim vs non-resim frames. Averaging the per-bucket deltas controls
  // for the confound that resim frames cluster where the scene is already heavy.
  type Bucket = { rs: number[]; nr: number[] };
  const buckets = new Map<number, Bucket>();
  for (let i = 0; i < n; i += 1) {
    const key = Math.floor((rigid[i] ?? 0) / bucketW);
    let b = buckets.get(key);
    if (!b) { b = { rs: [], nr: [] }; buckets.set(key, b); }
    (passesCol[i] > 0 ? b.rs : b.nr).push(totalCol[i]);
  }
  let wExtra = 0, wRatioNum = 0, wRatioDen = 0, wSum = 0;
  for (const b of buckets.values()) {
    if (b.rs.length === 0 || b.nr.length === 0) continue;
    const mr = b.rs.reduce((a, c) => a + c, 0) / b.rs.length;
    const mn = b.nr.reduce((a, c) => a + c, 0) / b.nr.length;
    const w = Math.min(b.rs.length, b.nr.length); // weight by the comparable count
    wExtra += (mr - mn) * w;
    wRatioNum += (mr / mn) * w;
    wRatioDen += w;
    wSum += w;
  }

  return {
    totalFrames: n,
    resimFrames,
    resimFraction: n > 0 ? resimFrames / n : 0,
    resimShareOfWallClockPct: totalSum > 0 ? (resimSum / totalSum) * 100 : 0,
    meanResimPassMs: resimFrames > 0 ? resimPassMsSum / resimFrames : 0,
    matchedExtraMsPerResimFrame: wSum > 0 ? wExtra / wSum : 0,
    matchedResimMultiplier: wRatioDen > 0 ? wRatioNum / wRatioDen : 0,
  };
}

// ── Human-readable report ─────────────────────────────────────────────────────

function fmt(v: number, w = 8, d = 3): string {
  return v.toFixed(d).padStart(w);
}

/** Render a {@link RecordingAnalysis} as a console-friendly multi-line report. */
export function formatAnalysisReport(a: RecordingAnalysis): string {
  const L: string[] = [];
  const push = (s = '') => L.push(s);

  push('═'.repeat(78));
  push(`  SESSION RECORDING ANALYSIS${a.demo ? `  —  ${a.demo}` : ''}`);
  push('═'.repeat(78));
  push(
    `  Frames: ${a.frames}   Duration: ${a.durationSeconds.toFixed(1)}s   ` +
      `Capture FPS: ${a.meanFps.toFixed(1)}`,
  );
  push(
    `  Scene: bodies ${a.scene.bodyCount.min}→${a.scene.bodyCount.max} ` +
      `(rigid ${a.scene.rigidBodies.min}→${a.scene.rigidBodies.max}), ` +
      `bonds ${a.scene.activeBonds.min}→${a.scene.activeBonds.max}, ` +
      `islands ${a.scene.islandCount.min}→${a.scene.islandCount.max}`,
  );
  push();
  push(
    `  Frame time: mean ${a.total.mean.toFixed(2)}ms  p95 ${a.total.p95.toFixed(2)}ms  ` +
      `max ${a.total.max.toFixed(2)}ms`,
  );
  push(
    `  Budget (${a.budget.budgetMs.toFixed(2)}ms): ${a.budget.pctMiss60.toFixed(0)}% miss 60 FPS, ` +
      `${a.budget.pctMiss30.toFixed(0)}% miss 30 FPS`,
  );
  const wf = a.budget.worstFrame;
  push(
    `  Worst frame #${wf.index}: ${wf.totalMs.toFixed(1)}ms ` +
      `(${wf.dominantLeaf}, ${wf.rigidBodies} bodies, ${wf.resimPasses} resim)`,
  );

  push();
  push('  PHASE BREAKDOWN (mean ms, share of frame)'.padEnd(60));
  push(`  ${'phase'.padEnd(24)}${'mean'.padStart(8)}${'p95'.padStart(8)}${'max'.padStart(9)}${'share'.padStart(9)}`);
  push('  ' + '─'.repeat(56));
  for (const p of a.phases) {
    if (p.kind === 'wrapper' && p.field !== 'resimMs' && p.field !== 'initialPassMs') continue;
    const indent = p.kind === 'sub' ? '   └ ' : '  ';
    const name = p.kind === 'sub' ? p.field : p.field;
    if (p.stats.mean < 0.01 && p.kind !== 'leaf') continue;
    push(
      `${indent}${name.padEnd(p.kind === 'sub' ? 21 : 22)}${fmt(p.stats.mean)}${fmt(p.stats.p95)}` +
        `${fmt(p.stats.max, 9, 2)}${(p.shareOfTotalPct).toFixed(1).padStart(8)}%`,
    );
  }
  push('  ' + '─'.repeat(56));
  push(
    `  ${'leaf sum (attributed)'.padEnd(22)}${fmt(a.leafSumMeanMs)}` +
      `        (unaccounted ${a.unaccountedMeanMs.toFixed(2)}ms)`,
  );

  push();
  push('  BODY-COUNT SCALING (ms added per 100 rigid bodies)');
  for (const s of a.scaling) {
    push(
      `  ${s.field.padEnd(18)} ${fmt(s.msPer100Bodies, 7, 3)} ms/100  ` +
        `(intercept ${s.fit.intercept.toFixed(2)}ms, R²=${s.fit.r2.toFixed(2)})`,
    );
  }

  push();
  push('  RESIMULATION (rollback + re-step on fracture)');
  push(
    `  ${a.resim.resimFrames}/${a.resim.totalFrames} frames ran a resim pass ` +
      `(${(a.resim.resimFraction * 100).toFixed(0)}%)`,
  );
  push(
    `  resim wall-clock share: ${a.resim.resimShareOfWallClockPct.toFixed(1)}%   ` +
      `mean resim pass: ${a.resim.meanResimPassMs.toFixed(2)}ms`,
  );
  push(
    `  body-matched: a resim frame costs +${a.resim.matchedExtraMsPerResimFrame.toFixed(2)}ms ` +
      `(${a.resim.matchedResimMultiplier.toFixed(2)}× a comparable non-resim frame)`,
  );

  if (a.hitches.length > 0) {
    push();
    push('  ONE-TIME HITCHES (spike ≫ field p95)');
    for (const h of a.hitches.slice(0, 6)) {
      push(
        `  frame #${String(h.frameIndex).padStart(4)}  ${h.field.padEnd(20)} ` +
          `${h.ms.toFixed(1)}ms  (${h.ratio.toFixed(0)}× its ${h.p95.toFixed(2)}ms p95)`,
      );
    }
  }
  push('═'.repeat(78));
  return L.join('\n');
}

/**
 * Unit tests for the pure-TS recording **bottleneck analyzer**
 * ({@link ../rapier/recordingAnalysis}).
 *
 * These build small synthetic {@link DecodedSimRecording}s with *known* timing
 * shapes and assert the analyzer recovers them: distribution stats, the
 * body-count scaling regression, the de-confounded resimulation multiplier, the
 * frame-budget percentages, and one-time-hitch detection. No WASM, no Rapier, no
 * fixtures — this is the correctness harness for the engine that powers the
 * mini-city regression test and the `inspect-recording`/analysis tooling.
 */
import { describe, it, expect } from 'vitest';
import { TIMING_FIELDS, type DecodedSimRecording } from '../rapier/sessionRecorder';
import {
  analyzeRecording,
  computeStats,
  linearFit,
  LEAF_TIMING_FIELDS,
  SUB_BREAKDOWN_FIELDS,
  WRAPPER_TIMING_FIELDS,
} from '../rapier/recordingAnalysis';

// ── Synthetic recording builder ───────────────────────────────────────────────
// The analyzer only reads `timing[*]`, `columns.*`, `durationFrames`,
// `durationSeconds`. We fill those and stub the (unused-by-analysis) body
// trajectory accessors so the object satisfies DecodedSimRecording.

type Cols = Partial<Record<string, number[]>>;

function makeRecording(opts: {
  frames: number;
  durationSeconds?: number;
  timing?: Cols; // keyed by TIMING_FIELDS name
  rigidBodies?: number[];
  bodyCount?: number[];
  activeBonds?: number[];
  islandCount?: number[];
  islandsSkipped?: number[];
  projectiles?: number[];
  meta?: Record<string, unknown>;
}): DecodedSimRecording {
  const n = opts.frames;
  const zeros = () => new Float32Array(n);
  const timing: Record<string, Float32Array> = {};
  for (const f of TIMING_FIELDS) {
    timing[f] = Float32Array.from(opts.timing?.[f] ?? new Array(n).fill(0));
  }
  const u32 = (a?: number[]) => Uint32Array.from(a ?? new Array(n).fill(0));
  const rec: DecodedSimRecording = {
    durationFrames: n,
    durationSeconds: opts.durationSeconds ?? n / 60,
    bodyStride: 14,
    bodyLayout: ['handle'],
    handleTable: new Float64Array(0),
    columns: {
      simTime: Float64Array.from(Array.from({ length: n }, (_, i) => i / 60)),
      dt: Float32Array.from(new Array(n).fill(1 / 60)),
      bodyCount: u32(opts.bodyCount ?? opts.rigidBodies),
      activeBonds: u32(opts.activeBonds),
      rigidBodies: u32(opts.rigidBodies),
      projectiles: u32(opts.projectiles),
      islandCount: u32(opts.islandCount),
      islandsSkipped: u32(opts.islandsSkipped),
    },
    bodies: new Float32Array(0),
    frameBodyOffset: new Uint32Array(n),
    events: [],
    timing,
    resimLog: [],
    frame: () => new Float32Array(0),
    bodyInFrame: () => null,
  };
  // Attach optional meta for the demo label (not part of DecodedSimRecording but
  // read defensively by the analyzer).
  if (opts.meta) (rec as unknown as { meta: unknown }).meta = opts.meta;
  void zeros;
  return rec;
}

describe('computeStats', () => {
  it('computes mean / percentiles / stdev on a known sample', () => {
    const s = computeStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(s.count).toBe(10);
    expect(s.min).toBe(1);
    expect(s.max).toBe(10);
    expect(s.mean).toBeCloseTo(5.5, 6);
    expect(s.sum).toBe(55);
    // p95 index = floor(0.95 * 9) = 8 → sorted[8] = 9
    expect(s.p95).toBe(9);
    expect(s.stdev).toBeCloseTo(Math.sqrt(8.25), 6);
  });

  it('is order-independent and handles the empty case', () => {
    expect(computeStats([5, 1, 3, 2, 4]).mean).toBeCloseTo(3, 6);
    const e = computeStats([]);
    expect(e.count).toBe(0);
    expect(e.mean).toBe(0);
    expect(e.max).toBe(0);
  });
});

describe('linearFit', () => {
  it('recovers an exact line with R²=1', () => {
    const xs = [0, 1, 2, 3, 4, 5];
    const ys = xs.map((x) => 2 * x + 3);
    const fit = linearFit(xs, ys);
    expect(fit.slope).toBeCloseTo(2, 9);
    expect(fit.intercept).toBeCloseTo(3, 9);
    expect(fit.r2).toBeCloseTo(1, 9);
  });

  it('returns a degenerate fit for constant x', () => {
    const fit = linearFit([3, 3, 3], [1, 2, 3]);
    expect(fit.slope).toBe(0);
    expect(fit.r2).toBe(0);
  });
});

describe('phase taxonomy', () => {
  it('partitions TIMING_FIELDS into disjoint leaf / sub / wrapper / count sets', () => {
    const counts = new Set(['resimPasses', 'rigidBodies']);
    const leaf = new Set<string>(LEAF_TIMING_FIELDS);
    const sub = new Set<string>(SUB_BREAKDOWN_FIELDS);
    const wrap = new Set<string>(WRAPPER_TIMING_FIELDS);
    // disjoint
    for (const f of leaf) expect(sub.has(f) || wrap.has(f)).toBe(false);
    for (const f of sub) expect(wrap.has(f)).toBe(false);
    // every timing field is classified exactly once
    for (const f of TIMING_FIELDS) {
      if (counts.has(f)) continue;
      const hits = [leaf.has(f), sub.has(f), wrap.has(f)].filter(Boolean).length;
      expect(hits, `field ${f} should be classified exactly once`).toBe(1);
    }
  });
});

describe('analyzeRecording — phase breakdown & accounting', () => {
  it('attributes per-phase share and unaccounted overhead correctly', () => {
    const n = 50;
    // total = rapier + solver + 1.0ms unattributed "other" each frame
    const rapier = new Array(n).fill(8);
    const solver = new Array(n).fill(5);
    const total = new Array(n).fill(14); // 8 + 5 + 1 overhead
    const a = analyzeRecording(
      makeRecording({
        frames: n,
        timing: { totalMs: total, rapierStepMs: rapier, solverUpdateMs: solver },
        rigidBodies: new Array(n).fill(100),
      }),
    );
    expect(a.total.mean).toBeCloseTo(14, 5);
    const rapierPhase = a.phases.find((p) => p.field === 'rapierStepMs')!;
    expect(rapierPhase.shareOfTotalPct).toBeCloseTo((8 / 14) * 100, 4);
    expect(rapierPhase.kind).toBe('leaf');
    // leaf sum is 13ms; 1ms is unaccounted overhead
    expect(a.leafSumMeanMs).toBeCloseTo(13, 5);
    expect(a.unaccountedMeanMs).toBeCloseTo(1, 5);
    // phases sorted by mean: rapier first
    expect(a.phases[0].field).toBe('totalMs'); // totalMs is largest but is wrapper
    const leafOrder = a.phases.filter((p) => p.kind === 'leaf').map((p) => p.field);
    expect(leafOrder[0]).toBe('rapierStepMs');
    expect(leafOrder[1]).toBe('solverUpdateMs');
  });
});

describe('analyzeRecording — body-count scaling', () => {
  it('recovers an injected ms-per-body slope for rapierStep', () => {
    const n = 200;
    const rigid: number[] = [];
    const rapier: number[] = [];
    const total: number[] = [];
    // rapierStep = 2ms + 0.01ms * bodies  →  1.0 ms per 100 bodies
    for (let i = 0; i < n; i += 1) {
      const bodies = 50 + i * 5; // 50 … 1045
      rigid.push(bodies);
      const r = 2 + 0.01 * bodies;
      rapier.push(r);
      total.push(r + 1);
    }
    const a = analyzeRecording(
      makeRecording({ frames: n, timing: { totalMs: total, rapierStepMs: rapier }, rigidBodies: rigid }),
    );
    const rapierScale = a.scaling.find((s) => s.field === 'rapierStepMs')!;
    expect(rapierScale.msPer100Bodies).toBeCloseTo(1.0, 3);
    expect(rapierScale.fit.intercept).toBeCloseTo(2.0, 3);
    expect(rapierScale.fit.r2).toBeCloseTo(1, 6);
  });
});

describe('analyzeRecording — resimulation cost (de-confounded)', () => {
  it('recovers a 2× matched multiplier even when resim frames cluster at high body counts', () => {
    // Build frames across body-count buckets. Within each bucket, resim frames cost
    // exactly 2× the non-resim frames. Resim frames are deliberately biased toward
    // higher body counts (the real-world confound) to prove the bucket-matching
    // controls for it rather than inflating the multiplier.
    const frames: Array<{ bodies: number; resim: boolean }> = [];
    for (let bucket = 0; bucket < 6; bucket += 1) {
      const bodies = 100 * bucket + 50;
      // more resim frames in higher buckets (confound), but both kinds present
      const nNon = 10;
      const nResim = 2 + bucket * 2;
      for (let i = 0; i < nNon; i += 1) frames.push({ bodies, resim: false });
      for (let i = 0; i < nResim; i += 1) frames.push({ bodies, resim: true });
    }
    const n = frames.length;
    const baseMsFor = (bodies: number) => 4 + 0.02 * bodies; // grows with body count
    const total: number[] = [];
    const resimMs: number[] = [];
    const resimPasses: number[] = [];
    const rigid: number[] = [];
    for (const f of frames) {
      const base = baseMsFor(f.bodies);
      rigid.push(f.bodies);
      if (f.resim) {
        total.push(base * 2); // resim frame costs 2× a comparable non-resim frame
        resimMs.push(base); // the extra pass ≈ one base pass
        resimPasses.push(1);
      } else {
        total.push(base);
        resimMs.push(0);
        resimPasses.push(0);
      }
    }
    const a = analyzeRecording(
      makeRecording({
        frames: n,
        timing: { totalMs: total, resimMs, resimPasses },
        rigidBodies: rigid,
      }),
      { resimBodyBucket: 100 },
    );
    expect(a.resim.totalFrames).toBe(n);
    expect(a.resim.resimFrames).toBe(frames.filter((f) => f.resim).length);
    // The headline guard: matched multiplier ≈ 2.0 despite the confound.
    expect(a.resim.matchedResimMultiplier).toBeCloseTo(2.0, 2);
    // wall-clock share = sum(resimMs)/sum(totalMs); both positive and < 50%.
    expect(a.resim.resimShareOfWallClockPct).toBeGreaterThan(0);
    expect(a.resim.resimShareOfWallClockPct).toBeLessThan(50);
    // mean marginal pass cost is a base-pass-sized chunk of time.
    expect(a.resim.meanResimPassMs).toBeGreaterThan(0);
  });

  it('reports zero resim when no frame runs a resim pass', () => {
    const n = 30;
    const a = analyzeRecording(
      makeRecording({ frames: n, timing: { totalMs: new Array(n).fill(5) }, rigidBodies: new Array(n).fill(100) }),
    );
    expect(a.resim.resimFrames).toBe(0);
    expect(a.resim.resimFraction).toBe(0);
    expect(a.resim.matchedResimMultiplier).toBe(0);
  });
});

describe('analyzeRecording — frame budget', () => {
  it('counts the exact fraction of frames over the 60/30 FPS budgets', () => {
    // 100 frames: 30 at 20ms (miss 60, make 30), 10 at 40ms (miss both), 60 at 10ms (ok)
    const total = [
      ...new Array(60).fill(10),
      ...new Array(30).fill(20),
      ...new Array(10).fill(40),
    ];
    const a = analyzeRecording(
      makeRecording({ frames: 100, durationSeconds: 100 / 60, timing: { totalMs: total }, rigidBodies: new Array(100).fill(10) }),
    );
    expect(a.budget.pctMiss60).toBeCloseTo(40, 5); // 30 + 10 over 16.67ms
    expect(a.budget.pctMiss30).toBeCloseTo(10, 5); // 10 over 33.3ms
    expect(a.budget.worstFrame.totalMs).toBeCloseTo(40, 5);
  });
});

describe('analyzeRecording — one-time hitch detection', () => {
  it('flags a single colliderRebuild spike far above its own p95', () => {
    const n = 100;
    const collider = new Array(n).fill(0.5);
    collider[90] = 150; // the startup fracture-cascade hitch
    const total = collider.map((c) => c + 8);
    const a = analyzeRecording(
      makeRecording({ frames: n, timing: { totalMs: total, colliderRebuildMs: collider }, rigidBodies: new Array(n).fill(500) }),
    );
    const hitch = a.hitches.find((h) => h.field === 'colliderRebuildMs');
    expect(hitch).toBeDefined();
    expect(hitch!.frameIndex).toBe(90);
    expect(hitch!.ms).toBeCloseTo(150, 3);
    expect(hitch!.ratio).toBeGreaterThan(10);
  });

  it('does not flag a steady phase with no spike', () => {
    const n = 100;
    const a = analyzeRecording(
      makeRecording({ frames: n, timing: { totalMs: new Array(n).fill(10), rapierStepMs: new Array(n).fill(8) }, rigidBodies: new Array(n).fill(500) }),
    );
    expect(a.hitches.find((h) => h.field === 'rapierStepMs')).toBeUndefined();
  });
});

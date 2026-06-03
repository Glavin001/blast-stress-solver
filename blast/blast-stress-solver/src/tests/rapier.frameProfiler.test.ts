/**
 * Tests for the live frame profiler (data layer + chart draw). Pure TS, no WASM:
 * the buffer/breakdown logic is asserted directly, and the canvas chart is
 * smoke-tested against a stub 2D context (it must execute without throwing and
 * actually draw, for empty / single / many / A/B inputs).
 */
import { describe, expect, it } from "vitest";
import {
  FrameProfilerBuffer,
  computeFrameBreakdown,
  drawFrameProfilerChart,
  frameProfilerToCsv,
  FRAME_PHASES,
  FRAME_PROFILER_EXPORT_SCHEMA,
  type FrameBreakdown,
} from "../rapier/frameProfiler";
import type { CoreProfilerSample } from "../rapier/types";

// Build a sample from a sparse set of fields; computeFrameBreakdown coalesces the
// rest to 0, so we only specify what each case exercises.
function sample(fields: Partial<Record<keyof CoreProfilerSample, number>>): CoreProfilerSample {
  return fields as unknown as CoreProfilerSample;
}

describe("computeFrameBreakdown", () => {
  it("sums leaf phases and assigns the remainder to 'other'", () => {
    const b = computeFrameBreakdown(
      sample({
        totalMs: 10,
        rapierStepMs: 4, // physics
        solverUpdateMs: 2, // stress
        contactDrainMs: 1, // contacts
        splitPlannerMs: 0.5, // splitPlanner
        // tracked = 7.5 -> other = 2.5
      }),
    );
    expect(b.phases.physics).toBe(4);
    expect(b.phases.stress).toBe(2);
    expect(b.phases.contacts).toBe(1);
    expect(b.phases.splitPlanner).toBe(0.5);
    expect(b.phases.other).toBeCloseTo(2.5, 6);
    // leaves + other reconstruct the total
    const sum = FRAME_PHASES.reduce((acc, p) => acc + b.phases[p.key], 0);
    expect(sum).toBeCloseTo(10, 6);
  });

  it("clamps 'other' at zero when leaves slightly exceed total", () => {
    const b = computeFrameBreakdown(sample({ totalMs: 1, rapierStepMs: 2 }));
    expect(b.phases.other).toBe(0);
  });

  it("identifies the dominant phase (the spike cause)", () => {
    const b = computeFrameBreakdown(
      sample({ totalMs: 50, rapierStepMs: 5, fractureMs: 40, bodyCreateMs: 3 }),
    );
    expect(b.dominant).toBe("fracture");
  });

  it("groups topology fields (bodyCreate + colliderRebuild + …) together", () => {
    const b = computeFrameBreakdown(
      sample({ totalMs: 10, bodyCreateMs: 3, colliderRebuildMs: 2, cleanupDisabledMs: 1 }),
    );
    expect(b.phases.topology).toBe(6);
  });

  it("computes projected-old total only when reference timing is present", () => {
    const without = computeFrameBreakdown(sample({ totalMs: 8, splitPlannerMs: 0.1 }));
    expect(without.projectedOldTotalMs).toBeUndefined();

    const withRef = computeFrameBreakdown(
      sample({ totalMs: 8, splitPlannerMs: 0.1, splitPlannerReferenceMs: 60 }),
    );
    // 8 - 0.1 + 60
    expect(withRef.projectedOldTotalMs).toBeCloseTo(67.9, 6);
  });

  it("passes through rigidBodies and resimPasses", () => {
    const b = computeFrameBreakdown(sample({ totalMs: 5, rigidBodies: 42, resimPasses: 2 }));
    expect(b.rigidBodies).toBe(42);
    expect(b.resimPasses).toBe(2);
  });
});

describe("FrameProfilerBuffer", () => {
  it("keeps frames in chronological order and respects capacity", () => {
    const buf = new FrameProfilerBuffer(3);
    for (let i = 0; i < 5; i += 1) buf.push(sample({ frameIndex: i, totalMs: i + 1 }));
    const frames = buf.frames();
    expect(frames.length).toBe(3);
    expect(frames.map((f) => f.frameIndex)).toEqual([2, 3, 4]); // oldest two evicted
    expect(buf.latest()?.frameIndex).toBe(4);
  });

  it("computes rolling stats (mean/p95/max, spikes, fps, worst)", () => {
    const buf = new FrameProfilerBuffer(240, 16.67);
    // 9 cheap frames (5ms) + 1 spike (50ms, fracture-dominated)
    for (let i = 0; i < 9; i += 1) buf.push(sample({ frameIndex: i, totalMs: 5, rapierStepMs: 5 }));
    buf.push(sample({ frameIndex: 9, totalMs: 50, fractureMs: 45, rapierStepMs: 5 }));

    const s = buf.stats();
    expect(s.count).toBe(10);
    expect(s.maxMs).toBe(50);
    expect(s.meanMs).toBeCloseTo((9 * 5 + 50) / 10, 6);
    expect(s.spikeCount).toBe(1); // only the 50ms frame is over 16.67
    expect(s.worst?.frameIndex).toBe(9);
    expect(s.worst?.dominant).toBe("fracture");
    expect(s.worstIndex).toBe(9); // last (newest) frame in the window
    expect(s.fps).toBeGreaterThan(0);
    expect(s.perPhaseMean.physics).toBeCloseTo(5, 6); // 5ms physics every frame
    // per-phase peak: fracture peaked at 45ms (the spike), physics at 5ms
    expect(s.perPhasePeak.fracture).toBe(45);
    expect(s.perPhasePeak.physics).toBe(5);
  });

  it("clear() empties the buffer (incl. raw samples) and stats are well-defined when empty", () => {
    const buf = new FrameProfilerBuffer(8);
    buf.push(sample({ totalMs: 5 }));
    buf.clear();
    expect(buf.frames().length).toBe(0);
    expect(buf.rawFrames().length).toBe(0);
    expect(buf.latest()).toBeNull();
    const s = buf.stats();
    expect(s.count).toBe(0);
    expect(s.fps).toBe(0);
    expect(s.worst).toBeNull();
  });

  it("retains raw samples in parallel with breakdowns (for the data dump)", () => {
    const buf = new FrameProfilerBuffer(3);
    for (let i = 0; i < 4; i += 1) buf.push(sample({ frameIndex: i, totalMs: 5, bufferedExternalContacts: i }));
    const raw = buf.rawFrames();
    expect(raw.length).toBe(3);
    expect(raw.map((s) => s.frameIndex)).toEqual([1, 2, 3]); // oldest evicted, chronological
    // raw samples keep fields the grouped breakdown drops (e.g. contact counts)
    expect((raw[2] as any).bufferedExternalContacts).toBe(3);
  });
});

describe("FrameProfilerBuffer.export + CSV", () => {
  it("produces a self-describing dump: schema, stats, phases, frames, raw samples, meta", () => {
    const buf = new FrameProfilerBuffer(50, 16.67);
    buf.push(sample({ frameIndex: 0, totalMs: 6, rapierStepMs: 4, solverUpdateMs: 2, rigidBodies: 10 }));
    buf.push(sample({ frameIndex: 1, totalMs: 40, fractureMs: 35, rapierStepMs: 5, rigidBodies: 80, resimPasses: 1 }));

    const dump = buf.export({ scenario: "tower-6x12" });
    expect(dump.schema).toBe(FRAME_PROFILER_EXPORT_SCHEMA);
    expect(typeof dump.generatedAt).toBe("string");
    expect(dump.budgetMs).toBeCloseTo(16.67, 2);
    expect(dump.frameCount).toBe(2);
    expect(dump.meta).toEqual({ scenario: "tower-6x12" });
    expect(dump.stats.maxMs).toBe(40);
    expect(dump.phases.map((p) => p.key)).toEqual(FRAME_PHASES.map((p) => p.key));
    expect(dump.frames.length).toBe(2);
    expect(dump.frames[1].dominant).toBe("fracture");
    // raw samples carry every counter (here: rigidBodies/resimPasses preserved)
    expect(dump.samples.length).toBe(2);
    expect((dump.samples[1] as any).rigidBodies).toBe(80);
    // serializable
    expect(() => JSON.stringify(dump)).not.toThrow();
  });

  it("frameProfilerToCsv emits a header + one row per frame with phase columns", () => {
    const buf = new FrameProfilerBuffer(50);
    buf.push(sample({ frameIndex: 7, totalMs: 6, rapierStepMs: 4, solverUpdateMs: 2 }));
    buf.push(sample({ frameIndex: 8, totalMs: 9, fractureMs: 9, splitPlannerMs: 0.1, splitPlannerReferenceMs: 60 }));
    const csv = frameProfilerToCsv(buf.export());
    const lines = csv.trim().split("\n");
    expect(lines.length).toBe(3); // header + 2 rows
    expect(lines[0]).toContain("frameIndex");
    expect(lines[0]).toContain("ms_splitPlanner");
    expect(lines[0]).toContain("projectedOldTotalMs");
    expect(lines[1].split(",")[0]).toBe("7");
    // projected-old present only for the frame that had reference timing
    expect(lines[1].split(",").at(-1)).toBe(""); // frame 7: no A/B
    expect(Number(lines[2].split(",").at(-1))).toBeCloseTo(9 - 0.1 + 60, 1); // frame 8: 68.9
  });
});

describe("drawFrameProfilerChart (stub canvas)", () => {
  function stubCtx() {
    const calls: Record<string, number> = {};
    const rec = (name: string) => {
      calls[name] = (calls[name] ?? 0) + 1;
    };
    const handler: ProxyHandler<any> = {
      get(_t, prop: string) {
        if (prop === "measureText") return () => ({ width: 8 });
        if (prop === "setLineDash") return () => rec("setLineDash");
        // any other accessed property is a drawing method we just count
        return (..._args: unknown[]) => rec(prop);
      },
      set() {
        return true; // fillStyle/strokeStyle/lineWidth assignments
      },
    };
    return { ctx: new Proxy({}, handler) as unknown as CanvasRenderingContext2D, calls };
  }

  function frames(n: number, withSpikeAndRef: boolean): FrameBreakdown[] {
    const out: FrameBreakdown[] = [];
    for (let i = 0; i < n; i += 1) {
      const spike = withSpikeAndRef && i === Math.floor(n / 2);
      out.push(
        computeFrameBreakdown(
          sample({
            frameIndex: i,
            totalMs: spike ? 40 : 6,
            rapierStepMs: spike ? 8 : 4,
            solverUpdateMs: 2,
            fractureMs: spike ? 25 : 0,
            splitPlannerMs: spike ? 0.2 : 0,
            splitPlannerReferenceMs: withSpikeAndRef && spike ? 70 : undefined,
          }),
        ),
      );
    }
    return out;
  }

  it("draws nothing-but-background for an empty series without throwing", () => {
    const { ctx, calls } = stubCtx();
    expect(() => drawFrameProfilerChart(ctx, 300, 120, [])).not.toThrow();
    expect(calls.fillRect ?? 0).toBeGreaterThan(0); // background painted
  });

  it("draws stacked areas + total line for a single frame", () => {
    const { ctx, calls } = stubCtx();
    expect(() => drawFrameProfilerChart(ctx, 300, 120, frames(1, false))).not.toThrow();
    expect(calls.fill ?? 0).toBeGreaterThan(0);
    expect(calls.stroke ?? 0).toBeGreaterThan(0);
  });

  it("draws many frames with a spike and the dashed projected-old line in A/B mode", () => {
    const { ctx, calls } = stubCtx();
    expect(() =>
      drawFrameProfilerChart(ctx, 600, 200, frames(120, true), { showProjectedOld: true }),
    ).not.toThrow();
    expect(calls.fill ?? 0).toBeGreaterThan(0); // phase areas
    expect(calls.stroke ?? 0).toBeGreaterThan(0); // total + guide + projected lines
    expect(calls.setLineDash ?? 0).toBeGreaterThan(0); // dashed guides/projection
  });
});

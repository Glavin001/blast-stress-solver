/**
 * Live frame profiler — turns the per-frame {@link CoreProfilerSample} stream from
 * `core.setProfiler(...)` into a rolling breakdown you can chart in real time, so a
 * dip below the frame budget is immediately attributable to a phase (physics step,
 * stress solve, fracture, split planning, topology edits, snapshots, …).
 *
 * Accounting note: the core's `initialPassMs` / `resimMs` are *wrapper* timers that
 * contain the solve/fracture/body-create work, and `rapierStepMs` is accumulated
 * across both the base step and any resim re-steps. So the breakdown here is built
 * from the **leaf** phases (which are ~mutually exclusive) plus an `other` remainder
 * (`totalMs − Σ leaves`, clamped ≥ 0). Resim cost therefore shows up as larger
 * physics/stress bands on the frames it fires, explained by the `resimPasses` count
 * — rather than as a double-counted band. The true `totalMs` is kept alongside so a
 * chart can draw it as the ground-truth line over the stacked leaves.
 */
import type { CoreProfilerSample } from "./types";

export type FramePhaseKey =
  | "physics"
  | "stress"
  | "contacts"
  | "fracture"
  | "splitPlanner"
  | "topology"
  | "snapshot"
  | "damage"
  | "spawn"
  | "other";

export type FramePhaseDef = {
  key: FramePhaseKey;
  label: string;
  color: string;
  /** Sample fields summed into this phase (leaf timers; wrappers excluded). */
  sampleKeys: Array<keyof CoreProfilerSample>;
};

/** The stacked phases, in draw order (bottom → top). `other` is the clamped
 *  remainder and carries no `sampleKeys`. Colors read on a dark background. */
export const FRAME_PHASES: FramePhaseDef[] = [
  { key: "physics", label: "Physics step", color: "#4f9dff", sampleKeys: ["rapierStepMs"] },
  { key: "stress", label: "Stress solve", color: "#5ad19a", sampleKeys: ["solverUpdateMs"] },
  {
    key: "contacts",
    label: "Contacts / forces",
    color: "#46c7c7",
    sampleKeys: ["contactDrainMs", "externalForceMs", "preStepSweepMs"],
  },
  { key: "fracture", label: "Fracture", color: "#ffd166", sampleKeys: ["fractureMs"] },
  {
    key: "splitPlanner",
    label: "Split planning",
    color: "#ff7ab6",
    sampleKeys: ["splitPlannerMs", "splitQueueMs"],
  },
  {
    key: "topology",
    label: "Body / collider edits",
    color: "#ff9f45",
    sampleKeys: ["bodyCreateMs", "colliderRebuildMs", "rebuildColliderMapMs", "cleanupDisabledMs"],
  },
  {
    key: "snapshot",
    label: "Snapshots",
    color: "#9aa7c7",
    sampleKeys: ["snapshotCaptureMs", "snapshotRestoreMs"],
  },
  {
    key: "damage",
    label: "Damage",
    color: "#e57373",
    sampleKeys: [
      "damageTickMs",
      "damageReplayMs",
      "damagePreviewMs",
      "damageRestoreMs",
      "damageSnapshotMs",
      "damagePreDestroyMs",
      "damageFlushMs",
    ],
  },
  { key: "spawn", label: "Spawn / cleanup", color: "#8d99ae", sampleKeys: ["spawnMs", "projectileCleanupMs"] },
  { key: "other", label: "Other / overhead", color: "rgba(180,190,220,0.28)", sampleKeys: [] },
];

const LEAF_PHASES = FRAME_PHASES.filter((p) => p.key !== "other");

export type FrameBreakdown = {
  frameIndex: number;
  /** wall-clock capture time (ms epoch, from the sample) — for "Ns ago" peaks. */
  timestamp: number;
  totalMs: number;
  /** ms per phase (includes the clamped `other` remainder). */
  phases: Record<FramePhaseKey, number>;
  /** phase contributing the most time this frame — the "cause" of a spike. */
  dominant: FramePhaseKey;
  rigidBodies: number;
  resimPasses: number;
  /** Present only in A/B mode: what `totalMs` would have been with the old
   *  (dense-Hungarian) split planner — `totalMs − splitPlannerMs + reference`. */
  projectedOldTotalMs?: number;
};

function num(sample: CoreProfilerSample, key: keyof CoreProfilerSample): number {
  const v = sample[key] as unknown;
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

export function computeFrameBreakdown(sample: CoreProfilerSample): FrameBreakdown {
  const phases = {} as Record<FramePhaseKey, number>;
  let tracked = 0;
  for (const def of LEAF_PHASES) {
    let sum = 0;
    for (const k of def.sampleKeys) sum += num(sample, k);
    phases[def.key] = sum;
    tracked += sum;
  }
  const totalMs = num(sample, "totalMs");
  phases.other = Math.max(0, totalMs - tracked);

  let dominant: FramePhaseKey = "other";
  let best = -1;
  for (const def of FRAME_PHASES) {
    if (phases[def.key] > best) {
      best = phases[def.key];
      dominant = def.key;
    }
  }

  const breakdown: FrameBreakdown = {
    frameIndex: num(sample, "frameIndex"),
    timestamp: num(sample, "timestamp"),
    totalMs,
    phases,
    dominant,
    rigidBodies: num(sample, "rigidBodies"),
    resimPasses: num(sample, "resimPasses"),
  };

  const refMs = sample.splitPlannerReferenceMs;
  if (typeof refMs === "number" && Number.isFinite(refMs)) {
    breakdown.projectedOldTotalMs = Math.max(0, totalMs - phases.splitPlanner + refMs);
  }
  return breakdown;
}

export type FrameProfilerStats = {
  count: number;
  fps: number; // derived from mean sim-step total (1000 / meanMs)
  meanMs: number;
  p95Ms: number;
  maxMs: number;
  /** frames over the budget, and over double the budget. */
  spikeCount: number;
  spikeCount2x: number;
  perPhaseMean: Record<FramePhaseKey, number>;
  /** peak (max over the window) per phase — the worst that phase ever cost. */
  perPhasePeak: Record<FramePhaseKey, number>;
  /** worst (max-total) frame in the window, and its position (0 = oldest). */
  worst: FrameBreakdown | null;
  worstIndex: number;
};

/** Fixed-capacity ring buffer of per-frame breakdowns + rolling stats. Retains
 *  the raw {@link CoreProfilerSample}s alongside the computed breakdowns so a full
 *  data dump (every counter, not just the grouped phases) can be exported. */
export class FrameProfilerBuffer {
  readonly capacity: number;
  readonly budgetMs: number;
  private ring: FrameBreakdown[] = [];
  private rawRing: CoreProfilerSample[] = [];
  private head = 0;

  constructor(capacity = 240, budgetMs = 1000 / 60) {
    this.capacity = Math.max(1, capacity);
    this.budgetMs = budgetMs;
  }

  /** Ingest a raw profiler sample; returns the computed breakdown. The sample is
   *  retained by reference (the core allocates a fresh one each frame). */
  push(sample: CoreProfilerSample): FrameBreakdown {
    const b = computeFrameBreakdown(sample);
    if (this.ring.length < this.capacity) {
      this.ring.push(b);
      this.rawRing.push(sample);
    } else {
      this.ring[this.head] = b;
      this.rawRing[this.head] = sample;
      this.head = (this.head + 1) % this.capacity;
    }
    return b;
  }

  /** Breakdowns in chronological (oldest → newest) order. */
  frames(): FrameBreakdown[] {
    if (this.ring.length < this.capacity) return this.ring.slice();
    return [...this.ring.slice(this.head), ...this.ring.slice(0, this.head)];
  }

  /** Raw profiler samples in chronological order (parallel to {@link frames}). */
  rawFrames(): CoreProfilerSample[] {
    if (this.rawRing.length < this.capacity) return this.rawRing.slice();
    return [...this.rawRing.slice(this.head), ...this.rawRing.slice(0, this.head)];
  }

  latest(): FrameBreakdown | null {
    if (this.ring.length === 0) return null;
    const idx = this.ring.length < this.capacity ? this.ring.length - 1 : (this.head + this.capacity - 1) % this.capacity;
    return this.ring[idx];
  }

  clear(): void {
    this.ring = [];
    this.rawRing = [];
    this.head = 0;
  }

  stats(): FrameProfilerStats {
    const frames = this.frames();
    const empty: FrameProfilerStats = {
      count: 0,
      fps: 0,
      meanMs: 0,
      p95Ms: 0,
      maxMs: 0,
      spikeCount: 0,
      spikeCount2x: 0,
      perPhaseMean: Object.fromEntries(FRAME_PHASES.map((p) => [p.key, 0])) as Record<FramePhaseKey, number>,
      perPhasePeak: Object.fromEntries(FRAME_PHASES.map((p) => [p.key, 0])) as Record<FramePhaseKey, number>,
      worst: null,
      worstIndex: -1,
    };
    if (frames.length === 0) return empty;

    const totals = frames.map((f) => f.totalMs);
    const sorted = [...totals].sort((a, b) => a - b);
    const sum = totals.reduce((a, b) => a + b, 0);
    const meanMs = sum / totals.length;
    const p95Ms = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
    const maxMs = sorted[sorted.length - 1];

    const perPhaseMean = Object.fromEntries(FRAME_PHASES.map((p) => [p.key, 0])) as Record<FramePhaseKey, number>;
    const perPhasePeak = Object.fromEntries(FRAME_PHASES.map((p) => [p.key, 0])) as Record<FramePhaseKey, number>;
    let worst = frames[0];
    let worstIndex = 0;
    frames.forEach((f, i) => {
      for (const def of FRAME_PHASES) {
        perPhaseMean[def.key] += f.phases[def.key];
        if (f.phases[def.key] > perPhasePeak[def.key]) perPhasePeak[def.key] = f.phases[def.key];
      }
      if (f.totalMs > worst.totalMs) {
        worst = f;
        worstIndex = i;
      }
    });
    for (const def of FRAME_PHASES) perPhaseMean[def.key] /= frames.length;

    return {
      count: frames.length,
      fps: meanMs > 0 ? 1000 / meanMs : 0,
      meanMs,
      p95Ms,
      maxMs,
      spikeCount: totals.filter((t) => t > this.budgetMs).length,
      spikeCount2x: totals.filter((t) => t > this.budgetMs * 2).length,
      perPhaseMean,
      perPhasePeak,
      worst,
      worstIndex,
    };
  }

  /** A complete, self-describing snapshot of the captured window for export: the
   *  computed per-frame breakdowns, the raw profiler samples (every counter), the
   *  rolling stats, the phase legend, and optional caller metadata. */
  export(meta?: Record<string, unknown>): FrameProfilerExport {
    return {
      schema: FRAME_PROFILER_EXPORT_SCHEMA,
      generatedAt: new Date().toISOString(),
      budgetMs: this.budgetMs,
      capacity: this.capacity,
      frameCount: this.ring.length,
      meta,
      stats: this.stats(),
      phases: FRAME_PHASES.map((p) => ({ key: p.key, label: p.label, color: p.color })),
      frames: this.frames(),
      samples: this.rawFrames(),
    };
  }
}

export const FRAME_PROFILER_EXPORT_SCHEMA = "blast-frame-profiler/v1" as const;

export type FrameProfilerExport = {
  schema: typeof FRAME_PROFILER_EXPORT_SCHEMA;
  /** ISO timestamp of the export. */
  generatedAt: string;
  budgetMs: number;
  capacity: number;
  frameCount: number;
  /** Caller-supplied context (scenario, config, build info, …). */
  meta?: Record<string, unknown>;
  stats: FrameProfilerStats;
  phases: Array<{ key: FramePhaseKey; label: string; color: string }>;
  frames: FrameBreakdown[];
  /** Raw per-frame `CoreProfilerSample`s — every counter, for deep analysis. */
  samples: CoreProfilerSample[];
};

/** Flatten an export's per-frame breakdowns to CSV (one row per frame: total,
 *  each phase, dominant, bodies, resim passes, projected-old). Handy for
 *  spreadsheets / quick plots; the JSON export carries the full raw samples. */
export function frameProfilerToCsv(data: FrameProfilerExport): string {
  const phaseKeys = FRAME_PHASES.map((p) => p.key);
  const header = [
    "frameIndex",
    "totalMs",
    ...phaseKeys.map((k) => `ms_${k}`),
    "dominant",
    "rigidBodies",
    "resimPasses",
    "projectedOldTotalMs",
  ];
  const rows = data.frames.map((f) => [
    f.frameIndex,
    round(f.totalMs),
    ...phaseKeys.map((k) => round(f.phases[k])),
    f.dominant,
    f.rigidBodies,
    f.resimPasses,
    f.projectedOldTotalMs != null ? round(f.projectedOldTotalMs) : "",
  ]);
  return [header, ...rows].map((r) => r.join(",")).join("\n");
}

function round(v: number): number {
  return Math.round(v * 1000) / 1000;
}


export const phaseLabel = (key: FramePhaseKey): string =>
  FRAME_PHASES.find((p) => p.key === key)?.label ?? key;
export const phaseColor = (key: FramePhaseKey): string =>
  FRAME_PHASES.find((p) => p.key === key)?.color ?? "#fff";

export type DrawChartOptions = {
  budgetMs?: number;
  /** Draw the dashed "with old split planner" projection where A/B data exists. */
  showProjectedOld?: boolean;
  /** Override the y-axis max (ms). Defaults to auto from the data + budget. */
  yMaxMs?: number;
  background?: string;
  /** Mark the window's worst (max-total) frame with a persistent marker + a
   *  peak line, so a spike that has scrolled left stays visible. Default true. */
  markPeak?: boolean;
};

/**
 * Render a stacked per-phase area chart of the frames over time onto a 2D canvas
 * context: phases stacked bottom→top, the true `totalMs` as a white line, budget
 * (60 fps) and 2× budget guide lines, and — in A/B mode — a dashed amber line for
 * the projected old-planner frame time. Pure (operates only on the passed `ctx`),
 * so it is environment-agnostic and unit-testable with a stub context.
 */
export function drawFrameProfilerChart(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  frames: FrameBreakdown[],
  opts: DrawChartOptions = {},
): void {
  const budgetMs = opts.budgetMs ?? 1000 / 60;
  const bg = opts.background ?? "#0a0d16";
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  if (frames.length === 0) return;

  const padL = 4;
  const padR = 4;
  const padT = 6;
  const padB = 6;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);

  // y-axis max: a bit above the largest of (total, projected-old, 2×budget).
  let dataMax = budgetMs * 2;
  for (const f of frames) {
    if (f.totalMs > dataMax) dataMax = f.totalMs;
    if (opts.showProjectedOld && f.projectedOldTotalMs && f.projectedOldTotalMs > dataMax) {
      dataMax = f.projectedOldTotalMs;
    }
  }
  const yMax = opts.yMaxMs ?? dataMax * 1.1;

  const n = frames.length;
  const xAt = (i: number) => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (ms: number) => padT + plotH - Math.min(1, ms / yMax) * plotH;

  // Stacked phase areas (bottom → top).
  let lower = new Array(n).fill(0);
  for (const def of FRAME_PHASES) {
    const upper = frames.map((f, i) => lower[i] + f.phases[def.key]);
    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(lower[0]));
    for (let i = 0; i < n; i += 1) ctx.lineTo(xAt(i), yAt(upper[i]));
    for (let i = n - 1; i >= 0; i -= 1) ctx.lineTo(xAt(i), yAt(lower[i]));
    ctx.closePath();
    ctx.fillStyle = def.color;
    ctx.fill();
    lower = upper;
  }

  // Budget guide lines.
  const guide = (ms: number, color: string, dash: number[]) => {
    if (ms > yMax) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(padL, yAt(ms));
    ctx.lineTo(padL + plotW, yAt(ms));
    ctx.stroke();
    ctx.setLineDash([]);
  };
  guide(budgetMs, "rgba(120,230,160,0.55)", [4, 3]); // 60 fps
  guide(budgetMs * 2, "rgba(255,180,90,0.4)", [2, 4]); // 30 fps

  // Projected old-planner total (dashed amber), where A/B data exists.
  if (opts.showProjectedOld) {
    ctx.strokeStyle = "#ffb454";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i += 1) {
      const v = frames[i].projectedOldTotalMs;
      if (v == null) {
        started = false;
        continue;
      }
      const x = xAt(i);
      const y = yAt(v);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // True total frame time (solid white line) — the ground truth over the stack.
  ctx.strokeStyle = "rgba(245,247,255,0.9)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const x = xAt(i);
    const y = yAt(frames[i].totalMs);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Persistent peak marker: the window's worst frame stays flagged even after it
  // scrolls left — a peak line at its height + a vertical guide + a ring at the
  // top, colored by the phase that caused it.
  if (opts.markPeak !== false) {
    let peakIdx = 0;
    for (let i = 1; i < n; i += 1) if (frames[i].totalMs > frames[peakIdx].totalMs) peakIdx = i;
    const peak = frames[peakIdx];
    if (peak.totalMs > 0) {
      const px = xAt(peakIdx);
      const py = yAt(peak.totalMs);
      const peakColor = FRAME_PHASES.find((p) => p.key === peak.dominant)?.color ?? "#ff5d6c";
      // peak-height line
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      // vertical guide to the peak
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      // ring at the peak, in the cause's color
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = peakColor;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.strokeStyle = peakColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

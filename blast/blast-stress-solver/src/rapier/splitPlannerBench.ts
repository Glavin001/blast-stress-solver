/**
 * Shared benchmark driver for the split planner — one source of truth used by
 * both the CLI micro-benchmark (`scripts/bench-split-planner.mjs`) and the
 * browser demo (`split-planner-bench.html`), mirroring the Rust harness's
 * "everything shares one driver" principle.
 *
 * It A/Bs the shipping planner ({@link planSplitMigration}, connected-component
 * decomposition) against the reference planner
 * ({@link planSplitMigrationReference}, the original single dense Hungarian) on
 * the two fracture regimes that drive topology edits, and verifies that the two
 * produce an equally-optimal assignment — so any measured speedup is genuinely
 * free (same result, less time), never bought by doing less work.
 */
import {
  planSplitMigration,
  planSplitMigrationReference,
  type ExistingBodyState,
  type PlannerChild,
  type SplitMigrationPlan,
} from "./splitMigrator";

export type PlannerVariant = "production" | "reference";
export type PlannerScenario = "shatter" | "merge";

export type BenchCase = {
  bodies: ExistingBodyState[];
  children: PlannerChild[];
};

/**
 * Shatter (1×N) — one parent body lets go into N single-node fragments. The
 * common forward-fracture / cascade case: the shipping planner solves it with a
 * single argmax (one-body component); the reference pads to an N×N Hungarian.
 */
export function buildShatterCase(n: number): BenchCase {
  const nodes = new Set<number>();
  for (let i = 0; i < n; i += 1) nodes.add(i);
  const bodies: ExistingBodyState[] = [{ handle: 1, nodeIndices: nodes, isFixed: false }];
  const children: PlannerChild[] = [];
  for (let i = 0; i < n; i += 1) {
    children.push({ index: i, actorIndex: i, nodes: [i], isSupport: false });
  }
  return { bodies, children };
}

/**
 * Merge / reparent (M→M/2) — `pairs` two-node children, each straddling two of
 * `2*pairs` distinct single-node bodies (child k pulls nodes 2k, 2k+1 from
 * bodies 2k, 2k+1). Every child competes for two bodies, so the reference runs
 * one dense Hungarian padded to (2·pairs)². The shipping planner sees `pairs`
 * tiny independent components (one child + two bodies each) and resolves each
 * with a single argmax — the case the connected-component decomposition exists
 * for.
 */
export function buildMergeCase(pairs: number): BenchCase {
  const bodyCount = pairs * 2;
  const bodies: ExistingBodyState[] = [];
  for (let b = 0; b < bodyCount; b += 1) {
    bodies.push({ handle: b + 1, nodeIndices: new Set([b]), isFixed: false });
  }
  const children: PlannerChild[] = [];
  for (let k = 0; k < pairs; k += 1) {
    children.push({ index: k, actorIndex: k, nodes: [2 * k, 2 * k + 1], isSupport: false });
  }
  return { bodies, children };
}

export function buildCase(scenario: PlannerScenario, n: number): BenchCase {
  return scenario === "shatter" ? buildShatterCase(n) : buildMergeCase(n);
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

const PLANNERS: Record<PlannerVariant, (b: ExistingBodyState[], c: PlannerChild[]) => SplitMigrationPlan> = {
  production: planSplitMigration,
  reference: planSplitMigrationReference,
};

/**
 * Time one planner on one case: warm up, then call repeatedly until a wall-clock
 * budget is hit (or a hard iteration cap), and report ms per plan. Auto-scaling
 * keeps cheap (sub-ms) and expensive (multi-100ms) variants both well-sampled.
 */
export function timePlanner(
  variant: PlannerVariant,
  testCase: BenchCase,
  opts: { budgetMs?: number; maxIters?: number; warmup?: number } = {},
): { msPerPlan: number; iters: number } {
  const { budgetMs = 50, maxIters = 500, warmup = 2 } = opts;
  const plan = PLANNERS[variant];
  const { bodies, children } = testCase;
  for (let i = 0; i < warmup; i += 1) plan(bodies, children);

  let iters = 0;
  let elapsed = 0;
  const t0 = now();
  do {
    plan(bodies, children);
    iters += 1;
    elapsed = now() - t0;
  } while (elapsed < budgetMs && iters < maxIters);
  return { msPerPlan: elapsed / iters, iters };
}

/** Total node overlap of the reused (body, child) pairs — the quantity the
 *  planner maximizes, and the correctness invariant we compare across variants. */
export function planTotalOverlap(
  plan: SplitMigrationPlan,
  bodies: ExistingBodyState[],
  children: PlannerChild[],
): number {
  const byHandle = new Map(bodies.map((b) => [b.handle, b]));
  const byIndex = new Map(children.map((c) => [c.index, c]));
  let total = 0;
  for (const r of plan.reuse) {
    const body = byHandle.get(r.bodyHandle);
    const child = byIndex.get(r.childIndex);
    if (!body || !child) continue;
    for (const n of child.nodes) if (body.nodeIndices.has(n)) total += 1;
  }
  return total;
}

export type BenchRow = {
  scenario: PlannerScenario;
  /** number of split children (the "N" in 1×N / M→N) */
  n: number;
  bodyCount: number;
  productionMs: number;
  /** null when the reference is skipped (would be too slow at this size) */
  referenceMs: number | null;
  /** referenceMs / productionMs (how much the optimization saved) */
  speedup: number | null;
  reuseProduction: number;
  reuseReference: number | null;
  overlapProduction: number;
  overlapReference: number | null;
  /** true when both planners reached the same optimum (speedup is free) */
  optimalEquivalent: boolean | null;
};

export function measureRow(
  scenario: PlannerScenario,
  n: number,
  opts: { budgetMs?: number; maxIters?: number; referenceMaxN?: number } = {},
): BenchRow {
  const { referenceMaxN = Infinity } = opts;
  const testCase = buildCase(scenario, n);
  const { bodies, children } = testCase;

  const prodPlan = planSplitMigration(bodies, children);
  const overlapProduction = planTotalOverlap(prodPlan, bodies, children);
  const prod = timePlanner("production", testCase, opts);

  const runReference = n <= referenceMaxN;
  let referenceMs: number | null = null;
  let overlapReference: number | null = null;
  let reuseReference: number | null = null;
  if (runReference) {
    const refPlan = planSplitMigrationReference(bodies, children);
    overlapReference = planTotalOverlap(refPlan, bodies, children);
    reuseReference = refPlan.reuse.length;
    const ref = timePlanner("reference", testCase, opts);
    referenceMs = ref.msPerPlan;
  }

  return {
    scenario,
    n,
    bodyCount: bodies.length,
    productionMs: prod.msPerPlan,
    referenceMs,
    speedup: referenceMs != null ? referenceMs / prod.msPerPlan : null,
    reuseProduction: prodPlan.reuse.length,
    reuseReference,
    overlapProduction,
    overlapReference,
    optimalEquivalent: overlapReference != null ? overlapReference === overlapProduction : null,
  };
}

export type BenchOptions = {
  scenarios?: PlannerScenario[];
  sizes?: number[];
  budgetMs?: number;
  maxIters?: number;
  /** Skip the (cubic) reference planner above this child count to bound runtime. */
  referenceMaxN?: number;
  /** Called after each row is measured (for live progress / streaming UIs). */
  onRow?: (row: BenchRow) => void;
};

export const DEFAULT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048];

/** Synchronous sweep over scenarios × sizes. Used by the CLI; the browser demo
 *  drives {@link measureRow} itself so it can yield to the event loop between
 *  rows and keep the page responsive. */
export function runSplitPlannerBenchmark(opts: BenchOptions = {}): { rows: BenchRow[] } {
  const {
    scenarios = ["shatter", "merge"],
    sizes = DEFAULT_SIZES,
    budgetMs,
    maxIters,
    referenceMaxN = 512,
    onRow,
  } = opts;

  const rows: BenchRow[] = [];
  for (const scenario of scenarios) {
    for (const n of sizes) {
      const row = measureRow(scenario, n, { budgetMs, maxIters, referenceMaxN });
      rows.push(row);
      onRow?.(row);
    }
  }
  return { rows };
}

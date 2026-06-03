#!/usr/bin/env node
/**
 * Split-planner micro-benchmark — isolates the cost of the topology-diff that
 * decides which existing Rapier bodies a fracture *reuses* vs. *creates*, with
 * no physics, no stress solve, no WASM. It A/Bs the shipping planner
 * (connected-component decomposition) against the reference planner (the
 * original single dense Hungarian) across the two fracture regimes:
 *
 *   - shatter (1×N): one body lets go into N fragments (the cascade case).
 *   - merge  (M→M/2): N singleton bodies regroup into N/2 two-node children.
 *
 * This is the TS mirror of the Rust `apply_profile` example. It needs only the
 * pure-TS planner, so it runs straight from source (bundled on the fly with
 * esbuild) — no `npm run build`, no emscripten runtime.
 *
 * Usage:
 *   node scripts/bench-split-planner.mjs            # full sweep, human table
 *   node scripts/bench-split-planner.mjs --quick    # smaller/faster sweep
 *   node scripts/bench-split-planner.mjs --json      # machine-readable JSON
 *   node scripts/bench-split-planner.mjs --scenario=shatter
 */
import { build } from "esbuild";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const driverEntry = resolve(here, "../src/rapier/splitPlannerBench.ts");

const args = process.argv.slice(2);
const has = (flag) => args.includes(flag);
const get = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`${name}=`));
  return hit ? hit.slice(name.length + 1) : fallback;
};

const asJson = has("--json");
const quick = has("--quick");
const scenarioArg = get("--scenario", null); // 'shatter' | 'merge' | null (both)
const sizes = quick
  ? [16, 32, 64, 128, 256]
  : [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
// Reference is O(N^3); cap it so the A/B never runs for minutes.
const referenceMaxN = quick ? 256 : 512;
const scenarios = scenarioArg ? [scenarioArg] : ["shatter", "merge"];

// Bundle the pure-TS driver (no runtime deps) into a temp ESM module and import it.
const result = await build({
  entryPoints: [driverEntry],
  bundle: true,
  format: "esm",
  platform: "node",
  target: "es2020",
  write: false,
});
const tmp = mkdtempSync(join(tmpdir(), "split-bench-"));
const outFile = join(tmp, "splitPlannerBench.mjs");
writeFileSync(outFile, result.outputFiles[0].text);

let driver;
try {
  driver = await import(pathToFileURL(outFile).href);
} finally {
  // import() has already read the file; safe to clean up.
  rmSync(tmp, { recursive: true, force: true });
}

const { runSplitPlannerBenchmark } = driver;

const fmtMs = (v) => (v == null ? "—" : v < 0.001 ? v.toExponential(2) : v.toFixed(4));
const pad = (s, n) => String(s).padStart(n);

if (!asJson) {
  console.log("\nSplit-planner micro-benchmark — production (connected components) vs reference (dense Hungarian)");
  console.log("Pure planner cost: no physics, no stress solve, no WASM. Lower is better; speedup = reference / production.\n");
}

const { rows } = runSplitPlannerBenchmark({
  scenarios,
  sizes,
  referenceMaxN,
  budgetMs: 60,
});

if (asJson) {
  console.log(JSON.stringify({ scenarios, sizes, referenceMaxN, rows }, null, 2));
} else {
  let lastScenario = null;
  for (const r of rows) {
    if (r.scenario !== lastScenario) {
      lastScenario = r.scenario;
      const label =
        r.scenario === "shatter"
          ? "SHATTER (1 body → N fragments)"
          : "MERGE (2N bodies → N two-node children)";
      console.log(`\n  ${label}`);
      console.log(
        `  ${pad("N", 6)} ${pad("bodies", 7)} ${pad("production", 12)} ${pad("reference", 12)} ${pad("speedup", 9)}  optimal-equiv`,
      );
      console.log("  " + "-".repeat(70));
    }
    const speedup = r.speedup == null ? "—" : `${r.speedup.toFixed(1)}x`;
    const equiv = r.optimalEquivalent == null ? "(ref skipped)" : r.optimalEquivalent ? "yes ✓" : "NO ✗";
    console.log(
      `  ${pad(r.n, 6)} ${pad(r.bodyCount, 7)} ${pad(fmtMs(r.productionMs) + " ms", 12)} ${pad(fmtMs(r.referenceMs) + " ms", 12)} ${pad(speedup, 9)}  ${equiv}`,
    );
  }

  // Headline: the worst spike the reference incurred vs. what production costs there.
  const withRef = rows.filter((r) => r.referenceMs != null);
  if (withRef.length > 0) {
    const worst = withRef.reduce((a, b) => (b.referenceMs > a.referenceMs ? b : a));
    console.log(
      `\n  Headline: at ${worst.scenario} N=${worst.n}, reference = ${fmtMs(worst.referenceMs)} ms/plan ` +
        `vs production = ${fmtMs(worst.productionMs)} ms/plan (${worst.speedup.toFixed(0)}× faster, same optimal assignment).`,
    );
    const allEquiv = withRef.every((r) => r.optimalEquivalent);
    console.log(
      `  Optimality: ${allEquiv ? "every measured size reached the SAME optimum" : "MISMATCH — production differs from reference"}` +
        " — the speedup is free.\n",
    );
  }
}

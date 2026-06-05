#!/usr/bin/env node
/**
 * Mini-city destruction stress harness.
 *
 * Runs a SUITE of large multi-building "city" scenarios (many islands, dense /
 * sparse destruction, cascading collapse, fractured bridge) through the full
 * destruction pipeline headlessly, collects the per-frame profiler samples, and
 * prints a ranked "costliest steps and sub-steps" report (plus optional JSON).
 *
 * This is a MEASUREMENT tool for finding bottlenecks — it does not change any
 * simulation behavior (same world, same solver iterations, same resim).
 *
 * Usage:
 *   node scripts/stress-city.mjs                         # small tier, all scenarios, table
 *   node scripts/stress-city.mjs --medium                # bigger grids
 *   node scripts/stress-city.mjs --large                 # bigger still (heavy)
 *   node scripts/stress-city.mjs --scenario=cascade      # one scenario (comma-list ok)
 *   node scripts/stress-city.mjs --json                  # also print JSON
 *   node scripts/stress-city.mjs --json --out=report.json
 *
 * Requires a full build (`npm run build`) so dist/stress_solver.wasm + dist/rapier.js exist.
 */
import { build } from 'esbuild';
import { existsSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, '../dist');

if (!existsSync(resolve(distDir, 'stress_solver.wasm')) || !existsSync(resolve(distDir, 'rapier.js'))) {
  console.error('ERROR: dist build not found. Run: npm run build');
  process.exit(1);
}

// ── args ──────────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const has = (f) => args.includes(f);
const get = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`${name}=`));
  return hit ? hit.slice(name.length + 1) : fallback;
};
const tier = has('--xl') ? 'xl' : has('--large') ? 'large' : has('--medium') ? 'medium' : 'small';
const only = (get('--scenario', '') || '').split(',').map((s) => s.trim()).filter(Boolean);
const asJson = has('--json');
const outFile = get('--out', null);
// A/B the island-aware skip-settled solve vs the default whole-graph solve.
const abIsland = has('--ab-island');

// ── bundle the pure-TS suite (engine injected; three/rapier externalized) ─────
const entry = resolve(here, '../src/tests/stress/runSuite.ts');
const bundled = await build({
  entryPoints: [entry],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'es2020',
  write: false,
  // Externalize the same things tsup does: the heavy peer libs (resolved from
  // node_modules at runtime) and the WASM runtime entry points. The high-rise
  // builds in pure-JS 'proximity' mode, so the WASM loader is never invoked.
  external: [
    'three',
    '@dgreenheck/three-pinata',
    '@dimforge/rapier3d-compat',
    './stress_solver.cjs',
    './stress_solver.mjs',
    './stress_solver.browser.mjs',
  ],
});
// Write the bundle INSIDE the package dir so Node resolves the externalized
// peer deps (three, rapier, ...) from this package's node_modules at runtime.
const tmp = mkdtempSync(resolve(here, '..', '.stress-city-'));
const bundleFile = join(tmp, 'runSuite.mjs');
writeFileSync(bundleFile, bundled.outputFiles[0].text);

let suite;
try {
  suite = await import(pathToFileURL(bundleFile).href);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
const { runSuite, runIslandAB, printReport, toJsonReport, printABReport, abToJsonReport } = suite;

// ── engine from dist (ESM build resolves the WASM URL cleanly in Node) ─────────
const { buildDestructibleCore } = await import(resolve(distDir, 'rapier.js'));

// Core emits debug/warn logs to stdout; mute so they don't pollute JSON.
const origDebug = console.debug;
const origWarn = console.warn;
if (asJson) {
  console.debug = () => {};
}

if (abIsland) {
  const cmp = await runIslandAB({
    buildCore: buildDestructibleCore,
    tier,
    scenario: only[0] || 'manyIslands',
    onProgress: (label) => {
      if (!asJson) console.error(`  ${label} ...`);
    },
  });
  console.debug = origDebug;
  console.warn = origWarn;
  if (!asJson || outFile) printABReport(cmp);
  if (asJson) {
    const json = JSON.stringify(abToJsonReport(cmp), null, 2);
    if (outFile) {
      writeFileSync(resolve(process.cwd(), outFile), json);
      console.error(`  wrote ${outFile}`);
    } else {
      console.log(json);
    }
  }
} else {
  const results = await runSuite({
    buildCore: buildDestructibleCore,
    tier,
    only,
    onProgress: (name, i, total) => {
      if (!asJson) console.error(`  [${i + 1}/${total}] ${name} ...`);
    },
  });

  console.debug = origDebug;
  console.warn = origWarn;

  if (!asJson || outFile) {
    printReport(results);
  }
  if (asJson) {
    const json = JSON.stringify(toJsonReport(results), null, 2);
    if (outFile) {
      writeFileSync(resolve(process.cwd(), outFile), json);
      console.error(`  wrote ${outFile}`);
    } else {
      console.log(json);
    }
  }
}

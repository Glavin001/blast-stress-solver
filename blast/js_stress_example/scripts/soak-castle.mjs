#!/usr/bin/env node
/**
 * soak-castle.mjs — headless stability soak for the brick-castle demo.
 *
 * The castle is a large *anchored* structure, so this checks two distinct
 * failure modes over time:
 *   1. It must STAND at rest. With small auto-bond contact areas the structure
 *      can silently crumble under its own weight if `materialScale`/tier areas
 *      are mis-tuned — so the "settle" phase asserts almost no bonds break and
 *      nothing moves fast while untouched. (The anchored analog of a vehicle's
 *      "solid at rest" check.)
 *   2. Sieges must COLLAPSE it without EXPLODING. Overlapping fractured chunks
 *      blow apart if many detach into a debris-vs-body penetration, so every
 *      siege phase asserts no body exceeds ~120 m/s (and 0 bodies > 60 m/s).
 *
 * Probe: `window.__castleDemo.metrics()`. Siege tools are driven directly via
 * `window.__castleDemo.{fireVolley,wreckingBall,boulderStorm}()` (no mouse).
 * Builds once and runs all phases on the same castle (a rebuild is ~15–20 s).
 *
 * Usage:
 *   1) serve the demos:          npm run serve:demos        (repo root)
 *   2) install the browser once: npx playwright install chromium
 *   3) run:                      node scripts/soak-castle.mjs [--url <url>]
 */
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const getArg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const URL = getArg('--url', 'http://localhost:8000/blast/js_stress_example/brick-castle.html');
const SPEED_LIMIT = 120;       // m/s; faster than a fast projectile ⇒ explosion
const SETTLE_BREAK_FRAC = 0.03; // ≤3% of bonds may break while merely settling

const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));

const metrics = () => page.evaluate(() => globalThis.__castleDemo?.metrics?.() || null);
const ready = async () => { const m = await metrics(); return m && m.ready; };
async function waitReady() {
  // The castle auto-bonds ~2–3k bricks on load — allow generous time.
  for (let i = 0; i < 140; i++) { if (await ready()) return true; await page.waitForTimeout(500); }
  return false;
}
const call = (fn, ...a) => page.evaluate(([fn, a]) => globalThis.__castleDemo?.[fn]?.(...a), [fn, a]);

const failures = [];
async function watch(secs) {
  let worst = { maxSpeed: 0, fastBodies: 0 };
  const steps = Math.max(1, Math.round(secs / 0.2));
  for (let i = 0; i < steps; i++) {
    const x = await metrics();
    if (x && x.ready) {
      worst.maxSpeed = Math.max(worst.maxSpeed, x.maxSpeed);
      worst.fastBodies = Math.max(worst.fastBodies, x.fastBodies);
    }
    await page.waitForTimeout(200);
  }
  return worst;
}
function logLine(status, label, worst, m) {
  console.log(
    `  [${status}] ${label}: peakSpeed=${worst.maxSpeed.toFixed(1)} ` +
    `peakFast=${worst.fastBodies} bodies=${m?.bodies ?? '?'} ` +
    `broken=${m?.brokenBonds ?? '?'}/${m?.initialBonds ?? '?'} ` +
    `liveColliders=${m?.activeColliders ?? '?'} islandsSkipped=${m?.islandsSkipped ?? '?'}`,
  );
}
function assertStable(label, worst, m) {
  const exploded = worst.fastBodies > 0 || worst.maxSpeed > SPEED_LIMIT;
  logLine(exploded ? 'EXPLODED' : 'ok', label, worst, m);
  if (exploded) failures.push(`${label} (explosion)`);
}

await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
if (!(await waitReady())) { console.error('castle did not initialize in time'); await browser.close(); process.exit(2); }
console.log(`Soaking ${URL}\n`);

// 1) Settle untouched — must STAND (no spontaneous collapse, no fast bodies).
{
  const w = await watch(6);
  const m = await metrics();
  const breakFrac = m && m.initialBonds ? m.brokenBonds / m.initialBonds : 0;
  const collapsed = breakFrac > SETTLE_BREAK_FRAC;
  logLine(collapsed ? 'COLLAPSED' : (w.fastBodies ? 'EXPLODED' : 'ok'), 'settle (no input)', w, m);
  if (collapsed) failures.push(`settle: spontaneous collapse (${(breakFrac * 100).toFixed(1)}% bonds broke at rest)`);
  if (w.fastBodies > 0 || w.maxSpeed > SPEED_LIMIT) failures.push('settle (explosion)');
}

// 2) Cannonball volley — a localized breach.
await call('fireVolley', 6);
{ const w = await watch(5); assertStable('cannonball volley', w, await metrics()); }

// 3) Wrecking ball — drop a wall section.
for (let i = 0; i < 3; i++) { await call('wreckingBall'); await page.waitForTimeout(1200); }
{ const w = await watch(6); assertStable('wrecking ball', w, await metrics()); }

// 4) Boulder storm — mass destruction / load test.
await call('boulderStorm', 36);
{ const w = await watch(8); assertStable('boulder storm', w, await metrics()); }

console.log('');
const final = await metrics();
if (final) console.log(`final: bodies=${final.bodies} brokenBonds=${final.brokenBonds}/${final.initialBonds} chunks=${final.chunks}`);
if (pageErrors.length) { console.log('page errors:'); pageErrors.slice(0, 10).forEach((e) => console.log('  ' + e)); }
await browser.close();

if (failures.length) {
  console.error(`\nSOAK FAILED — ${failures.join('; ')}`);
  process.exit(1);
}
console.log('\nSOAK PASSED — castle stands at rest and collapses without exploding.');

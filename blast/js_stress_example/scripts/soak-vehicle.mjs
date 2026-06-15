#!/usr/bin/env node
/**
 * soak-vehicle.mjs — headless stability soak for the destructible-vehicle demo.
 *
 * Drives the demo in headless Chromium through a battery of scenarios (settle,
 * light shots, heavy shots, drop) and asserts the simulation never "explodes":
 * fractured/split chunks have overlapping convex hulls, so a bug (e.g. debris
 * colliding with the body it just detached from) shows up as debris flung at
 * absurd speeds. The probe is `window.__vehicleDemo.metrics()` (max body speed,
 * count of bodies above 60 m/s, spread). Exits non-zero if any scenario blows up.
 *
 * Usage:
 *   1) build + serve the demos:  npm start            (repo root, needs emsdk)
 *      or, if already built:     npm run serve:demos
 *   2) install the browser once: npx playwright install chromium  (in js_stress_example)
 *   3) run:                      node scripts/soak-vehicle.mjs [--url <url>] [--shots N]
 *
 * This is a dev/QA tool (needs a running server + a browser), so it is not part
 * of CI; run it before asking a human to QA the demo in a real browser.
 */
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const getArg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const URL = getArg('--url', 'http://localhost:8000/blast/js_stress_example/destructible-vehicle.html');
const SHOT_SPEED_LIMIT = 120; // m/s; anything faster than a fast projectile = explosion

const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 760 } });
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));

const metrics = () => page.evaluate(() => globalThis.__vehicleDemo?.metrics?.() || null);
const waitReady = async () => { for (let i = 0; i < 160; i++) { if (await metrics()) return true; await page.waitForTimeout(500); } return false; };
const setSlider = (id, v) => page.evaluate(([id, v]) => { const s = document.getElementById(id); if (s) { s.value = String(v); s.dispatchEvent(new Event('input', { bubbles: true })); } }, [id, v]);

const failures = [];
/** Sample metrics for `secs`, returning the worst (max) speed/dist/fastBodies seen. */
async function watch(secs) {
  let worst = { maxSpeed: 0, maxDist: 0, fastBodies: 0 };
  const steps = Math.max(1, Math.round(secs / 0.2));
  for (let i = 0; i < steps; i++) {
    const x = await metrics();
    if (x) {
      worst.maxSpeed = Math.max(worst.maxSpeed, x.maxSpeed);
      worst.maxDist = Math.max(worst.maxDist, x.maxDist);
      worst.fastBodies = Math.max(worst.fastBodies, x.fastBodies);
    }
    await page.waitForTimeout(200);
  }
  return worst;
}
function assertStable(label, worst, final) {
  const exploded = worst.fastBodies > 0 || worst.maxSpeed > SHOT_SPEED_LIMIT;
  const status = exploded ? 'EXPLODED' : 'ok';
  console.log(
    `  [${status}] ${label}: peakSpeed=${worst.maxSpeed.toFixed(1)} ` +
    `peakFastBodies=${worst.fastBodies} bodies=${final?.bodies ?? '?'} ` +
    `bondsBroken=${final?.brokenBonds ?? '?'}/${final?.bonds0 ?? '?'}`,
  );
  if (exploded) failures.push(label);
}

await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
if (!(await waitReady())) { console.error('demo did not initialize'); process.exit(2); }
console.log(`Soaking ${URL}\n`);

// 1) Settle untouched — must stay calm AND keep its bonds (a parked car must not
// shed parts under gravity; this guards the solver warm-up against regression).
{
  const w = await watch(5);
  const final = await metrics();
  assertStable('settle (no input)', w, final);
  const frac = final?.brokenFrac ?? 0;
  if (frac > 0.03) { console.log(`  [SHED] settle broke ${(frac * 100).toFixed(1)}% of bonds at rest (warm-up regression?)`); failures.push('settle integrity'); }
}

const canvas = page.locator('#demo-canvas');
const box = await canvas.boundingBox();
const shootAt = async (fx, fy) => canvas.click({ position: { x: box.width * fx, y: box.height * fy } });

// 2) Light default shots at the cargo.
for (let s = 0; s < 5; s++) { await shootAt(0.52, 0.4); await page.waitForTimeout(350); }
{ const w = await watch(3); assertStable('light shots', w, await metrics()); }

// 3) Heavy shots (ram) across the body.
await setSlider('cfg-proj-mass', 2500); await setSlider('cfg-proj-speed', 50);
for (let s = 0; s < 8; s++) { await shootAt(0.42 + 0.02 * s, 0.46); await page.waitForTimeout(300); }
{ const w = await watch(3); assertStable('heavy shots', w, await metrics()); }

// 4) Drop from height.
await page.locator('#btn-drop').click();
await page.waitForTimeout(800); await waitReady();
{ const w = await watch(7); assertStable('drop from height', w, await metrics()); }

console.log('');
if (pageErrors.length) { console.log('page errors:'); pageErrors.slice(0, 10).forEach((e) => console.log('  ' + e)); }
await browser.close();

if (failures.length) {
  console.error(`\nSOAK FAILED — explosions in: ${failures.join(', ')}`);
  process.exit(1);
}
console.log('\nSOAK PASSED — no explosions across all scenarios.');

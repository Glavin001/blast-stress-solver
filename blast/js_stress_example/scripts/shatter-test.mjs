/**
 * shatter-test.mjs — the acceptance test for non-overlapping colliders.
 *
 * Enables FULL debris collision, then cuts every bond at once so all chunks become
 * free rigid bodies. With tight, non-overlapping colliders this must simply
 * COLLAPSE AND SETTLE — peak speed stays low and nothing is flung. Overlapping
 * convex hulls instead resolve their mutual penetration as an explosion (debris
 * launched far above any physical speed).
 *
 *   node scripts/shatter-test.mjs            # PASS/FAIL + peak speed
 */
import { chromium } from 'playwright';

const URL = process.env.URL || 'http://localhost:8000/blast/js_stress_example/destructible-vehicle.html';
// A gentle collapse: chunks fall from their resting positions and settle, so the
// peak stays low and the final speed is ~0. Overlap resolution instead injects
// energy (high peak) and keeps debris bouncing (high settled speed).
const PEAK_MAX = Number(process.env.PEAK_MAX ?? 8);    // m/s peak during collapse
const SETTLE_MAX = Number(process.env.SETTLE_MAX ?? 2); // m/s once settled

const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 760 } });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
const ev = (fn, arg) => page.evaluate(fn, arg);
const metrics = () => ev(() => globalThis.__vehicleDemo?.metrics?.() || null);
const sleep = (ms) => page.waitForTimeout(ms);

await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
for (let i = 0; i < 200 && !(await metrics()); i++) await sleep(500);
if (!(await metrics())) { console.error('demo did not init'); process.exit(2); }

await ev(() => globalThis.__vehicleDemo.setDebris('all'));
await sleep(1200); // settle on the ground first
const before = await metrics();

const cut = await ev(() => globalThis.__vehicleDemo.shatterAll());

let peak = 0, peakFast = 0;
for (let i = 0; i < 80; i++) { // ~12 s — long enough to settle
  const m = await metrics();
  if (m) { peak = Math.max(peak, m.maxSpeed); peakFast = Math.max(peakFast, m.fastBodies); }
  await sleep(150);
}
const after = await metrics();
const settled = after?.maxSpeed ?? 999;

const pass = peak < PEAK_MAX && settled < SETTLE_MAX;
console.log(`\nshatter-test: ${URL}`);
console.log(`  bonds cut: ${cut}   bodies: ${before?.bodies} → ${after?.bodies}`);
console.log(`  peak speed:    ${peak.toFixed(1)} m/s  (gentle < ${PEAK_MAX})   bodies>60m/s: ${peakFast}`);
console.log(`  settled speed: ${settled.toFixed(1)} m/s  (settled < ${SETTLE_MAX})`);
console.log(`  ${pass ? 'PASS — collapsed and settled' : `FAIL — did not settle (peak ${peak.toFixed(1)}, settled ${settled.toFixed(1)})`}`);
if (errs.length) console.log('  page errors:', errs.slice(0, 5));
await browser.close();
process.exit(pass ? 0 : 1);

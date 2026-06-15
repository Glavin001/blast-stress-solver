#!/usr/bin/env node
/**
 * tune-vehicle.mjs — headless DESTRUCTION-QUALITY probe for destructible-vehicle.
 *
 * Where soak-vehicle.mjs only asserts "nothing explodes", this reports how much
 * the stress solver actually breaks per scenario, so the material/contact tuning
 * can be dialed in. All breaking is solver-computed (no scripted cuts), so the
 * numbers below are pure physics:
 *
 *   settle  → brokenFrac must be ~0 (a parked car must NOT shed under gravity)
 *   light   → a few % (cargo/accessories shed, cage holds)
 *   heavy   → large  (frame shatters under a hard ram)
 *   drop    → moderate/large (hard landing breaks things)
 *
 * Probe: window.__vehicleDemo.metrics() → { brokenBonds, bonds0, brokenFrac,
 * overstressed, maxSpeed, fastBodies, bodies }. fastBodies>0 or maxSpeed>120 = explosion.
 *
 * Usage: node scripts/tune-vehicle.mjs [--url U] [--sweep-material] [--sweep-contact]
 */
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const has = (k) => args.includes(k);
const getArg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const URL = getArg('--url', 'http://localhost:8000/blast/js_stress_example/destructible-vehicle.html');

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
const click = (sel) => page.locator(sel).click();
const warmed = async () => (await metrics())?.warmedUp === true;
// Await the actual async rebuild (avoids measuring a half-built scene), then wait
// for the solver warm-up to clear (counted in steps, so wall-clock varies with
// the headless frame rate) so shots land in the live-fracture regime.
const reset = async (h = 0) => {
  await page.evaluate((h) => globalThis.__vehicleDemo.rebuild(h), h);
  await page.waitForTimeout(200); await waitReady();
  for (let i = 0; i < 200 && !(await warmed()); i++) await page.waitForTimeout(100);
};

/** Worst (peak) speed/fastBodies seen over `secs`, plus the final snapshot. */
async function watch(secs) {
  let peakSpeed = 0, peakFast = 0, peakOver = 0;
  const steps = Math.max(1, Math.round(secs / 0.2));
  for (let i = 0; i < steps; i++) {
    const x = await metrics();
    if (x) { peakSpeed = Math.max(peakSpeed, x.maxSpeed); peakFast = Math.max(peakFast, x.fastBodies); peakOver = Math.max(peakOver, x.overstressed); }
    await page.waitForTimeout(200);
  }
  return { peakSpeed, peakFast, peakOver, final: await metrics() };
}

await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
if (!(await waitReady())) { console.error('demo did not initialize'); process.exit(2); }

// Optional overrides: MAT=<log10 exponent> CF=<contact force scale>. Set once so
// the per-scenario reset()s (which rebuild from CONFIG) pick them up.
if (process.env.MAT) { await setSlider('cfg-material', Number(process.env.MAT)); console.log(`materialScale = 1e${process.env.MAT}`); }
if (process.env.CF) { await setSlider('cfg-contact-force', Number(process.env.CF)); console.log(`contactForceScale = ${process.env.CF}`); }

const canvas = page.locator('#demo-canvas');
const box = await canvas.boundingBox();
const shootAt = async (fx, fy) => canvas.click({ position: { x: box.width * fx, y: box.height * fy } });

function row(label, w) {
  const f = w.final ?? {};
  const exploded = w.peakFast > 0 || w.peakSpeed > 120;
  const pct = ((f.brokenFrac ?? 0) * 100).toFixed(1);
  console.log(
    `  ${label.padEnd(16)} broken=${String(f.brokenBonds ?? '?').padStart(4)}/${String(f.bonds0 ?? '?').padEnd(4)} (${pct.padStart(5)}%)` +
    `  bodies=${String(f.bodies ?? '?').padStart(3)}  peakOver=${String(w.peakOver).padStart(4)}` +
    `  peakSpeed=${w.peakSpeed.toFixed(0).padStart(3)}  ${exploded ? 'EXPLODED' : ''}`,
  );
}

async function scenarioBattery() {
  // 1) Settle — parked car under gravity, no input.
  await reset();
  row('settle', await watch(5));

  // 2) Light default shots at the cargo.
  await reset();
  for (let s = 0; s < 5; s++) { await shootAt(0.52, 0.4); await page.waitForTimeout(350); }
  row('light shots', await watch(3));

  // 3) Heavy shots (ram) across the body.
  await reset();
  await setSlider('cfg-proj-mass', 2500); await setSlider('cfg-proj-speed', 55);
  for (let s = 0; s < 8; s++) { await shootAt(0.42 + 0.02 * s, 0.46); await page.waitForTimeout(300); }
  row('heavy shots', await watch(3));
  await setSlider('cfg-proj-mass', 150); await setSlider('cfg-proj-speed', 38);

  // 4) Drop from height.
  await reset(4.0);
  await page.waitForTimeout(600);
  row('drop', await watch(6));
}

console.log(`\n== Destruction battery @ default tuning ==`);
console.log(`(${URL})\n`);
await scenarioBattery();

// Optional: sweep material toughness on a heavy ram + a settle, to find the window
// where settle stays intact (~0%) but a heavy hit breaks a lot.
if (has('--sweep-material')) {
  console.log(`\n== Material-scale sweep (cfg-material = log10 exponent) ==`);
  for (const exp of [8, 8.5, 9, 9.5, 10, 10.5, 11]) {
    // Set toughness BEFORE the rebuild so the spawn itself uses it (the slider
    // updates CONFIG + live-applies; reset() rebuilds from CONFIG).
    await setSlider('cfg-material', exp);
    await reset();
    const settle = await watch(4);
    await setSlider('cfg-material', exp);
    await reset();
    await setSlider('cfg-proj-mass', 2500); await setSlider('cfg-proj-speed', 55);
    for (let s = 0; s < 6; s++) { await shootAt(0.44 + 0.02 * s, 0.46); await page.waitForTimeout(280); }
    const heavy = await watch(3);
    await setSlider('cfg-proj-mass', 150); await setSlider('cfg-proj-speed', 38);
    const sp = (settle.final?.brokenFrac ?? 0) * 100, hp = (heavy.final?.brokenFrac ?? 0) * 100;
    console.log(`  1e${exp.toString().padEnd(4)}  settle=${sp.toFixed(1).padStart(5)}%  heavy=${hp.toFixed(1).padStart(5)}%` +
      `  heavyPeakSpeed=${heavy.peakSpeed.toFixed(0)} ${heavy.peakFast > 0 ? 'EXPLODED' : ''}`);
  }
}

if (has('--sweep-contact')) {
  console.log(`\n== Contact-force-scale sweep (cfg-contact-force), light shots ==`);
  for (const cf of [15, 30, 60, 100, 150, 200]) {
    await reset();
    await setSlider('cfg-contact-force', cf);
    for (let s = 0; s < 5; s++) { await shootAt(0.52, 0.4); await page.waitForTimeout(320); }
    const w = await watch(3);
    console.log(`  cf=${String(cf).padStart(3)}  light broken=${((w.final?.brokenFrac ?? 0) * 100).toFixed(1)}%  peakSpeed=${w.peakSpeed.toFixed(0)} ${w.peakFast > 0 ? 'EXPLODED' : ''}`);
  }
}

console.log('');
if (pageErrors.length) { console.log('page errors:'); pageErrors.slice(0, 10).forEach((e) => console.log('  ' + e)); }
await browser.close();

/**
 * probe-vehicle.mjs — sweep projectile mass/speed against the destructible-vehicle
 * demo and report, per setting: parts shed after 3 shots, peak debris speed, and
 * count of bodies >60 m/s ("explosion"). Used to (a) find a sensible default
 * projectile and (b) check stability without scripted fracture caps.
 *
 *   node scripts/probe-vehicle.mjs
 */
import { chromium } from 'playwright';

const URL = 'http://localhost:8000/blast/js_stress_example/destructible-vehicle.html';
const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 760 } });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));

const metrics = () => page.evaluate(() => globalThis.__vehicleDemo?.metrics?.() || null);
const setSlider = (id, v) => page.evaluate(([id, v]) => {
  const s = document.getElementById(id); if (s) { s.value = String(v); s.dispatchEvent(new Event('input', { bubbles: true })); }
}, [id, v]);
const click = (sel) => page.locator(sel).click();
const sleep = (ms) => page.waitForTimeout(ms);

await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
for (let i = 0; i < 160 && !(await metrics()); i++) await sleep(500);
if (!(await metrics())) { console.error('demo did not init'); process.exit(2); }

const canvas = page.locator('#demo-canvas');
const box = await canvas.boundingBox();
const shoot = (fx, fy) => canvas.click({ position: { x: box.width * fx, y: box.height * fy } });

// Sample peak speed / fast-body count over `secs`.
async function watch(secs) {
  let peakSpeed = 0, peakFast = 0;
  for (let i = 0; i < Math.round(secs / 0.15); i++) {
    const m = await metrics();
    if (m) { peakSpeed = Math.max(peakSpeed, m.maxSpeed); peakFast = Math.max(peakFast, m.fastBodies); }
    await sleep(150);
  }
  return { peakSpeed, peakFast };
}

const SETTINGS = [
  { mass: 150, speed: 45 },
  { mass: 300, speed: 55 },
  { mass: 600, speed: 45 },
  { mass: 600, speed: 60 },
];

console.log(`\nprobe: ${URL}`);
console.log('mass  speed | shed(3 shots)  peakSpeed  fastBodies(>60)  settleShed');
for (const s of SETTINGS) {
  await click('#btn-reset');
  await sleep(300);
  for (let i = 0; i < 160 && !(await metrics()); i++) await sleep(250);
  const settle = await watch(2); // must stay calm at rest
  const settleShed = (await metrics())?.shed ?? '?';
  await setSlider('cfg-proj-mass', s.mass);
  await setSlider('cfg-proj-speed', s.speed);
  let peakSpeed = 0, peakFast = 0;
  for (let k = 0; k < 3; k++) { await shoot(0.5, 0.45); const w = await watch(1.2); peakSpeed = Math.max(peakSpeed, w.peakSpeed); peakFast = Math.max(peakFast, w.peakFast); }
  const m = await metrics();
  console.log(
    `${String(s.mass).padStart(4)}  ${String(s.speed).padStart(5)} | ` +
    `${String(m?.shed ?? '?').padStart(12)}  ${peakSpeed.toFixed(1).padStart(9)}  ${String(peakFast).padStart(15)}  ${String(settleShed).padStart(10)}`,
  );
}
if (errs.length) { console.log('\npage errors:', errs.slice(0, 5)); }
await browser.close();

/**
 * Demo experiment flags — lets you A/B test physics-pipeline options live without editing code.
 *
 * Flags are read from localStorage (so a reload re-applies them) and can also be set via URL query
 * (?resim=off&bodyCcd=off&resimPasses=0). A small panel is injected bottom-left of every demo to
 * toggle them and reload. Defaults match the library defaults, so leaving everything untouched
 * reproduces today's behavior exactly.
 *
 * These exist to chase the "big fracturing chunks fall slow/floaty while the projectile and small
 * debris move normally" report: resim and CCD are the two options that only affect the
 * fracturing/in-contact bodies, so toggling them is the fastest way to find the culprit.
 */
export type ExperimentFlags = {
  /** resimulateOnFracture — roll back + re-resolve the contact against fractured pieces. */
  resim: boolean;
  /** maxResimulationPasses. */
  resimPasses: number;
  /** fractureBodyCcdEnabled — continuous collision detection on fracture fragments. CCD clamps a
   *  fast body's advance to its first predicted contact, which can look like "floaty" slow motion
   *  for big chunks near the debris pile while open-air debris is unaffected. */
  bodyCcd: boolean;
  /** projectileCcdEnabled. */
  projectileCcd: boolean;
};

const STORAGE_KEY = 'blastDemoExperiments';
const DEFAULTS: ExperimentFlags = { resim: true, resimPasses: 1, bodyCcd: true, projectileCcd: true };

const isOff = (v: string | null) => v === 'off' || v === '0' || v === 'false';

export function readExperimentFlags(): ExperimentFlags {
  const flags: ExperimentFlags = { ...DEFAULTS };
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    if (typeof stored.resim === 'boolean') flags.resim = stored.resim;
    if (Number.isFinite(stored.resimPasses)) flags.resimPasses = stored.resimPasses;
    if (typeof stored.bodyCcd === 'boolean') flags.bodyCcd = stored.bodyCcd;
    if (typeof stored.projectileCcd === 'boolean') flags.projectileCcd = stored.projectileCcd;
  } catch {
    /* no localStorage (SSR/headless) — fall back to defaults */
  }
  try {
    const q = new URLSearchParams(location.search);
    if (q.has('resim')) flags.resim = !isOff(q.get('resim'));
    if (q.has('resimPasses')) flags.resimPasses = Math.max(0, parseInt(q.get('resimPasses') || '0', 10) || 0);
    if (q.has('bodyCcd')) flags.bodyCcd = !isOff(q.get('bodyCcd'));
    if (q.has('projectileCcd')) flags.projectileCcd = !isOff(q.get('projectileCcd'));
  } catch {
    /* no location */
  }
  return flags;
}

/** Spread into a `buildDestructibleCore({ ... })` options object to apply the current flags. */
export function experimentCoreOverrides(): {
  resimulateOnFracture: boolean;
  maxResimulationPasses: number;
  fractureBodyCcdEnabled: boolean;
  projectileCcdEnabled: boolean;
} {
  const f = readExperimentFlags();
  const o = {
    resimulateOnFracture: f.resim,
    maxResimulationPasses: f.resimPasses,
    fractureBodyCcdEnabled: f.bodyCcd,
    projectileCcdEnabled: f.projectileCcd,
  };
  try {
    // Visible in the console on every load so you always know what's active.
    console.info('[experiments] active flags:', o);
  } catch {}
  return o;
}

function writeFlags(f: ExperimentFlags) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(f)); } catch {}
}

/** Inject a small fixed panel (bottom-left) with the toggles + an Apply&reload button. */
export function mountExperimentPanel(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById('exp-panel')) return;
  const f = readExperimentFlags();
  const panel = document.createElement('div');
  panel.id = 'exp-panel';
  panel.style.cssText =
    'position:fixed;bottom:8px;left:8px;z-index:99999;background:rgba(10,13,19,.92);color:#cdd6e4;' +
    'font:12px/1.5 ui-monospace,monospace;padding:8px 10px;border:1px solid #2a3550;border-radius:6px;max-width:240px';
  panel.innerHTML =
    '<div style="font-weight:bold;margin-bottom:4px">Experiments <span style="opacity:.55">· reload to apply</span></div>' +
    '<label style="display:block"><input type="checkbox" id="exp-resim"> resimulateOnFracture</label>' +
    '<label style="display:block;margin-left:18px">passes <input type="number" id="exp-resimPasses" min="0" max="8" style="width:44px"></label>' +
    '<label style="display:block"><input type="checkbox" id="exp-bodyCcd"> fragment CCD</label>' +
    '<label style="display:block"><input type="checkbox" id="exp-projectileCcd"> projectile CCD</label>' +
    '<div style="margin-top:6px"><button id="exp-apply">Apply &amp; reload</button> ' +
    '<button id="exp-reset">Reset</button></div>';
  const add = () => {
    document.body.appendChild(panel);
    const $ = <T extends HTMLElement>(id: string) => panel.querySelector('#' + id) as T;
    $<HTMLInputElement>('exp-resim').checked = f.resim;
    $<HTMLInputElement>('exp-resimPasses').value = String(f.resimPasses);
    $<HTMLInputElement>('exp-bodyCcd').checked = f.bodyCcd;
    $<HTMLInputElement>('exp-projectileCcd').checked = f.projectileCcd;
    $<HTMLButtonElement>('exp-apply').addEventListener('click', () => {
      writeFlags({
        resim: $<HTMLInputElement>('exp-resim').checked,
        resimPasses: parseInt($<HTMLInputElement>('exp-resimPasses').value, 10) || 0,
        bodyCcd: $<HTMLInputElement>('exp-bodyCcd').checked,
        projectileCcd: $<HTMLInputElement>('exp-projectileCcd').checked,
      });
      location.reload();
    });
    $<HTMLButtonElement>('exp-reset').addEventListener('click', () => {
      try { localStorage.removeItem(STORAGE_KEY); } catch {}
      location.reload();
    });
  };
  if (document.body) add();
  else window.addEventListener('DOMContentLoaded', add);
}

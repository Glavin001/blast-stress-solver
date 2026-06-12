/**
 * Shared Physics / Optimization / Features controls for the modern demos' `#sidebar`.
 *
 * Sibling of `pipeline-controls.ts`: it self-injects the standard control sections (so each
 * demo no longer hand-rolls the same rows in its HTML + wiring), exposes
 * `physicsCoreOverrides()` to spread into every `buildDestructibleCore(...)`, and drives the
 * live setters through the demo's current core. Build-time options (friction, restitution,
 * custom damage) apply on the demo's existing Reset/Rebuild button.
 *
 * Defaults match the rest of the demos: full debris collisions, small-body damping off,
 * debris cleanup after the first ground hit.
 */
import type { DestructibleCore } from 'blast-stress-solver/rapier';
import { RECOMMENDED_DAMPING } from './demo-optimization-preset.js';

export type DebrisCollisionMode = 'all' | 'noDebrisPairs' | 'debrisGroundOnly' | 'debrisNone';
export type DampingMode = 'off' | 'always' | 'afterGroundCollision';
export type CleanupMode = 'off' | 'always' | 'afterGroundCollision';

export type PhysicsConfig = {
  debrisCollisionMode: DebrisCollisionMode;
  friction: number;
  restitution: number;
  smallBodyDampingMode: DampingMode;
  debrisCleanupMode: CleanupMode;
  debrisTtlMs: number;
  maxCollidersForDebris: number;
  centrifugal: boolean;
  damage: boolean;
  debug: boolean;
};

/** Shared, mutable control state (one demo page = one instance, like `pipelineFlags`). */
export const physicsConfig: PhysicsConfig = {
  debrisCollisionMode: 'all', // full collisions by default
  friction: 0.25,
  restitution: 0,
  smallBodyDampingMode: 'off',
  debrisCleanupMode: 'afterGroundCollision',
  debrisTtlMs: 10000,
  maxCollidersForDebris: 3,
  centrifugal: false,
  damage: false,
  debug: false,
};

/** Spread into a `buildDestructibleCore({ ... })` options object to apply the current state. */
export function physicsCoreOverrides() {
  return {
    friction: physicsConfig.friction,
    restitution: physicsConfig.restitution,
    debrisCollisionMode: physicsConfig.debrisCollisionMode as DebrisCollisionMode,
    smallBodyDamping: {
      mode: physicsConfig.smallBodyDampingMode as DampingMode,
      ...RECOMMENDED_DAMPING,
    },
    debrisCleanup: {
      mode: physicsConfig.debrisCleanupMode as CleanupMode,
      debrisTtlMs: physicsConfig.debrisTtlMs,
      maxCollidersForDebris: physicsConfig.maxCollidersForDebris,
    },
    damage: { enabled: physicsConfig.damage },
  };
}

export type PhysicsControlsOptions = {
  /** Returns the demo's current core (or null while rebuilding) — for the live setters. */
  getCore: () => DestructibleCore | null;
  /** Toggle the demo's own debug renderer / debug-lines (the demo owns that object). */
  onDebug?: (on: boolean) => void;
  /** Apply a build-time change (default: click the demo's `#btn-reset`). */
  onRebuild?: () => void;
  /** Drop sections/rows a particular demo doesn't want. All default to true. */
  include?: { centrifugal?: boolean; damage?: boolean; debug?: boolean };
  /** Per-demo starting overrides (e.g. a demo that wants more friction). */
  defaults?: Partial<PhysicsConfig>;
};

const LIVE = '<small style="font-weight:normal;opacity:.5">&#9733; = live</small>';

/**
 * Inject the Physics / Optimization / Features sections into the right-hand `#sidebar` and
 * wire them. No-op if the sidebar is absent or the sections already exist.
 */
export function mountPhysicsControls(opts: PhysicsControlsOptions): void {
  if (typeof document === 'undefined') return;
  Object.assign(physicsConfig, opts.defaults ?? {});
  const inc = { centrifugal: true, damage: true, debug: true, ...opts.include };
  const rebuild = opts.onRebuild ?? (() => (document.getElementById('btn-reset') as HTMLButtonElement | null)?.click());

  const mount = () => {
    const sidebar = document.getElementById('sidebar') as (HTMLElement & { __physWired?: boolean }) | null;
    if (!sidebar || sidebar.__physWired) return;
    sidebar.__physWired = true;

    // If a demo still carries its inline Physics/Optimization rows, wire those in place;
    // otherwise inject the standard sections (the DRY path for demos that drop the markup).
    if (!document.getElementById('cfg-debris-collision')) {
    const physics =
      `<section class="config-section" id="cfg-physics-section"><h2 class="section-title">Physics ${LIVE}</h2>` +
      (inc.centrifugal
        ? '<div class="config-row"><label class="config-label" for="cfg-centrifugal">Centrifugal (spin self-fracture) &#9733;</label>' +
          '<input type="checkbox" id="cfg-centrifugal" /></div>'
        : '') +
      '<div class="config-row"><span class="config-label">Debris Collision &#9733;</span>' +
      '<select id="cfg-debris-collision" class="config-select">' +
      '<option value="all">All (full collisions)</option><option value="noDebrisPairs">No debris pairs</option>' +
      '<option value="debrisGroundOnly">Debris &harr; ground only</option><option value="debrisNone">Debris: none</option></select></div>' +
      '<div class="config-row"><span class="config-label">Friction</span>' +
      '<input type="range" class="config-slider" id="cfg-friction" min="0" max="2" step="0.05" />' +
      '<span class="config-value" id="cfg-friction-value"></span></div>' +
      '<div class="config-row"><span class="config-label">Restitution</span>' +
      '<input type="range" class="config-slider" id="cfg-restitution" min="0" max="1" step="0.05" />' +
      '<span class="config-value" id="cfg-restitution-value"></span></div></section>';

    const optimization =
      `<section class="config-section" id="cfg-optimization-section"><h2 class="section-title">Optimization ${LIVE}</h2>` +
      '<div class="config-row"><span class="config-label">Small Body Damping &#9733;</span>' +
      '<select id="cfg-damping-mode" class="config-select"><option value="off">Off</option>' +
      '<option value="always">Always</option><option value="afterGroundCollision">After ground hit</option></select></div>' +
      '<div class="config-row"><span class="config-label">Debris Cleanup &#9733;</span>' +
      '<select id="cfg-cleanup-mode" class="config-select"><option value="off">Off</option>' +
      '<option value="always">Always</option><option value="afterGroundCollision">After ground hit</option></select></div>' +
      '<div class="config-row"><span class="config-label">Debris TTL &#9733;</span>' +
      '<input type="range" class="config-slider" id="cfg-debris-ttl" min="1000" max="30000" step="500" />' +
      '<span class="config-value" id="cfg-debris-ttl-value"></span></div></section>';

    const features =
      (inc.damage || inc.debug) ?
      '<section class="config-section" id="cfg-features-section"><h2 class="section-title">Features</h2>' +
      (inc.damage
        ? '<label class="toggle-row"><input type="checkbox" id="opt-damage" /><span class="toggle-text">Custom damage system' +
          '<small>Per-chunk health/splash. Off by default (it can over-soften the frame). Toggling rebuilds the scene.</small></span></label>'
        : '') +
      (inc.debug
        ? '<label class="toggle-row"><input type="checkbox" id="opt-debug" /><span class="toggle-text">Debug wireframe' +
          '<small>Show Rapier colliders &amp; bond lines</small></span></label>'
        : '') +
      '</section>' : '';

    // Drop the standard sections after the demo's own config controls but before its action
    // buttons / Status panels — matching each demo's original layout. A top-of-panel Reset bar
    // (e.g. high-rise) has no config-section before it, so it's skipped as an anchor.
    const wrap = document.createElement('div');
    wrap.innerHTML = physics + optimization + features;
    const frag = document.createDocumentFragment();
    while (wrap.firstChild) frag.appendChild(wrap.firstChild);
    let anchor: Element | null = null;
    let seenConfig = false;
    for (const el of Array.from(sidebar.children)) {
      if (el.classList.contains('config-section') && el.id !== 'recorder-slot') { seenConfig = true; continue; }
      if (el.classList.contains('status-panel')) { anchor = el; break; }
      if (el.classList.contains('control-actions') && seenConfig) { anchor = el; break; }
    }
    anchor = anchor ?? document.getElementById('recorder-slot');
    if (anchor && anchor.parentElement) anchor.parentElement.insertBefore(frag, anchor);
    else sidebar.appendChild(frag);
    }

    wireControls(opts.getCore, opts.onDebug, rebuild, inc);
  };
  if (document.body) mount();
  else window.addEventListener('DOMContentLoaded', mount);
}

function bindSlider(id: string, get: () => number, set: (v: number) => void, fmt: (v: number) => string, onInput?: (v: number) => void) {
  const s = document.getElementById(id) as HTMLInputElement | null;
  const d = document.getElementById(id + '-value');
  if (!s) return;
  s.value = String(get());
  if (d) d.textContent = fmt(get());
  s.addEventListener('input', () => { const v = parseFloat(s.value); set(v); if (d) d.textContent = fmt(v); onInput?.(v); });
}
function bindSelect(id: string, get: () => string, set: (v: string) => void, onChange?: (v: string) => void) {
  const s = document.getElementById(id) as HTMLSelectElement | null;
  if (!s) return;
  s.value = get();
  s.addEventListener('change', () => { set(s.value); onChange?.(s.value); });
}

function wireControls(
  getCore: () => DestructibleCore | null,
  onDebug: ((on: boolean) => void) | undefined,
  rebuild: () => void,
  inc: { centrifugal?: boolean; damage?: boolean; debug?: boolean },
) {
  // Physics
  if (inc.centrifugal) {
    const c = document.getElementById('cfg-centrifugal') as HTMLInputElement | null;
    if (c) {
      c.checked = physicsConfig.centrifugal;
      c.addEventListener('change', () => {
        physicsConfig.centrifugal = c.checked;
        getCore()?.setSolverCentrifugalEnabled?.(c.checked);
      });
    }
  }
  bindSelect('cfg-debris-collision', () => physicsConfig.debrisCollisionMode, (v) => { physicsConfig.debrisCollisionMode = v as DebrisCollisionMode; },
    (v) => getCore()?.setDebrisCollisionMode?.(v as DebrisCollisionMode));
  bindSlider('cfg-friction', () => physicsConfig.friction, (v) => { physicsConfig.friction = v; }, (v) => v.toFixed(2));
  bindSlider('cfg-restitution', () => physicsConfig.restitution, (v) => { physicsConfig.restitution = v; }, (v) => v.toFixed(2));

  // Optimization (all live)
  bindSelect('cfg-damping-mode', () => physicsConfig.smallBodyDampingMode, (v) => { physicsConfig.smallBodyDampingMode = v as DampingMode; },
    (v) => getCore()?.setSmallBodyDamping?.({ mode: v as DampingMode }));
  bindSelect('cfg-cleanup-mode', () => physicsConfig.debrisCleanupMode, (v) => { physicsConfig.debrisCleanupMode = v as CleanupMode; },
    (v) => getCore()?.setDebrisCleanup?.({ mode: v as CleanupMode, debrisTtlMs: physicsConfig.debrisTtlMs }));
  bindSlider('cfg-debris-ttl', () => physicsConfig.debrisTtlMs, (v) => { physicsConfig.debrisTtlMs = v; }, (v) => (v / 1000).toFixed(1) + 's',
    (v) => getCore()?.setDebrisCleanup?.({ mode: physicsConfig.debrisCleanupMode, debrisTtlMs: v }));

  // Features
  if (inc.damage) {
    const d = document.getElementById('opt-damage') as HTMLInputElement | null;
    if (d) { d.checked = physicsConfig.damage; d.addEventListener('change', () => { physicsConfig.damage = d.checked; rebuild(); }); }
  }
  if (inc.debug) {
    const dbg = document.getElementById('opt-debug') as HTMLInputElement | null;
    if (dbg) { dbg.checked = physicsConfig.debug; dbg.addEventListener('change', () => { physicsConfig.debug = dbg.checked; onDebug?.(dbg.checked); }); }
  }
}

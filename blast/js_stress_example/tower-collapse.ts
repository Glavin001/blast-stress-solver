/**
 * Tower Collapse Demo
 *
 * Showcases the high-level blast-stress-solver/rapier and blast-stress-solver/three
 * APIs with a tall tower that collapses under its own weight or projectile impacts.
 *
 * Click the viewport to launch projectiles at the tower.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import {
  buildDestructibleCore,
  FrameProfilerBuffer,
  drawFrameProfilerChart,
  FRAME_PHASES,
} from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
  applyAutoBondingToScenario,
} from 'blast-stress-solver/three';
import { buildTowerScenario } from 'blast-stress-solver/scenarios';

// ── Live frame profiler ───────────────────────────────────────
// Streams the core's per-frame profiler samples into a rolling buffer and draws
// the per-phase breakdown, so a dip below 60fps is immediately attributable to a
// phase (and, in A/B mode, compared against the old split planner's cost).
const profilerBuffer = new FrameProfilerBuffer(180);
const profilerState = { show: true, measureOld: false };

function applyProfiler(core: any) {
  core.setProfiler({
    enabled: true,
    onSample: (s: any) => profilerBuffer.push(s),
    measureReferencePlanner: profilerState.measureOld,
  });
}

// ── Config ────────────────────────────────────────────────────

const CONFIG = {
  tower: {
    side: 4,
    stories: 16,
    spacing: { x: 0.42, y: 0.42, z: 0.42 },
    totalMass: 5_000,
    areaScale: 0.05,
    addDiagonals: true,
    diagScale: 0.55,
    normalizeAreas: true,
  },
  projectile: {
    radius: 0.35,
    mass: 1_000,
    speed: 22,
  },
  solver: {
    gravity: -9.81,
    materialScale: 1e10,
  },
  physics: {
    debrisCollisionMode: 'all' as string,
    friction: 0.25,
    restitution: 0.0,
    contactForceScale: 30,
    skipSingleBodies: false,
  },
  optimization: {
    smallBodyDampingMode: 'always' as string,
    debrisCleanupMode: 'always' as string,
    debrisTtlMs: 10000,
    maxCollidersForDebris: 2,
  },
  autoBonds: false,
};

// ── Three.js setup ────────────────────────────────────────────

const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e15);
scene.fog = new THREE.FogExp2(0x0b0e15, 0.015);

const camera = new THREE.PerspectiveCamera(
  55,
  canvas.clientWidth / canvas.clientHeight,
  0.1,
  200,
);
camera.position.set(6, 5, 12);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 2.5, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.35));

const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
dirLight.position.set(10, 18, 8);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -12;
dirLight.shadow.camera.right = 12;
dirLight.shadow.camera.top = 16;
dirLight.shadow.camera.bottom = -4;
scene.add(dirLight);

// Ground
const groundGeo = new THREE.PlaneGeometry(60, 60);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x1a1e2f,
  roughness: 0.85,
  metalness: 0.1,
});
const groundMesh = new THREE.Mesh(groundGeo, groundMat);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.4;
groundMesh.receiveShadow = true;
scene.add(groundMesh);

// ── Stats panel (FPS / MS / MB) ───────────────────────────────

const stats = new Stats();
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0';
stats.dom.style.left = '0';
(document.querySelector('.viewport') as HTMLElement)?.appendChild(stats.dom);

// ── Perf tracking ─────────────────────────────────────────────

let _physicsMs = 0;
let _renderMs = 0;
const EMA = 0.12; // exponential moving-average smoothing factor

function updatePerfStats() {
  const el = (id: string) => document.getElementById(id);
  el('stat-physics-ms')!.textContent = _physicsMs.toFixed(1) + ' ms';
  el('stat-render-ms')!.textContent = _renderMs.toFixed(1) + ' ms';
  el('stat-draw-calls')!.textContent = String(renderer.info.render.calls);
  el('stat-triangles')!.textContent = renderer.info.render.triangles.toLocaleString();
}

// ── Status HUD ────────────────────────────────────────────────

function updateStatus(core: any) {
  const el = (id: string) => document.getElementById(id);
  el('stat-bodies')!.textContent = String(core.getRigidBodyCount());
  el('stat-bonds')!.textContent = String(core.getActiveBondsCount());
  el('stat-projectiles')!.textContent = String(core.projectiles.length);
  const active = core.chunks.filter((c: any) => c.active).length;
  const detached = core.chunks.filter((c: any) => c.detached).length;
  el('stat-chunks')!.textContent = `${active} / ${detached} detached`;
}

// ── Main ──────────────────────────────────────────────────────

let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let showDebug = false;

async function initScene() {
  let scenario = buildTowerScenario(CONFIG.tower);

  // Attach fragment geometries for auto-bonding support
  const sp = scenario.spacing!;
  const fragmentGeometries = scenario.nodes.map(
    () => new THREE.BoxGeometry(sp.x, sp.y, sp.z),
  );
  scenario = {
    ...scenario,
    parameters: { ...scenario.parameters, fragmentGeometries },
  };

  // Auto-bonding: replace manual grid bonds with geometry-derived bonds
  if (CONFIG.autoBonds) {
    scenario = await applyAutoBondingToScenario(scenario, { mode: 'average', maxSeparation: 0.01 });
  }

  console.log(
    `Tower: ${scenario.nodes.length} nodes, ${scenario.bonds.length} bonds` +
      (CONFIG.autoBonds ? ' (auto-bonded)' : ' (manual)'),
  );

  console.log('[tower-collapse] buildDestructibleCore config:', {
    debrisCollisionMode: CONFIG.physics.debrisCollisionMode,
    friction: CONFIG.physics.friction,
    restitution: CONFIG.physics.restitution,
    contactForceScale: CONFIG.physics.contactForceScale,
  });

  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.solver.gravity,
    materialScale: CONFIG.solver.materialScale,
    friction: CONFIG.physics.friction,
    restitution: CONFIG.physics.restitution,
    contactForceScale: CONFIG.physics.contactForceScale,
    debrisCollisionMode: CONFIG.physics.debrisCollisionMode as any,
    skipSingleBodies: CONFIG.physics.skipSingleBodies,
    damage: {
      enabled: false,
    },
    debrisCleanup: {
      mode: CONFIG.optimization.debrisCleanupMode as any,
      debrisTtlMs: CONFIG.optimization.debrisTtlMs,
      maxCollidersForDebris: CONFIG.optimization.maxCollidersForDebris,
    },
    smallBodyDamping: {
      mode: CONFIG.optimization.smallBodyDampingMode as any,
      colliderCountThreshold: 3,
      minLinearDamping: 2,
      minAngularDamping: 2,
    },
  });

  const group = new THREE.Group();
  scene.add(group);

  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    useBatchedMesh: true,
    batchedMeshOptions: { enableBVH: false, bvhMargin: 5 },
    includeDebugLines: true,
  });

  // Rapier collider wireframe overlay
  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

  coreRef = core;
  visualsRef = visuals;

  // Feed the live frame profiler from this core.
  profilerBuffer.clear();
  applyProfiler(core);
}

// ── Frame profiler HUD ────────────────────────────────────────

const profilerCanvas = document.getElementById('profiler-canvas') as HTMLCanvasElement | null;
const profilerCtx = profilerCanvas?.getContext('2d') ?? null;

function renderProfilerHUD() {
  const overlay = document.getElementById('profiler-overlay');
  if (overlay) overlay.style.display = profilerState.show ? '' : 'none';
  if (!profilerState.show || !profilerCanvas || !profilerCtx) return;

  // Size canvas to its CSS box (DPR-aware).
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const cssW = profilerCanvas.clientWidth || 360;
  const cssH = profilerCanvas.clientHeight || 110;
  if (profilerCanvas.width !== Math.round(cssW * dpr) || profilerCanvas.height !== Math.round(cssH * dpr)) {
    profilerCanvas.width = Math.round(cssW * dpr);
    profilerCanvas.height = Math.round(cssH * dpr);
  }
  profilerCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const frames = profilerBuffer.frames();
  drawFrameProfilerChart(profilerCtx, cssW, cssH, frames, {
    showProjectedOld: profilerState.measureOld,
  });

  const stats = profilerBuffer.stats();
  const latest = profilerBuffer.latest();

  // Cause callout.
  const causeEl = document.getElementById('profiler-cause');
  if (causeEl && latest) {
    const def = FRAME_PHASES.find((p) => p.key === latest.dominant);
    causeEl.innerHTML = latest.totalMs > profilerBuffer.budgetMs
      ? `spike cause: <b style="color:${def?.color}">${def?.label}</b> (${latest.totalMs.toFixed(1)} ms)`
      : `<span style="opacity:.55">within budget (${latest.totalMs.toFixed(1)} ms)</span>`;
  }

  // Legend with live per-phase ms (mean over window), sorted heaviest first.
  const legendEl = document.getElementById('profiler-legend');
  if (legendEl) {
    legendEl.innerHTML = FRAME_PHASES
      .map((p) => ({ p, ms: stats.perPhaseMean[p.key] }))
      .filter((x) => x.ms > 0.01)
      .sort((a, b) => b.ms - a.ms)
      .map(
        ({ p, ms }) =>
          `<span class="pl-item"><i style="background:${p.color}"></i>${p.label} <b>${ms.toFixed(2)}</b></span>`,
      )
      .join('');
  }

  // Stats line: sim cost + spikes + A/B headline.
  const statsEl = document.getElementById('profiler-stats');
  if (statsEl) {
    let s =
      `sim <b>${stats.meanMs.toFixed(2)} ms</b> avg · p95 ${stats.p95Ms.toFixed(1)} · max ${stats.maxMs.toFixed(1)} · ` +
      `${stats.spikeCount} spike${stats.spikeCount === 1 ? '' : 's'}>16.7ms`;
    if (latest) s += ` · resim ${latest.resimPasses} · bodies ${latest.rigidBodies}`;
    if (profilerState.measureOld) {
      // Worst projected-old frame over the window — the cost we removed.
      let worstOld = 0;
      let worstNew = 0;
      for (const f of frames) {
        if (f.projectedOldTotalMs && f.projectedOldTotalMs > worstOld) {
          worstOld = f.projectedOldTotalMs;
          worstNew = f.totalMs;
        }
      }
      if (worstOld > 0) {
        s += ` · <span style="color:#ffb454">OLD planner peak ≈ ${worstOld.toFixed(1)} ms (now ${worstNew.toFixed(1)} ms)</span>`;
      }
    }
    statsEl.innerHTML = s;
  }
}

// ── Projectile shooting ───────────────────────────────────────

function shootProjectile(ndcX: number, ndcY: number) {
  const core = coreRef;
  if (!core) return;

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const dir = raycaster.ray.direction.clone().normalize();

  core.enqueueProjectile({
    position: {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    },
    velocity: {
      x: dir.x * CONFIG.projectile.speed,
      y: dir.y * CONFIG.projectile.speed,
      z: dir.z * CONFIG.projectile.speed,
    },
    radius: CONFIG.projectile.radius,
    mass: CONFIG.projectile.mass,
    ttl: 8000,
  });
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  shootProjectile(ndcX, ndcY);
});

// ── UI wiring ─────────────────────────────────────────────────

document.getElementById('btn-reset')?.addEventListener('click', async () => {
  visualsRef?.dispose();
  coreRef?.dispose();
  coreRef = null;
  visualsRef = null;
  await initScene();
});

document.getElementById('btn-debug')?.addEventListener('click', () => {
  showDebug = !showDebug;
  rapierDebug?.setEnabled(showDebug);
  const btn = document.getElementById('btn-debug')!;
  btn.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
});

// Config sliders
function bindSlider(id: string, obj: Record<string, any>, key: string, fmt?: (v: number) => string) {
  const slider = document.getElementById(id) as HTMLInputElement | null;
  const display = document.getElementById(id + '-value');
  if (!slider) return;
  slider.value = String(obj[key]);
  if (display) display.textContent = fmt ? fmt(obj[key]) : String(obj[key]);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    obj[key] = v;
    if (display) display.textContent = fmt ? fmt(v) : String(v);
  });
}

function bindSelect(id: string, obj: Record<string, any>, key: string, onChange?: (v: string) => void) {
  const select = document.getElementById(id) as HTMLSelectElement | null;
  if (!select) return;
  select.value = String(obj[key]);
  select.addEventListener('change', () => {
    obj[key] = select.value;
    onChange?.(select.value);
  });
}

function bindCheckbox(id: string, obj: Record<string, any>, key: string, onChange?: (v: boolean) => void) {
  const checkbox = document.getElementById(id) as HTMLInputElement | null;
  if (!checkbox) return;
  checkbox.checked = !!obj[key];
  checkbox.addEventListener('change', () => {
    obj[key] = checkbox.checked;
    onChange?.(checkbox.checked);
  });
}

bindSlider('cfg-side', CONFIG.tower, 'side');
bindSlider('cfg-stories', CONFIG.tower, 'stories');
bindSlider('cfg-area-scale', CONFIG.tower, 'areaScale', (v) => v.toFixed(3));
bindSlider('cfg-total-mass', CONFIG.tower, 'totalMass', (v) => v.toLocaleString());
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2));
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString());
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0));
bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1));
// Material scale uses a log slider: slider value is the exponent (log10)
{
  const slider = document.getElementById('cfg-material') as HTMLInputElement | null;
  const display = document.getElementById('cfg-material-value');
  if (slider) {
    const exp = Math.log10(CONFIG.solver.materialScale);
    slider.value = String(exp);
    if (display) display.textContent = `1e${exp.toFixed(0)}`;
    slider.addEventListener('input', () => {
      const exp = parseFloat(slider.value);
      CONFIG.solver.materialScale = Math.pow(10, exp);
      if (display) display.textContent = `1e${exp.toFixed(1)}`;
    });
  }
}

// Auto-bonds toggle
{
  const checkbox = document.getElementById('cfg-auto-bonds') as HTMLInputElement | null;
  if (checkbox) {
    checkbox.checked = CONFIG.autoBonds;
    checkbox.addEventListener('change', () => {
      CONFIG.autoBonds = checkbox.checked;
    });
  }
}

// Physics controls
bindSelect('cfg-debris-collision', CONFIG.physics, 'debrisCollisionMode', (v) => {
  coreRef?.setDebrisCollisionMode(v as any);
});
bindSlider('cfg-friction', CONFIG.physics, 'friction', (v) => v.toFixed(2));
bindSlider('cfg-restitution', CONFIG.physics, 'restitution', (v) => v.toFixed(2));
bindSlider('cfg-contact-force', CONFIG.physics, 'contactForceScale', (v) => v.toFixed(0));
bindCheckbox('cfg-skip-single', CONFIG.physics, 'skipSingleBodies');

// Optimization controls
bindSelect('cfg-damping-mode', CONFIG.optimization, 'smallBodyDampingMode', (v) => {
  coreRef?.setSmallBodyDamping({ mode: v as any });
});
bindSelect('cfg-cleanup-mode', CONFIG.optimization, 'debrisCleanupMode', (v) => {
  coreRef?.setDebrisCleanup({ mode: v as any, debrisTtlMs: CONFIG.optimization.debrisTtlMs });
});
bindSlider('cfg-debris-ttl', CONFIG.optimization, 'debrisTtlMs', (v) => (v / 1000).toFixed(1) + 's');
bindSlider('cfg-max-debris-colliders', CONFIG.optimization, 'maxCollidersForDebris', (v) => v.toFixed(0));

// Frame profiler controls
{
  const showCb = document.getElementById('cfg-profiler-show') as HTMLInputElement | null;
  if (showCb) {
    showCb.checked = profilerState.show;
    showCb.addEventListener('change', () => { profilerState.show = showCb.checked; });
  }
  const oldCb = document.getElementById('cfg-profiler-old') as HTMLInputElement | null;
  if (oldCb) {
    oldCb.checked = profilerState.measureOld;
    oldCb.addEventListener('change', () => {
      profilerState.measureOld = oldCb.checked;
      if (coreRef) applyProfiler(coreRef); // re-apply A/B flag to the live core
    });
  }
}

// ── Render loop ───────────────────────────────────────────────

const clock = new THREE.Clock();

function loop() {
  requestAnimationFrame(loop);
  stats.begin();

  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    const t0 = performance.now();
    coreRef.step(dt);
    _physicsMs += ((performance.now() - t0) - _physicsMs) * EMA;

    visualsRef.update({
      debug: showDebug,
      updateBVH: false,
      updateProjectiles: true,
    });
    rapierDebug?.update();
    updateStatus(coreRef);
    renderProfilerHUD();
  }

  const t1 = performance.now();
  renderer.render(scene, camera);
  _renderMs += ((performance.now() - t1) - _renderMs) * EMA;

  updatePerfStats();
  stats.end();
}

// ── Resize ────────────────────────────────────────────────────

function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

// ── Boot ──────────────────────────────────────────────────────

initScene().then(() => loop());

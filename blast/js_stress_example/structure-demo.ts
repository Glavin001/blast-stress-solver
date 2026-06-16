/**
 * Reusable destructible-structure demo harness.
 *
 * Each vibe-city `/destructible-stress` preset (frame tower, concrete hut,
 * courtyard bungalow) gets its own thin page that just calls
 * {@link mountStructureDemo} with a scenario builder + label. All of the scene,
 * physics, shooter, profiler, recorder and UI wiring lives here so the per-
 * structure entrypoints stay a few lines long.
 *
 * The pages share one HTML skeleton (`styles/demo-common.css` + the standard
 * `cfg-*` / `stat-*` element ids), so this harness drives every page identically.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay } from 'blast-stress-solver/rapier';
import type { ScenarioDesc } from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
  applyAutoBondingToScenario,
} from 'blast-stress-solver/three';
import { pipelineCoreOverrides, mountPipelineControls } from './pipeline-controls.js';
import { mountPhysicsControls, physicsCoreOverrides, physicsConfig } from './physics-controls.js';
import { mountShooter } from './shooter-fps.js';

export type StructureDemoOptions = {
  /** Builds the scenario for this page (e.g. `() => buildFrameTowerScenario()`). */
  build: () => ScenarioDesc;
  /** Human-readable label, used for the slug in profiler/recorder exports + logs. */
  label: string;
  /** Optional projectile defaults (radius m / mass kg / speed m/s). */
  projectile?: { radius?: number; mass?: number; speed?: number };
};

function slugify(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

/**
 * Boot a single-structure destructible demo into the current page. Call once per
 * page (after the DOM is ready). Returns nothing — the harness owns its own loop.
 */
export function mountStructureDemo(options: StructureDemoOptions): void {
  const slug = slugify(options.label);

  // ── Config ────────────────────────────────────────────────────
  const CONFIG = {
    projectile: {
      radius: options.projectile?.radius ?? 0.4,
      mass: options.projectile?.mass ?? 4_000,
      speed: options.projectile?.speed ?? 55,
    },
    solver: {
      gravity: -9.81,
      materialScale: 1e10,
    },
    physics: {
      debrisCollisionMode: 'all' as string,
      friction: 0.4,
      restitution: 0.0,
      contactForceScale: 30,
      skipSingleBodies: false,
    },
    autoBonds: false,
  };

  // ── Live frame profiler + session recorder ─────────────────────
  const profiler = createFrameProfilerOverlay({
    exportName: `${slug}-profile`,
    getMeta: () => {
      let bodies: number | undefined;
      let bonds: number | undefined;
      try { bodies = coreRef?.getRigidBodyCount(); } catch { /* ignore */ }
      try { bonds = coreRef?.getActiveBondsCount(); } catch { /* ignore */ }
      return {
        demo: slug,
        structure: options.label,
        config: { solver: { ...CONFIG.solver }, physics: { ...CONFIG.physics } },
        live: { bodies, bonds },
      };
    },
  });

  const recorder = createRecordingOverlay({
    mount: document.getElementById('recorder-slot') ?? undefined,
    exportName: `${slug}-recording`,
    getProfilerExport: () => profiler.exportData(),
  });

  // ── Three.js setup ────────────────────────────────────────────
  const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e15);
  scene.fog = new THREE.FogExp2(0x0b0e15, 0.01);

  const camera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
  camera.position.set(14, 12, 22);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 4, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));

  const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
  dirLight.position.set(18, 30, 14);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.left = -24;
  dirLight.shadow.camera.right = 24;
  dirLight.shadow.camera.top = 34;
  dirLight.shadow.camera.bottom = -6;
  dirLight.shadow.camera.far = 120;
  scene.add(dirLight);

  const groundGeo = new THREE.PlaneGeometry(160, 160);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x1a1e2f, roughness: 0.85, metalness: 0.1 });
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

  // ── Perf + status HUD ─────────────────────────────────────────
  let _physicsMs = 0;
  let _renderMs = 0;
  const EMA = 0.12;

  const el = (id: string) => document.getElementById(id);

  function updatePerfStats() {
    el('stat-physics-ms')!.textContent = _physicsMs.toFixed(1) + ' ms';
    el('stat-render-ms')!.textContent = _renderMs.toFixed(1) + ' ms';
    el('stat-draw-calls')!.textContent = String(renderer.info.render.calls);
    el('stat-triangles')!.textContent = renderer.info.render.triangles.toLocaleString();
  }

  function updateStatus(core: any) {
    el('stat-bodies')!.textContent = String(core.getRigidBodyCount());
    el('stat-bonds')!.textContent = String(core.getActiveBondsCount());
    el('stat-projectiles')!.textContent = String(core.projectiles.length);
    const active = core.chunks.filter((c: any) => c.active).length;
    const detached = core.chunks.filter((c: any) => c.detached).length;
    el('stat-chunks')!.textContent = `${active} / ${detached} detached`;
  }

  // ── Camera framing from scenario bounds ───────────────────────
  function frameScenario(scenario: ScenarioDesc) {
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const n of scenario.nodes) box.expandByPoint(v.set(n.centroid.x, n.centroid.y, n.centroid.z));
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5;
    controls.target.copy(center);
    const dist = Math.max(8, radius * 2.6);
    camera.position.set(center.x + dist * 0.7, center.y + radius * 0.9 + 2, center.z + dist * 0.9);
    camera.near = 0.1;
    camera.far = Math.max(200, dist * 6);
    camera.updateProjectionMatrix();
    controls.update();
  }

  // ── Main scene build ──────────────────────────────────────────
  let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
  let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
  let rapierDebug: RapierDebugRenderer | null = null;
  let showDebug = false;
  let shooter: ReturnType<typeof mountShooter> | null = null;

  async function initScene() {
    let scenario = options.build();

    // Attach per-node box geometries (from the uniform cell) so the renderer and
    // the optional WASM auto-bonder both have explicit fragment geometry.
    const sp = scenario.spacing!;
    const fragmentGeometries = scenario.nodes.map(() => new THREE.BoxGeometry(sp.x, sp.y, sp.z));
    scenario = { ...scenario, parameters: { ...scenario.parameters, fragmentGeometries } };

    if (CONFIG.autoBonds) {
      scenario = await applyAutoBondingToScenario(scenario, { mode: 'average', maxSeparation: 0.01 });
    }

    console.log(
      `[${slug}] ${options.label}: ${scenario.nodes.length} nodes, ${scenario.bonds.length} bonds` +
        (CONFIG.autoBonds ? ' (auto-bonded)' : ' (manual)'),
    );

    const core = await buildDestructibleCore({
      scenario,
      gravity: CONFIG.solver.gravity,
      materialScale: CONFIG.solver.materialScale,
      contactForceScale: CONFIG.physics.contactForceScale,
      skipSingleBodies: CONFIG.physics.skipSingleBodies,
      ...physicsCoreOverrides(),
      ...pipelineCoreOverrides(),
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

    rapierDebug?.dispose();
    rapierDebug = new RapierDebugRenderer(scene, core.world as any, { enabled: showDebug });

    coreRef = core;
    core.setSolverCentrifugalEnabled(physicsConfig.centrifugal);
    visualsRef = visuals;

    frameScenario(scenario);

    recorder.attach(core, { scenario, meta: { demo: slug, structure: options.label, config: CONFIG } });
    profiler.attach(core);
  }

  async function rebuild() {
    visualsRef?.dispose();
    coreRef?.dispose();
    coreRef = null;
    visualsRef = null;
    await initScene();
  }

  // ── UI wiring ─────────────────────────────────────────────────
  el('btn-reset')?.addEventListener('click', () => { void rebuild(); });

  el('btn-debug')?.addEventListener('click', () => {
    showDebug = !showDebug;
    rapierDebug?.setEnabled(showDebug);
    const btn = el('btn-debug')!;
    btn.textContent = showDebug ? '◈ Hide Debug' : '◇ Show Debug';
  });

  function bindSlider(id: string, obj: Record<string, any>, key: string, fmt?: (v: number) => string) {
    const slider = el(id) as HTMLInputElement | null;
    const display = el(id + '-value');
    if (!slider) return;
    slider.value = String(obj[key]);
    if (display) display.textContent = fmt ? fmt(obj[key]) : String(obj[key]);
    slider.addEventListener('input', () => {
      const value = parseFloat(slider.value);
      obj[key] = value;
      if (display) display.textContent = fmt ? fmt(value) : String(value);
    });
  }

  function bindCheckbox(id: string, obj: Record<string, any>, key: string) {
    const checkbox = el(id) as HTMLInputElement | null;
    if (!checkbox) return;
    checkbox.checked = !!obj[key];
    checkbox.addEventListener('change', () => { obj[key] = checkbox.checked; });
  }

  bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2));
  bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString());
  bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0));
  bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1));

  // Material scale uses a log slider: slider value is the exponent (log10).
  {
    const slider = el('cfg-material') as HTMLInputElement | null;
    const display = el('cfg-material-value');
    if (slider) {
      const exp = Math.log10(CONFIG.solver.materialScale);
      slider.value = String(exp);
      if (display) display.textContent = `1e${exp.toFixed(0)}`;
      slider.addEventListener('input', () => {
        const e = parseFloat(slider.value);
        CONFIG.solver.materialScale = Math.pow(10, e);
        if (display) display.textContent = `1e${e.toFixed(1)}`;
      });
    }
  }

  bindCheckbox('cfg-auto-bonds', CONFIG, 'autoBonds');

  // Shared Physics / Optimization controls + demo-specific contact knobs.
  mountPhysicsControls({ getCore: () => coreRef, include: { debug: false } });
  bindSlider('cfg-contact-force', CONFIG.physics, 'contactForceScale', (v) => v.toFixed(0));
  bindCheckbox('cfg-skip-single', CONFIG.physics, 'skipSingleBodies');
  bindSlider('cfg-max-debris-colliders', physicsConfig, 'maxCollidersForDebris', (v) => v.toFixed(0));

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

      visualsRef.update({ debug: showDebug, updateBVH: false, updateProjectiles: true });
      rapierDebug?.update();
      updateStatus(coreRef);
      profiler.render();
      recorder.render();
    }

    shooter?.update();

    const t1 = performance.now();
    renderer.render(scene, camera);
    _renderMs += ((performance.now() - t1) - _renderMs) * EMA;

    updatePerfStats();
    stats.end();
  }

  function onResize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', onResize);

  // ── Boot ──────────────────────────────────────────────────────
  mountPipelineControls();
  shooter = mountShooter({
    canvas,
    camera,
    controls,
    scene,
    getCore: () => coreRef,
    getBallParams: () => CONFIG.projectile,
  });
  initScene().then(() => loop());
}

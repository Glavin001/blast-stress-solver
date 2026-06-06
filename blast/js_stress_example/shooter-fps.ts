/**
 * First-person shooter + dual-mode weapon controls for the modern demos.
 *
 * A single self-mounting helper, shared by every modern demo (wall, tower,
 * fractured-*, high-rise, mini-city, cracking, fracture-policy …) the same way
 * `pipeline-controls.ts` is. It adds:
 *
 *   • A "First-Person Shooter" section in the right-hand `#sidebar` with the
 *     FPS toggle (the required right-panel switch), a shooting-mode selector, a
 *     Detonate button and blast-tuning sliders.
 *   • A first-person camera controller: WASD to walk, arrow keys + mouse to
 *     look, Space to jump. Toggling it off restores the demo's OrbitControls.
 *   • Two shooting modes, usable in OR out of first-person mode:
 *        1. Ball Projectile        — the demo's existing click-to-shoot ball.
 *        2. Sticky Explosive Launcher — fires a charge that sticks where it
 *           first touches (walls, pillars, debris). Place as many as you like,
 *           then press the detonate key to blow them all at once. Each blast
 *           applies an outward radial force in a radius, seeding stress into the
 *           bond graph so structures actually come apart (same mechanism the
 *           mini-city "Detonate" button uses).
 *
 * Key bindings (shown in the panel + the on-screen badge):
 *   WASD walk · arrows/mouse look · Space jump · left-click shoot
 *   1 Ball · 2 Sticky · Q cycle mode · F detonate · V toggle first-person
 *
 * The module is intentionally framework-light: it talks to the demo only
 * through a small options object and reads the live `DestructibleCore` through a
 * getter, so a Reset/Rebuild that swaps the core (and its Rapier world) is
 * detected automatically and the placed charges are cleared.
 */
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { DestructibleCore } from 'blast-stress-solver/rapier';
import { createVehicle, type VehicleHandle } from './vehicle-drive.js';

export type BallParams = { radius: number; mass: number; speed: number };

export type ShooterOptions = {
  canvas: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  scene: THREE.Scene;
  /** Returns the demo's current core (or null while rebuilding). */
  getCore: () => DestructibleCore | null;
  /** Live ball-projectile params (usually the demo's `CONFIG.projectile`). */
  getBallParams?: () => BallParams;
  /**
   * Optional override for the ball-projectile shot, given the aim point in
   * normalized device coords (0,0 = screen centre when pointer-locked). Demos
   * with bespoke ball behaviour (e.g. high-rise's building-box standoff) pass
   * their existing shoot function here; otherwise the default camera-origin
   * ball (using `getBallParams`) is used.
   */
  shootBall?: (ndcX: number, ndcY: number) => void;
  /** Y of the walkable floor; the eye rests at `floorY + eyeHeight`. Default 0. */
  floorY?: number;
  /** Eye height above the floor, in metres. Default 1.7. */
  eyeHeight?: number;
};

export type ShooterHandle = {
  /** Call once per frame, after `core.step(...)`, before `renderer.render(...)`. */
  update: () => void;
  setFpsEnabled: (on: boolean) => void;
  isFpsEnabled: () => boolean;
  setDriveEnabled: (on: boolean) => void;
  isDriveEnabled: () => boolean;
  setMode: (mode: ShootMode) => void;
  detonateAll: () => void;
};

export type ShootMode = 'ball' | 'sticky' | 'grenade';

type Vec3 = { x: number; y: number; z: number };

type StickyExplosive = {
  mesh: THREE.Mesh;
  /** Rapier body handle while the charge is still in flight; null once stuck. */
  bodyHandle: number | null;
  colliderHandle: number | null;
  flying: boolean;
  stuck: boolean;
  exploded: boolean;
  /** Grenade behaviour: explode the instant it touches something, no detonate step. */
  explodeOnContact: boolean;
  spawnTime: number;
  radius: number;
  /** Body the charge stuck to (so it follows moving pillars). null = world-pinned. */
  attachBodyHandle: number | null;
  relPos: THREE.Vector3;
  relQuat: THREE.Quaternion;
  /** Last known world position — the blast centre. */
  worldPos: THREE.Vector3;
};

type Transient = {
  mesh: THREE.Mesh;
  light: THREE.PointLight | null;
  born: number;
  ttl: number;
  radius: number;
};

const DEFAULT_BALL: BallParams = { radius: 0.35, mass: 1000, speed: 25 };

// Peak solver force (Newtons) per unit of blast "strength", injected into the
// stress solver so blasts overstress bonds and break the intact structure — the
// same load path projectile impacts use. Tuned to the ~1e10 material scale the
// destruction demos use; the Blast force slider scales it up/down.
const BLAST_SOLVER_FORCE = 8e5;

// A chunk caught in a blast, with its outward direction + falloff frozen at
// detonation time (so a piece flies the right way whenever it finally breaks free).
type BlastCandidate = { node: number; falloff: number; nx: number; ny: number; nz: number };

type BlastField = {
  cx: number; cy: number; cz: number;
  up: number;
  solverForce: number; kick: number;
  age: number; life: number; solverFrames: number;
  candidates: BlastCandidate[];
  /** Body handles already kicked, so each freed piece gets exactly one impulse. */
  kicked: Set<number>;
};

export function mountShooter(opts: ShooterOptions): ShooterHandle {
  const { canvas, camera, controls, scene, getCore } = opts;
  const getBallParams = opts.getBallParams ?? (() => DEFAULT_BALL);
  const floorY = opts.floorY ?? 0;

  // ── Tunable state (mutated by the sidebar) ──────────────────────
  const cfg = {
    fps: false,
    drive: false, // vehicle driving mode (mutually exclusive with fps)
    mode: 'ball' as ShootMode,
    headlamp: true, // camera-mounted light so the view is always lit
    walkSpeed: 9, // m/s
    jetUpSpeed: 10, // m/s — jetpack rise while Space is held (hold to fly up)
    fallGravity: 16, // m/s² downward acceleration when not thrusting
    maxFall: 35, // m/s terminal fall speed
    keyLookSpeed: 1.9, // rad/s for arrow-key look
    mouseSensitivity: 0.0022, // rad / pixel
    sticky: { radius: 0.28, mass: 60, speed: 44, ttl: 30 },
    blast: { radius: 7, strength: 55, up: 0.4 },
    // Third-person chase camera (drive mode).
    chase: { dist: 6.5, height: 2.8, look: 0.9, lerp: 6 },
    // Live-tunable car properties (sidebar sliders; pushed via vehicle.setTuning).
    car: { maxSpeed: 45, engineForce: 12000, maxSteer: 0.6 },
  };

  // ── Runtime state ───────────────────────────────────────────────
  let lastCore: DestructibleCore | null = null;
  const stickies: StickyExplosive[] = [];
  const transient: Transient[] = [];
  const blastFields: BlastField[] = [];
  const keys = new Set<string>();
  let yaw = 0;
  let pitch = 0;
  let velY = 0;
  let grounded = true;
  let pointerLocked = false;
  let lastT = performance.now();

  // ── Rapier raycast vehicle (the drive-mode car) ─────────────────
  // Spawned in the core's world when drive mode turns on; recreated after a
  // Reset (new world), torn down when drive mode turns off. See vehicle-drive.ts.
  let vehicle: VehicleHandle | null = null;
  let vehicleCore: DestructibleCore | null = null; // world the car lives in
  let camOrbitYaw = 0; // mouse-look offset around the car (decays back to centre)

  // ── Rapier kinematic character controller (the FPS body) ────────
  // A capsule that physically collides with floors, debris and structures, so
  // you can walk on rubble and clamber on top of things. Created in the core's
  // world when first-person mode turns on; recreated after a Reset (new world).
  const CAPSULE = { halfHeight: 0.6, radius: 0.35 };
  const EYE_ABOVE_CENTER = CAPSULE.halfHeight + CAPSULE.radius + 0.05; // eye ≈ capsule top
  let charBody: RAPIER.RigidBody | null = null;
  let charCollider: RAPIER.Collider | null = null;
  let charController: RAPIER.KinematicCharacterController | null = null;
  let charCore: DestructibleCore | null = null; // world the character lives in
  // The character controller ignores projectiles and thrown charges, so they
  // never block you or get shoved by you — they pass straight through.
  const charQueryFilter = (col: RAPIER.Collider): boolean => {
    const ud = col.parent()?.userData as { projectile?: boolean; stickyExplosive?: boolean } | undefined;
    return !(ud && (ud.projectile || ud.stickyExplosive));
  };

  // Scratch objects (avoid per-frame allocation on the hot path).
  const _ray = new THREE.Raycaster();
  const _ndc = new THREE.Vector2();
  const _q = new THREE.Quaternion();
  const _qinv = new THREE.Quaternion();
  const _e = new THREE.Euler(0, 0, 0, 'YXZ');
  const _fwd = new THREE.Vector3();
  const _right = new THREE.Vector3();
  const _move = new THREE.Vector3();
  const _v = new THREE.Vector3();
  const _v2 = new THREE.Vector3();
  const _look = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);

  // ── Camera headlamp ─────────────────────────────────────────────
  // A spotlight that tracks the camera each frame (position + aim), so wherever
  // you stand and look is lit — independent of the demo's fixed scene lights and
  // of the core (survives Reset). decay=0 keeps far structures lit too.
  const headlamp = new THREE.SpotLight(0xfff2e0, 2.6, 0, Math.PI / 4, 0.4, 0);
  headlamp.castShadow = false;
  headlamp.visible = cfg.headlamp;
  scene.add(headlamp);
  scene.add(headlamp.target);

  // ── Overlay UI (crosshair + status badge) ───────────────────────
  const viewport = (canvas.parentElement as HTMLElement | null) ?? document.body;
  const crosshair = document.createElement('div');
  crosshair.className = 'shooter-crosshair';
  crosshair.style.display = 'none';
  viewport.appendChild(crosshair);

  const badge = document.createElement('div');
  badge.className = 'shooter-badge';
  viewport.appendChild(badge);

  // ── Sidebar controls (injected like pipeline-controls) ──────────
  let fpsCheckbox: HTMLInputElement | null = null;
  let driveCheckbox: HTMLInputElement | null = null;
  let headlampCheckbox: HTMLInputElement | null = null;
  let modeSelect: HTMLSelectElement | null = null;
  let armedCountEl: HTMLElement | null = null;

  function mountSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar || document.getElementById('cfg-shooter-section')) return;

    const section = document.createElement('section');
    section.className = 'config-section';
    section.id = 'cfg-shooter-section';
    section.innerHTML =
      '<h2 class="section-title">🎮 First-Person Shooter</h2>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-fps" />' +
      '<span class="toggle-text">First-person mode' +
      '<small>WASD walk · arrows / mouse look · hold Space = jetpack (fly up) · ' +
      'click shoots. Walk on debris &amp; floors. Press <b>V</b> to toggle.</small></span></label>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-drive" />' +
      '<span class="toggle-text">Drive a vehicle' +
      '<small>Hop in a car: WASD / arrows drive &amp; steer · Space handbrake · ' +
      '<b>R</b> flips it upright · mouse looks around. Ram the buildings! Press <b>C</b> to toggle.</small></span></label>' +
      '<div class="config-row"><label class="config-label" for="cfg-car-speed">Car top speed</label>' +
      '<input type="range" id="cfg-car-speed" class="config-slider" min="10" max="90" step="5" value="45" />' +
      '<span class="config-value" id="cfg-car-speed-value">162 km/h</span></div>' +
      '<div class="config-row"><label class="config-label" for="cfg-car-power">Car engine power</label>' +
      '<input type="range" id="cfg-car-power" class="config-slider" min="2000" max="24000" step="1000" value="12000" />' +
      '<span class="config-value" id="cfg-car-power-value">12000 N</span></div>' +
      '<div class="config-row"><label class="config-label" for="cfg-car-steer">Car steering</label>' +
      '<input type="range" id="cfg-car-steer" class="config-slider" min="0.3" max="1" step="0.05" value="0.6" />' +
      '<span class="config-value" id="cfg-car-steer-value">34°</span></div>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-headlamp" checked />' +
      '<span class="toggle-text">Headlamp' +
      '<small>Camera-mounted light — always lights wherever you are and look.</small></span></label>' +
      '<div class="config-row"><label class="config-label" for="cfg-shoot-mode">Weapon</label>' +
      '<select id="cfg-shoot-mode" class="config-select">' +
      '<option value="ball">① Ball projectile</option>' +
      '<option value="sticky">② Sticky explosive</option>' +
      '<option value="grenade">③ Grenade (explode on contact)</option>' +
      '</select></div>' +
      '<div class="config-row"><span class="config-label">Armed charges</span>' +
      '<span class="config-value" id="cfg-armed">0</span></div>' +
      '<div class="config-row"><label class="config-label" for="cfg-blast-radius">Blast radius</label>' +
      '<input type="range" id="cfg-blast-radius" class="config-slider" min="2" max="20" step="0.5" value="7" />' +
      '<span class="config-value" id="cfg-blast-radius-value">7 m</span></div>' +
      '<div class="config-row"><label class="config-label" for="cfg-blast-strength">Blast force</label>' +
      '<input type="range" id="cfg-blast-strength" class="config-slider" min="5" max="150" step="5" value="55" />' +
      '<span class="config-value" id="cfg-blast-strength-value">55×</span></div>' +
      '<div class="config-row"><label class="config-label" for="cfg-sticky-speed">Charge speed</label>' +
      '<input type="range" id="cfg-sticky-speed" class="config-slider" min="15" max="90" step="1" value="44" />' +
      '<span class="config-value" id="cfg-sticky-speed-value">44 m/s</span></div>' +
      '<div class="control-actions" style="padding:0.5rem 0 0;border:none">' +
      '<button id="cfg-detonate" class="button button-primary">💥 Detonate all (F)</button></div>';

    // Place the shooter panel right under the header so the toggle is prominent.
    const header = sidebar.querySelector('header');
    if (header && header.parentElement) header.parentElement.insertBefore(section, header.nextSibling);
    else sidebar.insertBefore(section, sidebar.firstChild);

    fpsCheckbox = section.querySelector('#cfg-fps');
    driveCheckbox = section.querySelector('#cfg-drive');
    headlampCheckbox = section.querySelector('#cfg-headlamp');
    modeSelect = section.querySelector('#cfg-shoot-mode');
    armedCountEl = section.querySelector('#cfg-armed');

    fpsCheckbox!.checked = cfg.fps;
    fpsCheckbox!.addEventListener('change', () => setFpsEnabled(fpsCheckbox!.checked));

    driveCheckbox!.checked = cfg.drive;
    driveCheckbox!.addEventListener('change', () => setDriveEnabled(driveCheckbox!.checked));

    headlampCheckbox!.checked = cfg.headlamp;
    headlampCheckbox!.addEventListener('change', () => {
      cfg.headlamp = headlampCheckbox!.checked;
      headlamp.visible = cfg.headlamp;
    });

    modeSelect!.value = cfg.mode;
    modeSelect!.addEventListener('change', () => setMode(modeSelect!.value as ShootMode));

    bindSlider(section, '#cfg-blast-radius', (v) => { cfg.blast.radius = v; }, (v) => v.toFixed(1) + ' m');
    bindSlider(section, '#cfg-blast-strength', (v) => { cfg.blast.strength = v; }, (v) => v.toFixed(0) + '×');
    bindSlider(section, '#cfg-sticky-speed', (v) => { cfg.sticky.speed = v; }, (v) => v.toFixed(0) + ' m/s');
    // Car tuning — pushed straight into the live vehicle (no respawn needed).
    bindSlider(section, '#cfg-car-speed', (v) => { cfg.car.maxSpeed = v; vehicle?.setTuning({ maxSpeed: v }); }, (v) => (v * 3.6).toFixed(0) + ' km/h');
    bindSlider(section, '#cfg-car-power', (v) => { cfg.car.engineForce = v; vehicle?.setTuning({ engineForce: v }); }, (v) => v.toFixed(0) + ' N');
    bindSlider(section, '#cfg-car-steer', (v) => { cfg.car.maxSteer = v; vehicle?.setTuning({ maxSteer: v }); }, (v) => ((v * 180) / Math.PI).toFixed(0) + '°');

    section.querySelector('#cfg-detonate')!.addEventListener('click', () => detonateAll());
    refreshUi();
  }

  function bindSlider(
    root: HTMLElement,
    id: string,
    set: (v: number) => void,
    fmt: (v: number) => string,
  ) {
    const slider = root.querySelector(id) as HTMLInputElement | null;
    const display = root.querySelector(id + '-value') as HTMLElement | null;
    if (!slider) return;
    const sync = () => {
      const v = parseFloat(slider.value);
      set(v);
      if (display) display.textContent = fmt(v);
    };
    sync();
    slider.addEventListener('input', sync);
  }

  function refreshUi() {
    if (fpsCheckbox) fpsCheckbox.checked = cfg.fps;
    if (driveCheckbox) driveCheckbox.checked = cfg.drive;
    if (headlampCheckbox) headlampCheckbox.checked = cfg.headlamp;
    if (modeSelect) modeSelect.value = cfg.mode;
    const armed = countArmed();
    if (armedCountEl) armedCountEl.textContent = String(armed);
    crosshair.style.display = cfg.fps ? 'block' : 'none';
    if (cfg.drive) {
      // Live speed is refreshed each frame by updateDrive(); this is the resting text.
      badge.innerHTML =
        '<span class="shooter-badge-mode">🚗 Drive</span>' +
        '<span class="shooter-badge-dim"> · WASD drive · Space brake · R flip · C exit</span>';
      return;
    }
    const modeLabel =
      cfg.mode === 'ball' ? '① Ball' : cfg.mode === 'sticky' ? '② Sticky explosive' : '③ Grenade';
    const fpsLabel = cfg.fps ? (pointerLocked ? 'on' : 'on — click to look') : 'off';
    badge.innerHTML =
      `<span class="shooter-badge-mode">${modeLabel}</span>` +
      `<span class="shooter-badge-dim"> · armed ${armed} · FPS ${fpsLabel}</span>`;
  }

  function countArmed(): number {
    let n = 0;
    for (const s of stickies) if (s.stuck && !s.exploded) n++;
    return n;
  }

  // ── Mode + FPS toggles ──────────────────────────────────────────
  function setMode(mode: ShootMode) {
    cfg.mode = mode;
    refreshUi();
  }
  const MODE_ORDER: ShootMode[] = ['ball', 'sticky', 'grenade'];
  function cycleMode() {
    setMode(MODE_ORDER[(MODE_ORDER.indexOf(cfg.mode) + 1) % MODE_ORDER.length]);
  }

  function setFpsEnabled(on: boolean) {
    if (on === cfg.fps) {
      refreshUi();
      return;
    }
    cfg.fps = on;
    if (on) {
      setDriveEnabled(false); // the two player modes are mutually exclusive
      controls.enabled = false;
      // Face the structure (yaw from the current orbit view). The character body
      // spawns near here and drives the camera from then on.
      camera.lookAt(controls.target);
      _e.setFromQuaternion(camera.quaternion, 'YXZ');
      yaw = _e.y;
      pitch = 0;
      velY = 0;
      grounded = false;
      const core = getCore();
      if (core) ensureCharacter(core);
    } else {
      destroyCharacter(true);
      controls.enabled = true;
      if (pointerLocked && document.pointerLockElement === canvas) document.exitPointerLock();
    }
    refreshUi();
  }

  // ── Drive mode (Rapier raycast vehicle) ─────────────────────────
  function setDriveEnabled(on: boolean) {
    if (on === cfg.drive) {
      refreshUi();
      return;
    }
    cfg.drive = on;
    if (on) {
      setFpsEnabled(false); // mutually exclusive with first-person
      controls.enabled = false;
      camOrbitYaw = 0;
      const core = getCore();
      if (core) ensureVehicle(core); // also re-spawned lazily by updateDrive
    } else {
      disposeVehicle();
      controls.enabled = true;
      if (pointerLocked && document.pointerLockElement === canvas) document.exitPointerLock();
    }
    refreshUi();
  }

  // Ground-level spawn near where the orbit camera was looking, facing the
  // structure. Local +Z is the car's forward, so heading = atan2(dirX, dirZ).
  function carSpawnPoint(): { position: Vec3; headingY: number } {
    const target = controls.target;
    _v.set(camera.position.x - target.x, 0, camera.position.z - target.z);
    let dist = _v.length();
    if (dist < 1e-3) _v.set(0, 0, 1), (dist = 1);
    _v.normalize();
    dist = Math.min(Math.max(dist, 12), 40);
    const x = target.x + _v.x * dist;
    const z = target.z + _v.z * dist;
    return {
      position: { x, y: floorY + 0.95, z }, // 0.95 ≈ wheel-on-ground ride height
      headingY: Math.atan2(target.x - x, target.z - z),
    };
  }

  function ensureVehicle(core: DestructibleCore) {
    if (vehicle && vehicleCore === core) return;
    if (vehicle && vehicleCore !== core) {
      // Refs belong to a stale (disposed) world after a Reset — drop the visuals
      // without touching the dead world, then respawn below.
      vehicle.disposeVisuals();
      vehicle = null;
      vehicleCore = null;
    }
    const sp = carSpawnPoint();
    vehicle = createVehicle({
      world: core.world as RAPIER.World,
      scene,
      position: sp.position,
      headingY: sp.headingY,
      tuning: cfg.car,
    });
    vehicleCore = core;
  }

  function disposeVehicle() {
    if (vehicle) {
      // Live world at toggle-off time → full teardown; dispose() also removes meshes.
      if (vehicleCore === getCore()) vehicle.dispose();
      else vehicle.disposeVisuals();
    }
    vehicle = null;
    vehicleCore = null;
  }

  // Ground-level spawn near where the orbit camera was looking.
  function spawnPoint(): { x: number; y: number; z: number } {
    const target = controls.target;
    _v.set(camera.position.x - target.x, 0, camera.position.z - target.z);
    let dist = _v.length();
    if (dist < 1e-3) _v.set(0, 0, 1), (dist = 1);
    _v.normalize();
    dist = Math.min(Math.max(dist, 8), 45);
    return {
      x: target.x + _v.x * dist,
      y: floorY + CAPSULE.halfHeight + CAPSULE.radius + 0.3,
      z: target.z + _v.z * dist,
    };
  }

  function ensureCharacter(core: DestructibleCore) {
    if (charBody && charCore === core) return;
    if (charCore !== core) {
      // Refs belong to a stale (disposed) world after a Reset — drop them.
      charBody = null;
      charCollider = null;
      charController = null;
      charCore = null;
    }
    const world = core.world as RAPIER.World;
    const s = spawnPoint();
    charBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(s.x, s.y, s.z),
    );
    charCollider = world.createCollider(
      RAPIER.ColliderDesc.capsule(CAPSULE.halfHeight, CAPSULE.radius),
      charBody,
    );
    // Ghost the capsule physically (filter = 0): nothing bounces off the player,
    // so projectiles/charges fired from the camera don't collide with you. The
    // controller still navigates the world via its own query (permissive
    // filterGroups + charQueryFilter) below, so you're still blocked by walls,
    // stand on floors/debris and can push debris.
    charCollider.setCollisionGroups(0x00010000);
    const ctrl = world.createCharacterController(0.02);
    ctrl.enableAutostep(0.5, 0.2, true); // climb small debris / steps
    ctrl.enableSnapToGround(0.5); // stay glued to surfaces walking down
    ctrl.setApplyImpulsesToDynamicBodies(true); // shove loose debris around
    ctrl.setCharacterMass(80);
    ctrl.setMaxSlopeClimbAngle((50 * Math.PI) / 180);
    ctrl.setMinSlopeSlideAngle((40 * Math.PI) / 180);
    charController = ctrl;
    charCore = core;
    velY = 0;
    grounded = false;
  }

  function destroyCharacter(removeFromWorld: boolean) {
    if (removeFromWorld && charCore) {
      const world = charCore.world as RAPIER.World;
      try { if (charController) world.removeCharacterController(charController); } catch { /* ignore */ }
      try { if (charBody) world.removeRigidBody(charBody); } catch { /* ignore */ }
    }
    charBody = null;
    charCollider = null;
    charController = null;
    charCore = null;
  }

  // ── Shooting ────────────────────────────────────────────────────
  function aimDirection(ndcX: number, ndcY: number, out: THREE.Vector3) {
    _ndc.set(ndcX, ndcY);
    _ray.setFromCamera(_ndc, camera);
    out.copy(_ray.ray.direction).normalize();
  }

  function shoot(ndcX: number, ndcY: number) {
    const core = getCore();
    if (!core) return;
    if (cfg.mode === 'ball') {
      if (opts.shootBall) opts.shootBall(ndcX, ndcY);
      else { aimDirection(ndcX, ndcY, _v); shootBall(core, _v); }
    } else {
      aimDirection(ndcX, ndcY, _v);
      shootCharge(core, _v, cfg.mode === 'grenade');
    }
  }

  function shootBall(core: DestructibleCore, dir: THREE.Vector3) {
    const p = getBallParams();
    const speed = p.speed ?? DEFAULT_BALL.speed;
    core.enqueueProjectile({
      position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
      velocity: { x: dir.x * speed, y: dir.y * speed, z: dir.z * speed },
      radius: p.radius ?? DEFAULT_BALL.radius,
      mass: p.mass ?? DEFAULT_BALL.mass,
      ttl: 6000,
    });
  }

  // Launch a thrown charge. `explodeOnContact` => grenade (blows up the instant
  // it touches anything); otherwise it's a sticky charge that waits for detonate.
  function shootCharge(core: DestructibleCore, dir: THREE.Vector3, explodeOnContact: boolean) {
    const world = core.world as RAPIER.World;
    const r = cfg.sticky.radius;
    // Spawn essentially at the camera (tiny offset only) so the charge starts
    // from the eye and doesn't skip over objects right in front of you. The
    // character capsule is excluded from contact below so it can't self-trigger.
    const off = r + 0.05;
    const ox = camera.position.x + dir.x * off;
    const oy = camera.position.y + dir.y * off;
    const oz = camera.position.z + dir.z * off;
    const speed = cfg.sticky.speed;

    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(ox, oy, oz)
      .setLinvel(dir.x * speed, dir.y * speed, dir.z * speed)
      .setCcdEnabled(true)
      .setUserData({ stickyExplosive: true });
    const body = world.createRigidBody(bodyDesc);
    const colDesc = RAPIER.ColliderDesc.ball(r)
      .setMass(cfg.sticky.mass)
      .setFriction(1.0)
      .setRestitution(0.0);
    const collider = world.createCollider(colDesc, body);

    const mesh = makeChargeMesh(r, explodeOnContact);
    mesh.position.set(ox, oy, oz);
    scene.add(mesh);

    stickies.push({
      mesh,
      bodyHandle: body.handle,
      colliderHandle: collider.handle,
      flying: true,
      stuck: false,
      exploded: false,
      explodeOnContact,
      spawnTime: performance.now(),
      radius: r,
      attachBodyHandle: null,
      relPos: new THREE.Vector3(),
      relQuat: new THREE.Quaternion(),
      worldPos: new THREE.Vector3(ox, oy, oz),
    });
  }

  function makeChargeMesh(r: number, grenade: boolean): THREE.Mesh {
    const color = grenade ? 0xffa022 : 0xff2a2a; // grenade = orange, sticky = red
    const geo = new THREE.SphereGeometry(r, 18, 14);
    const mat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.6,
      roughness: 0.5,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    return mesh;
  }

  // ── Sticky lifecycle (called each frame, after the world has stepped) ──
  function updateStickies(core: DestructibleCore) {
    const world = core.world as RAPIER.World;
    const now = performance.now();
    for (let i = stickies.length - 1; i >= 0; i--) {
      const s = stickies[i];
      if (s.exploded) {
        removeSticky(s);
        stickies.splice(i, 1);
        continue;
      }

      if (s.flying) {
        const body = s.bodyHandle != null ? world.getRigidBody(s.bodyHandle) : null;
        if (!body) {
          removeSticky(s);
          stickies.splice(i, 1);
          continue;
        }
        const t = body.translation();
        // Missed everything and fell out of the world → drop it.
        if (t.y < floorY - 80 || now - s.spawnTime > cfg.sticky.ttl * 1000) {
          try { world.removeRigidBody(body); } catch { /* ignore */ }
          removeSticky(s);
          stickies.splice(i, 1);
          continue;
        }

        // Find the first thing we're touching and stick to it.
        let hit: RAPIER.RigidBody | null = null;
        let hitIsFixed = false;
        const col = s.colliderHandle != null ? world.getCollider(s.colliderHandle) : null;
        if (col) {
          world.contactPairsWith(col, (other: RAPIER.Collider) => {
            if (hit) return;
            const pb = other.parent();
            // Ignore the charge's own body and the player's character capsule.
            if (pb && pb.handle !== body.handle && (!charBody || pb.handle !== charBody.handle)) {
              hit = pb;
              hitIsFixed = pb.isFixed();
            }
          });
        }
        // Fallback: it has come to rest on something (e.g. a soft ground landing).
        const lv = body.linvel();
        const slow = lv.x * lv.x + lv.y * lv.y + lv.z * lv.z < 0.5;
        const settled = slow && now - s.spawnTime > 250;

        if ((hit || settled) && s.explodeOnContact) {
          // Grenade: blow up the instant it first touches anything.
          explodeCharge(core, s, body);
          removeSticky(s);
          stickies.splice(i, 1);
          refreshUi();
        } else if (!hit && settled) {
          stickInPlace(s, body);
        } else if (hit) {
          stickToBody(s, body, hit, hitIsFixed);
        } else {
          // Still flying — track the body.
          const rot = body.rotation();
          s.mesh.position.set(t.x, t.y, t.z);
          s.mesh.quaternion.set(rot.x, rot.y, rot.z, rot.w);
          s.worldPos.set(t.x, t.y, t.z);
        }
        continue;
      }

      // Stuck: follow the attach body so charges ride moving pillars/debris.
      if (s.stuck) {
        const attach = s.attachBodyHandle != null ? world.getRigidBody(s.attachBodyHandle) : null;
        if (attach) {
          const t = attach.translation();
          const r = attach.rotation();
          _q.set(r.x, r.y, r.z, r.w);
          _v2.copy(s.relPos).applyQuaternion(_q);
          s.worldPos.set(t.x + _v2.x, t.y + _v2.y, t.z + _v2.z);
          s.mesh.position.copy(s.worldPos);
          s.mesh.quaternion.copy(_q).multiply(s.relQuat);
        }
        // Gentle "armed" pulse.
        const mat = s.mesh.material as THREE.MeshStandardMaterial;
        mat.emissiveIntensity = 0.5 + 0.4 * (0.5 + 0.5 * Math.sin(now * 0.008));
      }
    }
  }

  function stickToBody(s: StickyExplosive, body: RAPIER.RigidBody, attach: RAPIER.RigidBody, _fixed: boolean) {
    const world = (getCore()!.world as RAPIER.World);
    const t = body.translation();
    const rot = body.rotation();
    const at = attach.translation();
    const ar = attach.rotation();
    // Charge transform expressed in the attach body's local frame.
    _q.set(ar.x, ar.y, ar.z, ar.w).invert();
    _v2.set(t.x - at.x, t.y - at.y, t.z - at.z).applyQuaternion(_q);
    s.relPos.copy(_v2);
    s.relQuat.set(rot.x, rot.y, rot.z, rot.w).premultiply(_q);
    s.attachBodyHandle = attach.handle;
    s.worldPos.set(t.x, t.y, t.z);
    finishStick(s, body, world);
  }

  function stickInPlace(s: StickyExplosive, body: RAPIER.RigidBody) {
    const world = (getCore()!.world as RAPIER.World);
    const t = body.translation();
    s.attachBodyHandle = null;
    s.worldPos.set(t.x, t.y, t.z);
    s.mesh.position.set(t.x, t.y, t.z);
    finishStick(s, body, world);
  }

  function finishStick(s: StickyExplosive, body: RAPIER.RigidBody, world: RAPIER.World) {
    try { world.removeRigidBody(body); } catch { /* ignore */ }
    s.bodyHandle = null;
    s.colliderHandle = null;
    s.flying = false;
    s.stuck = true;
    refreshUi();
  }

  // ── Detonation + explosion ──────────────────────────────────────
  function detonateAll() {
    const core = getCore();
    if (!core) return;
    let any = false;
    for (const s of stickies) {
      if (!s.stuck || s.exploded) continue;
      explodeAt(core, s.worldPos);
      spawnShockwave(s.worldPos, cfg.blast.radius);
      s.exploded = true;
      any = true;
    }
    if (any) refreshUi();
  }

  // Blow up a single in-flight charge (grenade) at its current position.
  function explodeCharge(core: DestructibleCore, s: StickyExplosive, body: RAPIER.RigidBody) {
    const t = body.translation();
    s.worldPos.set(t.x, t.y, t.z);
    try { (core.world as RAPIER.World).removeRigidBody(body); } catch { /* ignore */ }
    s.bodyHandle = null;
    s.colliderHandle = null;
    s.flying = false;
    s.exploded = true;
    explodeAt(core, s.worldPos);
    spawnShockwave(s.worldPos, cfg.blast.radius);
  }

  // Register a blast at `center`. Intact chunks live on the *fixed* root/actor
  // bodies, so a Rapier force does nothing to them — fracturing them needs the
  // load fed into the stress solver (the path projectile impacts use). And the
  // outward fling can't be applied while a chunk is still fixed/bonded, so it
  // must wait until the piece actually breaks free. We freeze the in-range chunk
  // set + outward direction here, then over the next ~0.6 s updateBlastFields()
  // keeps seeding the fracture load AND kicks each piece outward the instant it
  // becomes dynamic — however many frames the fracture cascade takes.
  function explodeAt(core: DestructibleCore, center: THREE.Vector3) {
    const world = core.world as RAPIER.World;
    const radius = cfg.blast.radius;
    const candidates: BlastCandidate[] = [];
    for (const chunk of core.chunks) {
      if (!chunk.active || chunk.destroyed || chunk.bodyHandle == null) continue;
      const body = world.getRigidBody(chunk.bodyHandle);
      if (!body) continue;
      const t = body.translation();
      const r = body.rotation();
      _q.set(r.x, r.y, r.z, r.w);
      _v.set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z).applyQuaternion(_q);
      const dx = t.x + _v.x - center.x;
      const dy = t.y + _v.y - center.y;
      const dz = t.z + _v.z - center.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > radius) continue;
      const inv = dist > 1e-4 ? 1 / dist : 0;
      candidates.push({
        node: chunk.nodeIndex,
        falloff: 1 - dist / radius,
        nx: dist > 1e-4 ? dx * inv : 0,
        ny: dist > 1e-4 ? dy * inv : 1,
        nz: dist > 1e-4 ? dz * inv : 0,
      });
    }
    if (!candidates.length) return;
    blastFields.push({
      cx: center.x,
      cy: center.y,
      cz: center.z,
      up: cfg.blast.up,
      solverForce: cfg.blast.strength * BLAST_SOLVER_FORCE, // peak Newtons at centre
      kick: Math.min(cfg.blast.strength, 100) * 0.32, // peak Δv (m/s) at centre, applied once per piece
      age: 0,
      life: 45, // ~0.75 s: long enough for the fracture cascade to free every piece
      solverFrames: 4, // sustain the fracture load for a few frames to propagate collapse
      candidates,
      kicked: new Set<number>(),
    });
  }

  // Drive active blast fields: keep seeding bond stress, and give each piece a
  // single outward impulse the moment it becomes a free dynamic body.
  function updateBlastFields(core: DestructibleCore) {
    if (!blastFields.length) return;
    const world = core.world as RAPIER.World;
    const solver = core.solver;
    for (let i = blastFields.length - 1; i >= 0; i--) {
      const f = blastFields[i];
      const injectSolver = f.age < f.solverFrames;

      for (const c of f.candidates) {
        const chunk = core.chunks[c.node];
        if (!chunk || !chunk.active || chunk.destroyed || chunk.bodyHandle == null) continue;
        const body = world.getRigidBody(chunk.bodyHandle);
        if (!body) continue;

        // Fracture pass: feed the outward load into the stress solver so bonds
        // overstress and the structure comes apart (works on fixed bodies too).
        if (injectSolver) {
          const mag = f.solverForce * c.falloff;
          _v2.set(c.nx * mag, c.ny * mag + mag * f.up, c.nz * mag);
          const r = body.rotation();
          _qinv.set(r.x, r.y, r.z, r.w).conjugate();
          _v2.applyQuaternion(_qinv); // world force → body-local for the solver
          try {
            solver.addForce(c.node, chunk.baseLocalOffset, { x: _v2.x, y: _v2.y, z: _v2.z });
          } catch { /* ignore */ }
        }

        // Kinetic pass: as soon as a piece is free (its own dynamic body), kick
        // it outward once. This is what was missing — pieces freed a few frames
        // after impact (bonded "grey" structure) now fly back, not just settle.
        if (body.isDynamic() && !f.kicked.has(chunk.bodyHandle)) {
          f.kicked.add(chunk.bodyHandle);
          const j = body.mass() * f.kick * c.falloff;
          body.applyImpulse({ x: c.nx * j, y: (c.ny + f.up) * j, z: c.nz * j }, true);
        }
      }

      f.age += 1;
      f.life -= 1;
      if (f.life <= 0) blastFields.splice(i, 1);
    }
  }

  // ── Shockwave VFX ───────────────────────────────────────────────
  function spawnShockwave(center: THREE.Vector3, radius: number) {
    const geo = new THREE.SphereGeometry(1, 20, 16);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffb14a,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(center);
    mesh.scale.setScalar(0.1);
    scene.add(mesh);
    let light: THREE.PointLight | null = null;
    try {
      light = new THREE.PointLight(0xffa030, 6, radius * 3);
      light.position.copy(center);
      scene.add(light);
    } catch { /* ignore */ }
    transient.push({ mesh, light, born: performance.now(), ttl: 480, radius });
  }

  function updateTransient(now: number) {
    for (let i = transient.length - 1; i >= 0; i--) {
      const fx = transient[i];
      const k = (now - fx.born) / fx.ttl;
      if (k >= 1) {
        scene.remove(fx.mesh);
        fx.mesh.geometry.dispose();
        (fx.mesh.material as THREE.Material).dispose();
        if (fx.light) scene.remove(fx.light);
        transient.splice(i, 1);
        continue;
      }
      const ease = 1 - (1 - k) * (1 - k); // ease-out
      fx.mesh.scale.setScalar(Math.max(0.1, fx.radius * ease));
      (fx.mesh.material as THREE.MeshBasicMaterial).opacity = 0.5 * (1 - k);
      if (fx.light) fx.light.intensity = 6 * (1 - k);
    }
  }

  // ── FPS camera (driven by the Rapier kinematic character controller) ──
  function updateCamera(dt: number) {
    // Arrow-key look (complements mouse look; works without pointer lock).
    if (keys.has('ArrowLeft')) yaw += cfg.keyLookSpeed * dt;
    if (keys.has('ArrowRight')) yaw -= cfg.keyLookSpeed * dt;
    if (keys.has('ArrowUp')) pitch += cfg.keyLookSpeed * dt;
    if (keys.has('ArrowDown')) pitch -= cfg.keyLookSpeed * dt;
    pitch = Math.min(Math.max(pitch, -1.5), 1.5);

    _e.set(pitch, yaw, 0, 'YXZ');
    camera.quaternion.setFromEuler(_e);

    const core = getCore();
    if (!core) return; // no world yet (e.g. mid-rebuild)
    ensureCharacter(core);
    if (!charBody || !charCollider || !charController) return;
    const world = core.world as RAPIER.World;

    // Horizontal move direction (yaw-relative, on the ground plane).
    _fwd.set(0, 0, -1).applyQuaternion(camera.quaternion);
    _fwd.y = 0;
    if (_fwd.lengthSq() < 1e-6) _fwd.set(0, 0, -1);
    _fwd.normalize();
    _right.copy(_fwd).cross(UP).normalize();

    _move.set(0, 0, 0);
    if (keys.has('KeyW')) _move.add(_fwd);
    if (keys.has('KeyS')) _move.sub(_fwd);
    if (keys.has('KeyD')) _move.add(_right);
    if (keys.has('KeyA')) _move.sub(_right);
    if (_move.lengthSq() > 1e-6) _move.normalize().multiplyScalar(cfg.walkSpeed);

    // Vertical: Space is a jetpack (hold to rise); otherwise gravity pulls you
    // down until the controller reports you're standing on something.
    let dy: number;
    if (keys.has('Space')) {
      velY = cfg.jetUpSpeed;
      dy = velY * dt;
    } else if (grounded) {
      velY = 0;
      dy = -0.1; // small downward bias keeps snap-to-ground engaged
    } else {
      velY = Math.max(velY - cfg.fallGravity * dt, -cfg.maxFall);
      dy = velY * dt;
    }

    // Resolve the desired motion against the world (slides, steps, lands).
    // Permissive filterGroups (the capsule itself is a physics ghost) + a
    // predicate that skips projectiles/charges so they don't block the player.
    charController.computeColliderMovement(
      charCollider,
      { x: _move.x * dt, y: dy, z: _move.z * dt },
      undefined,
      0xffffffff,
      charQueryFilter,
    );
    grounded = charController.computedGrounded();
    const corr = charController.computedMovement();
    const p = charBody.translation();
    const nx = p.x + corr.x;
    const ny = p.y + corr.y;
    const nz = p.z + corr.z;
    charBody.setNextKinematicTranslation({ x: nx, y: ny, z: nz });
    camera.position.set(nx, ny + EYE_ABOVE_CENTER, nz);
  }

  // ── Drive update (raycast vehicle + chase camera) ───────────────
  function updateDrive(dt: number) {
    const core = getCore();
    if (!core) return; // no world yet (e.g. mid-rebuild)
    ensureVehicle(core);
    if (!vehicle) return;

    let throttle = 0;
    if (keys.has('KeyW') || keys.has('ArrowUp')) throttle += 1;
    if (keys.has('KeyS') || keys.has('ArrowDown')) throttle -= 1;
    // Steer relative to the chase camera: A/left turns the car left on screen
    // (toward world +X, which is screen-left because the camera looks down +Z).
    let steer = 0;
    if (keys.has('KeyA') || keys.has('ArrowLeft')) steer += 1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) steer -= 1;
    const handbrake = keys.has('Space');

    vehicle.applyControls({ throttle, brake: 0, steer, handbrake });
    vehicle.step(dt); // sets wheels + updateVehicle(dt); impulses integrate next core.step
    vehicle.syncMeshes();
    updateChaseCamera(dt);

    // Live speedometer in the on-screen badge.
    badge.innerHTML =
      '<span class="shooter-badge-mode">🚗 Drive</span>' +
      `<span class="shooter-badge-dim"> · ${Math.round(Math.abs(vehicle.speedKmh()))} km/h · R flip · C exit</span>`;
  }

  // Third-person follow camera: behind + above the car along its heading,
  // smoothed; mouse drag (while pointer-locked) orbits and decays back to centre.
  function updateChaseCamera(dt: number) {
    if (!vehicle) return;
    const body = vehicle.chassisBody();
    const p = body.translation();
    const r = body.rotation();
    // Yaw only, so the camera stays upright even when the car pitches/rolls.
    const yawCar = Math.atan2(2 * (r.w * r.y + r.x * r.z), 1 - 2 * (r.y * r.y + r.z * r.z));
    camOrbitYaw *= Math.max(0, 1 - 3 * dt);
    const ang = yawCar + camOrbitYaw;
    const fx = Math.sin(ang);
    const fz = Math.cos(ang);
    _v.set(p.x - fx * cfg.chase.dist, p.y + cfg.chase.height, p.z - fz * cfg.chase.dist);
    if (_v.y < floorY + 0.6) _v.y = floorY + 0.6; // never sink under the ground
    camera.position.lerp(_v, 1 - Math.exp(-cfg.chase.lerp * dt));
    _v2.set(p.x, p.y + cfg.chase.look, p.z);
    camera.lookAt(_v2);
  }

  // ── Per-frame entry point ───────────────────────────────────────
  function update() {
    const now = performance.now();
    const dt = Math.min(Math.max((now - lastT) / 1000, 0), 1 / 20);
    lastT = now;

    const core = getCore();
    if (core !== lastCore) {
      // The demo rebuilt: the old world (and all our charge bodies) is gone.
      handleCoreSwap();
      lastCore = core;
    }

    if (cfg.fps) updateCamera(dt);
    else if (cfg.drive) updateDrive(dt);
    if (core) updateStickies(core);
    if (core) updateBlastFields(core);
    updateTransient(now);

    // Headlamp rides the camera (FPS and orbit alike), aiming where you look.
    headlamp.visible = cfg.headlamp;
    if (cfg.headlamp) {
      headlamp.position.copy(camera.position);
      _look.set(0, 0, -1).applyQuaternion(camera.quaternion);
      headlamp.target.position.copy(camera.position).add(_look);
    }

    // Keep the armed-count + pulse readouts live.
    if (armedCountEl) {
      const n = countArmed();
      if (armedCountEl.textContent !== String(n)) refreshUi();
    }
  }

  function handleCoreSwap() {
    // The old world (and our character body) is gone; drop refs without touching
    // it. updateCamera re-creates the character in the new world if FPS is on.
    charBody = null;
    charCollider = null;
    charController = null;
    charCore = null;
    // Same for the car: remove its meshes (the bodies died with the old world);
    // updateDrive respawns it in the new world if drive mode is still on.
    if (vehicle) {
      vehicle.disposeVisuals();
      vehicle = null;
      vehicleCore = null;
    }
    blastFields.length = 0;
    for (const s of stickies) removeSticky(s);
    stickies.length = 0;
    for (const fx of transient) {
      scene.remove(fx.mesh);
      try { fx.mesh.geometry.dispose(); } catch { /* ignore */ }
      try { (fx.mesh.material as THREE.Material).dispose(); } catch { /* ignore */ }
      if (fx.light) scene.remove(fx.light);
    }
    transient.length = 0;
    refreshUi();
  }

  function removeSticky(s: StickyExplosive) {
    scene.remove(s.mesh);
    try { s.mesh.geometry.dispose(); } catch { /* ignore */ }
    try { (s.mesh.material as THREE.Material).dispose(); } catch { /* ignore */ }
  }

  // ── Input wiring ────────────────────────────────────────────────
  function isTyping(): boolean {
    const el = document.activeElement as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || el.isContentEditable;
  }

  canvas.addEventListener('click', (e: MouseEvent) => {
    if (cfg.drive) {
      // Grab the pointer so the mouse can orbit the chase camera; no shooting.
      if (!pointerLocked) {
        try {
          const p = canvas.requestPointerLock() as unknown as Promise<void> | undefined;
          if (p && typeof p.catch === 'function') p.catch(() => { /* ignore */ });
        } catch { /* ignore */ }
      }
      return;
    }
    if (cfg.fps) {
      if (!pointerLocked) {
        // requestPointerLock returns a promise in newer browsers; swallow a
        // rejection (e.g. user exited too recently) so it isn't logged.
        try {
          const p = canvas.requestPointerLock() as unknown as Promise<void> | undefined;
          if (p && typeof p.catch === 'function') p.catch(() => { /* ignore */ });
        } catch { /* ignore */ }
        return; // first click only grabs the pointer
      }
      shoot(0, 0); // locked → fire straight through the crosshair
    } else {
      const rect = canvas.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      shoot(ndcX, ndcY);
    }
  });

  document.addEventListener('pointerlockchange', () => {
    pointerLocked = document.pointerLockElement === canvas;
    refreshUi();
  });

  document.addEventListener('mousemove', (e: MouseEvent) => {
    if (!pointerLocked) return;
    if (cfg.drive) {
      // Orbit the chase camera around the car (decays back behind it).
      camOrbitYaw -= e.movementX * cfg.mouseSensitivity;
      camOrbitYaw = Math.min(Math.max(camOrbitYaw, -1.2), 1.2);
      return;
    }
    if (!cfg.fps) return;
    yaw -= e.movementX * cfg.mouseSensitivity;
    pitch -= e.movementY * cfg.mouseSensitivity;
    pitch = Math.min(Math.max(pitch, -1.5), 1.5);
  });

  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (isTyping()) return;
    // R flips the car upright — only while driving, so it doesn't shadow other keys.
    if (cfg.drive && e.code === 'KeyR') {
      vehicle?.recover();
      e.preventDefault();
      return;
    }
    switch (e.code) {
      case 'Digit1': setMode('ball'); return;
      case 'Digit2': setMode('sticky'); return;
      case 'Digit3': setMode('grenade'); return;
      case 'KeyQ': cycleMode(); return;
      case 'KeyF': detonateAll(); return;
      case 'KeyV': setFpsEnabled(!cfg.fps); return;
      case 'KeyC': setDriveEnabled(!cfg.drive); return;
    }
    if (cfg.fps || cfg.drive) {
      // Movement / look keys — track and stop the page from scrolling.
      if (
        e.code === 'KeyW' || e.code === 'KeyA' || e.code === 'KeyS' || e.code === 'KeyD' ||
        e.code === 'Space' || e.code === 'ArrowUp' || e.code === 'ArrowDown' ||
        e.code === 'ArrowLeft' || e.code === 'ArrowRight'
      ) {
        keys.add(e.code);
        e.preventDefault();
      }
    }
  });

  window.addEventListener('keyup', (e: KeyboardEvent) => {
    keys.delete(e.code);
  });

  // ── Boot ────────────────────────────────────────────────────────
  if (document.getElementById('sidebar')) mountSidebar();
  else if (document.body) mountSidebar();
  else window.addEventListener('DOMContentLoaded', mountSidebar);
  refreshUi();

  return {
    update,
    setFpsEnabled,
    isFpsEnabled: () => cfg.fps,
    setDriveEnabled,
    isDriveEnabled: () => cfg.drive,
    setMode,
    detonateAll,
  };
}

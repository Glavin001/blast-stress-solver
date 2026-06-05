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

export function mountShooter(opts: ShooterOptions): ShooterHandle {
  const { canvas, camera, controls, scene, getCore } = opts;
  const getBallParams = opts.getBallParams ?? (() => DEFAULT_BALL);
  const floorY = opts.floorY ?? 0;
  const eyeHeight = opts.eyeHeight ?? 1.7;

  // ── Tunable state (mutated by the sidebar) ──────────────────────
  const cfg = {
    fps: false,
    mode: 'ball' as ShootMode,
    headlamp: true, // camera-mounted light so the view is always lit
    walkSpeed: 9, // m/s
    jumpSpeed: 7, // m/s
    gravity: 20, // m/s² (snappier than real g for a responsive walker)
    keyLookSpeed: 1.9, // rad/s for arrow-key look
    mouseSensitivity: 0.0022, // rad / pixel
    sticky: { radius: 0.28, mass: 60, speed: 44, ttl: 30 },
    blast: { radius: 7, strength: 55, up: 0.4 },
  };

  // ── Runtime state ───────────────────────────────────────────────
  let lastCore: DestructibleCore | null = null;
  const stickies: StickyExplosive[] = [];
  const transient: Transient[] = [];
  const keys = new Set<string>();
  let yaw = 0;
  let pitch = 0;
  let velY = 0;
  let grounded = true;
  let pointerLocked = false;
  let lastT = performance.now();

  // Scratch objects (avoid per-frame allocation on the hot path).
  const _ray = new THREE.Raycaster();
  const _ndc = new THREE.Vector2();
  const _q = new THREE.Quaternion();
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
      '<small>WASD walk · arrows / mouse look · Space jump · click shoots. ' +
      'Press <b>V</b> to toggle.</small></span></label>' +
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
    headlampCheckbox = section.querySelector('#cfg-headlamp');
    modeSelect = section.querySelector('#cfg-shoot-mode');
    armedCountEl = section.querySelector('#cfg-armed');

    fpsCheckbox!.checked = cfg.fps;
    fpsCheckbox!.addEventListener('change', () => setFpsEnabled(fpsCheckbox!.checked));

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
    if (headlampCheckbox) headlampCheckbox.checked = cfg.headlamp;
    if (modeSelect) modeSelect.value = cfg.mode;
    const armed = countArmed();
    if (armedCountEl) armedCountEl.textContent = String(armed);
    crosshair.style.display = cfg.fps ? 'block' : 'none';
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
      controls.enabled = false;
      // Drop in at ground level near where the orbit camera was looking, facing
      // the structure — so toggling on doesn't disorient or bury the camera.
      const target = controls.target.clone();
      _v.set(camera.position.x - target.x, 0, camera.position.z - target.z);
      let dist = _v.length();
      if (dist < 1e-3) _v.set(0, 0, 1), (dist = 1);
      _v.normalize();
      dist = Math.min(Math.max(dist, 8), 45);
      camera.position.set(target.x + _v.x * dist, floorY + eyeHeight, target.z + _v.z * dist);
      camera.lookAt(target.x, floorY + eyeHeight, target.z);
      _e.setFromQuaternion(camera.quaternion, 'YXZ');
      yaw = _e.y;
      pitch = 0;
      velY = 0;
      grounded = true;
    } else {
      controls.enabled = true;
      if (pointerLocked && document.pointerLockElement === canvas) document.exitPointerLock();
    }
    refreshUi();
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
    // Spawn just ahead of the eye so the charge never starts inside the camera.
    const ox = camera.position.x + dir.x * (r + 0.4);
    const oy = camera.position.y + dir.y * (r + 0.4);
    const oz = camera.position.z + dir.z * (r + 0.4);
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
            if (pb && pb.handle !== body.handle) {
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

  function explodeAt(core: DestructibleCore, center: THREE.Vector3) {
    const world = core.world as RAPIER.World;
    const radius = cfg.blast.radius;
    const accel = cfg.blast.strength * 9.81; // peak acceleration at the centre
    const up = cfg.blast.up;

    // Per-body node counts → approximate per-node mass for an even blast.
    const nodeCount = new Map<number, number>();
    for (const c of core.chunks) {
      if (c.active && c.bodyHandle != null && !c.destroyed) {
        nodeCount.set(c.bodyHandle, (nodeCount.get(c.bodyHandle) ?? 0) + 1);
      }
    }
    const chunkBodies = new Set<number>();

    // 1) Fracture pass — push every live chunk outward via the stress solver.
    for (const chunk of core.chunks) {
      if (!chunk.active || chunk.destroyed || chunk.bodyHandle == null) continue;
      const body = world.getRigidBody(chunk.bodyHandle);
      if (!body || body.isFixed()) continue;
      chunkBodies.add(chunk.bodyHandle);

      const t = body.translation();
      const r = body.rotation();
      _q.set(r.x, r.y, r.z, r.w);
      _v.set(chunk.baseLocalOffset.x, chunk.baseLocalOffset.y, chunk.baseLocalOffset.z).applyQuaternion(_q);
      const wx = t.x + _v.x;
      const wy = t.y + _v.y;
      const wz = t.z + _v.z;

      const dx = wx - center.x;
      const dy = wy - center.y;
      const dz = wz - center.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > radius) continue;
      const falloff = 1 - dist / radius;
      const inv = dist > 1e-4 ? 1 / dist : 0;
      const nx = dist > 1e-4 ? dx * inv : 0;
      const ny = dist > 1e-4 ? dy * inv : 1;
      const nz = dist > 1e-4 ? dz * inv : 0;

      const perNode = Math.max(1, body.mass() / (nodeCount.get(chunk.bodyHandle) ?? 1));
      const mag = perNode * accel * falloff;
      core.applyExternalForce(
        chunk.nodeIndex,
        { x: wx, y: wy, z: wz },
        { x: nx * mag, y: ny * mag + perNode * accel * up * falloff, z: nz * mag },
      );
    }

    // 2) Kinetic pass — fling loose bodies (projectiles, in-flight charges)
    //    that aren't part of the chunk graph, for extra spectacle.
    const kick = Math.min(cfg.blast.strength, 60) * 0.45; // peak Δv at the centre
    world.forEachRigidBody((body: RAPIER.RigidBody) => {
      if (body.isFixed()) return;
      const h = body.handle;
      if (chunkBodies.has(h)) return;
      const t = body.translation();
      const dx = t.x - center.x;
      const dy = t.y - center.y;
      const dz = t.z - center.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > radius) return;
      const falloff = 1 - dist / radius;
      const inv = dist > 1e-4 ? 1 / dist : 0;
      const nx = dist > 1e-4 ? dx * inv : 0;
      const ny = dist > 1e-4 ? dy * inv : 1;
      const nz = dist > 1e-4 ? dz * inv : 0;
      const j = body.mass() * kick * falloff;
      body.applyImpulse({ x: nx * j, y: (ny + up) * j, z: nz * j }, true);
    });
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

  // ── FPS camera ──────────────────────────────────────────────────
  function updateCamera(dt: number) {
    // Arrow-key look (complements mouse look; works without pointer lock).
    if (keys.has('ArrowLeft')) yaw += cfg.keyLookSpeed * dt;
    if (keys.has('ArrowRight')) yaw -= cfg.keyLookSpeed * dt;
    if (keys.has('ArrowUp')) pitch += cfg.keyLookSpeed * dt;
    if (keys.has('ArrowDown')) pitch -= cfg.keyLookSpeed * dt;
    pitch = Math.min(Math.max(pitch, -1.5), 1.5);

    _e.set(pitch, yaw, 0, 'YXZ');
    camera.quaternion.setFromEuler(_e);

    // Horizontal forward / right from the current orientation.
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
    if (_move.lengthSq() > 1e-6) {
      _move.normalize().multiplyScalar(cfg.walkSpeed * dt);
      camera.position.add(_move);
    }

    // Gravity + jump.
    if (keys.has('Space') && grounded) {
      velY = cfg.jumpSpeed;
      grounded = false;
    }
    velY -= cfg.gravity * dt;
    camera.position.y += velY * dt;
    const floor = floorY + eyeHeight;
    if (camera.position.y <= floor) {
      camera.position.y = floor;
      velY = 0;
      grounded = true;
    }
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
    if (core) updateStickies(core);
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
    if (!cfg.fps || !pointerLocked) return;
    yaw -= e.movementX * cfg.mouseSensitivity;
    pitch -= e.movementY * cfg.mouseSensitivity;
    pitch = Math.min(Math.max(pitch, -1.5), 1.5);
  });

  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (isTyping()) return;
    switch (e.code) {
      case 'Digit1': setMode('ball'); return;
      case 'Digit2': setMode('sticky'); return;
      case 'Digit3': setMode('grenade'); return;
      case 'KeyQ': cycleMode(); return;
      case 'KeyF': detonateAll(); return;
      case 'KeyV': setFpsEnabled(!cfg.fps); return;
    }
    if (cfg.fps) {
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
    setMode,
    detonateAll,
  };
}

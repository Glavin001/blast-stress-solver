/**
 * Rapier raycast-vehicle wrapper for the modern demos.
 *
 * A small, self-contained "car" built on Rapier's `DynamicRayCastVehicleController`
 * (a dynamic chassis rigid body + four raycast wheels with spring suspension). It
 * knows nothing about input, UI or demo modes — `shooter-fps.ts` owns those and
 * drives this through a tiny imperative API:
 *
 *   const car = createVehicle({ world, scene, position, headingY });
 *   car.applyControls({ throttle, brake, steer, handbrake });  // each frame
 *   car.step(dt);        // map controls → wheels, then controller.updateVehicle(dt)
 *   car.syncMeshes();    // chassis + wheel meshes ← physics
 *   …
 *   car.dispose();       // remove controller + body + meshes
 *
 * The controller applies suspension/engine/brake/friction impulses to the chassis;
 * the demo's existing `core.step()` integrates them on the following step (the same
 * "apply now, integrate next step" pattern the blast impulses already use). Because
 * the chassis is a normal dynamic body it collides with the destructible structures
 * and debris, so driving into a building feeds the usual contact-force → stress
 * solver path and the structure actually comes apart.
 *
 * Tuning defaults mirror the Glavin001/kinocat raycast-vehicle adapter, which uses
 * this same `@dimforge/rapier3d-compat` package.
 */
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';

export type VehicleControls = {
  /** -1 (reverse) … +1 (full throttle). */
  throttle: number;
  /** 0 … 1 foot brake (unused by default; handbrake covers most needs). */
  brake: number;
  /** -1 (left) … +1 (right) raw steer input; smoothed internally. */
  steer: number;
  /** Strong brake on every wheel while held. */
  handbrake: boolean;
};

export type VehicleTuning = {
  engineForce: number; // N applied to the drive wheels at full throttle
  brakeForce: number; // N applied to every wheel under (hand)brake
  idleBrake: number; // gentle N when coasting so the car rolls to a stop
  reverseFactor: number; // reverse engine force = engineForce * this
  maxSpeed: number; // m/s soft cap (engine cuts out above this)
  maxSteer: number; // rad lock-to-lock half-angle
  steerRate: number; // rad/s steering slew (toward target / back to centre)
  // Geometry
  wheelBase: number; // half front↔rear spacing (local Z)
  wheelTrack: number; // half left↔right spacing (local X)
  wheelRadius: number;
  suspensionRestLength: number;
  // Chassis box (half-extents) + mass
  chassisHalf: { x: number; y: number; z: number };
  chassisMass: number;
  connectionY: number; // wheel hard-point Y in chassis-local space (below centre)
  // Suspension / friction (Rapier WheelTuning)
  suspensionStiffness: number;
  maxSuspensionForce: number;
  suspensionCompression: number;
  suspensionRelaxation: number;
  maxSuspensionTravel: number;
  frictionSlip: number;
  sideFrictionStiffness: number;
};

export const DEFAULT_TUNING: VehicleTuning = {
  engineForce: 4000,
  brakeForce: 2000,
  idleBrake: 40,
  reverseFactor: 0.5,
  maxSpeed: 32,
  maxSteer: 0.6,
  steerRate: 3.5,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  chassisHalf: { x: 0.9, y: 0.3, z: 1.7 },
  chassisMass: 1200,
  connectionY: -0.25,
  suspensionStiffness: 80,
  maxSuspensionForce: 12000,
  suspensionCompression: 0.83,
  suspensionRelaxation: 20,
  maxSuspensionTravel: 0.2,
  frictionSlip: 1.8,
  sideFrictionStiffness: 1.0,
};

// Local axes: Y up, Z forward. Front wheels steer; rear wheels drive (RWD).
const UP_AXIS = 1; // Y
const FORWARD_AXIS = 2; // Z
const FRONT_WHEELS = [0, 1];
const DRIVE_WHEELS = [2, 3];

export type VehicleHandle = {
  applyControls: (c: VehicleControls) => void;
  /** Map the stored controls onto the wheels and advance the controller. */
  step: (dt: number) => void;
  /** Sync the chassis + wheel meshes from the physics state. */
  syncMeshes: () => void;
  chassisBody: () => RAPIER.RigidBody;
  /** Signed forward speed, km/h, for the HUD. */
  speedKmh: () => number;
  /** Flip the car upright in place (frees a rolled / stuck car). */
  recover: () => void;
  /** Remove + dispose the meshes only (safe after the Rapier world is gone). */
  disposeVisuals: () => void;
  /** Full teardown: remove the controller + body from the (live) world, then meshes. */
  dispose: () => void;
};

export type CreateVehicleOptions = {
  world: RAPIER.World;
  scene: THREE.Scene;
  position: { x: number; y: number; z: number };
  /** Yaw (radians) so local +Z faces the structure: atan2(dirX, dirZ). */
  headingY: number;
  tuning?: Partial<VehicleTuning>;
};

function yawQuat(yaw: number): { x: number; y: number; z: number; w: number } {
  return { x: 0, y: Math.sin(yaw * 0.5), z: 0, w: Math.cos(yaw * 0.5) };
}

// Skip projectiles / thrown charges (and the chassis itself) when the wheels
// raycast for the ground, so the car never "rides" a ball or sticky charge.
function wheelRayFilter(col: RAPIER.Collider): boolean {
  const ud = col.parent()?.userData as
    | { projectile?: boolean; stickyExplosive?: boolean; vehicle?: boolean }
    | undefined;
  return !(ud && (ud.projectile || ud.stickyExplosive || ud.vehicle));
}

export function createVehicle(opts: CreateVehicleOptions): VehicleHandle {
  const { world, scene } = opts;
  const t: VehicleTuning = { ...DEFAULT_TUNING, ...(opts.tuning ?? {}) };

  // ── Chassis rigid body + collider ───────────────────────────────
  const chassis = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(opts.position.x, opts.position.y, opts.position.z)
      .setRotation(yawQuat(opts.headingY))
      .setLinearDamping(0.1)
      .setAngularDamping(0.5)
      .setCcdEnabled(true) // don't tunnel through thin walls at speed
      .setUserData({ vehicle: true }),
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(t.chassisHalf.x, t.chassisHalf.y, t.chassisHalf.z)
      .setMass(t.chassisMass)
      .setFriction(0.7)
      .setRestitution(0)
      // Report contact forces so ramming feeds the structure's damage path,
      // like a projectile impact does.
      .setActiveEvents(RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS)
      .setContactForceEventThreshold(0),
    chassis,
  );

  // ── Raycast vehicle controller + four wheels ────────────────────
  const vehicle = world.createVehicleController(chassis);
  vehicle.indexUpAxis = UP_AXIS;
  // rapier.js names the forward-axis setter `setIndexForwardAxis` (a write-only
  // accessor), so it's assigned, not called.
  vehicle.setIndexForwardAxis = FORWARD_AXIS;

  const directionCs = { x: 0, y: -1, z: 0 };
  const axleCs = { x: -1, y: 0, z: 0 };
  // [x = ±track, z = ±base]; +Z = front. Order: FL, FR, RL, RR.
  const wheelXZ: Array<[number, number]> = [
    [-t.wheelTrack, +t.wheelBase],
    [+t.wheelTrack, +t.wheelBase],
    [-t.wheelTrack, -t.wheelBase],
    [+t.wheelTrack, -t.wheelBase],
  ];
  for (const [x, z] of wheelXZ) {
    vehicle.addWheel(
      { x, y: t.connectionY, z },
      directionCs,
      axleCs,
      t.suspensionRestLength,
      t.wheelRadius,
    );
  }
  for (let i = 0; i < 4; i++) {
    vehicle.setWheelSuspensionStiffness(i, t.suspensionStiffness);
    vehicle.setWheelMaxSuspensionForce(i, t.maxSuspensionForce);
    vehicle.setWheelSuspensionCompression(i, t.suspensionCompression);
    vehicle.setWheelSuspensionRelaxation(i, t.suspensionRelaxation);
    vehicle.setWheelMaxSuspensionTravel(i, t.maxSuspensionTravel);
    vehicle.setWheelFrictionSlip(i, t.frictionSlip);
    vehicle.setWheelSideFrictionStiffness(i, t.sideFrictionStiffness);
  }

  // ── Meshes (chassis box + four wheels) ──────────────────────────
  const chassisMesh = new THREE.Mesh(
    new THREE.BoxGeometry(t.chassisHalf.x * 2, t.chassisHalf.y * 2, t.chassisHalf.z * 2),
    new THREE.MeshStandardMaterial({ color: 0x2f6fed, metalness: 0.5, roughness: 0.4 }),
  );
  chassisMesh.castShadow = true;
  // A little cabin so the car reads as a car and shows its facing.
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(t.chassisHalf.x * 1.5, t.chassisHalf.y * 1.4, t.chassisHalf.z * 0.9),
    new THREE.MeshStandardMaterial({ color: 0x9fc0ff, metalness: 0.3, roughness: 0.3 }),
  );
  cabin.position.set(0, t.chassisHalf.y * 1.1, -t.chassisHalf.z * 0.1);
  cabin.castShadow = true;
  chassisMesh.add(cabin);
  scene.add(chassisMesh);

  const wheelGeo = new THREE.CylinderGeometry(t.wheelRadius, t.wheelRadius, 0.3, 20);
  wheelGeo.rotateZ(Math.PI / 2); // cylinder axis → local X (the axle)
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.8 });
  const wheelMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < 4; i++) {
    const m = new THREE.Mesh(wheelGeo, wheelMat);
    m.castShadow = true;
    scene.add(m);
    wheelMeshes.push(m);
  }

  // ── State + scratch ─────────────────────────────────────────────
  const controls: VehicleControls = { throttle: 0, brake: 0, steer: 0, handbrake: false };
  let steerAngle = 0; // current (smoothed) front-wheel steer, rad
  const _cp = new THREE.Vector3();
  const _cq = new THREE.Quaternion();
  const _wheelLocal = new THREE.Vector3();
  const _wheelQuat = new THREE.Quaternion();
  const _steerQ = new THREE.Quaternion();
  const _spinQ = new THREE.Quaternion();
  const _yAxis = new THREE.Vector3(0, 1, 0);
  const _xAxis = new THREE.Vector3(1, 0, 0);

  function applyControls(c: VehicleControls) {
    controls.throttle = c.throttle;
    controls.brake = c.brake;
    controls.steer = c.steer;
    controls.handbrake = c.handbrake;
  }

  function step(dt: number) {
    // Steering: slew toward the target lock; recentre when released.
    const target = THREE.MathUtils.clamp(controls.steer, -1, 1) * t.maxSteer;
    const maxDelta = t.steerRate * dt;
    steerAngle += THREE.MathUtils.clamp(target - steerAngle, -maxDelta, maxDelta);
    for (const i of FRONT_WHEELS) vehicle.setWheelSteering(i, steerAngle);

    // Engine / brake.
    const speed = vehicle.currentVehicleSpeed(); // signed m/s (+ forward)
    let engine = 0;
    let brake = controls.handbrake ? t.brakeForce : 0;
    if (!controls.handbrake) {
      if (controls.throttle > 0.001) {
        engine = speed < t.maxSpeed ? t.engineForce * controls.throttle : 0;
      } else if (controls.throttle < -0.001) {
        // Brake first if we're still rolling forward, then reverse.
        if (speed > 1.0) brake = t.brakeForce;
        else engine = t.engineForce * t.reverseFactor * controls.throttle;
      } else {
        brake = t.idleBrake; // gentle coast-down
      }
    }
    for (const i of DRIVE_WHEELS) vehicle.setWheelEngineForce(i, engine);
    for (let i = 0; i < 4; i++) vehicle.setWheelBrake(i, brake);

    // Raycast wheels + accumulate impulses on the chassis (integrated next step).
    vehicle.updateVehicle(
      dt,
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      undefined,
      wheelRayFilter,
    );
  }

  function syncMeshes() {
    const p = chassis.translation();
    const r = chassis.rotation();
    _cp.set(p.x, p.y, p.z);
    _cq.set(r.x, r.y, r.z, r.w);
    chassisMesh.position.copy(_cp);
    chassisMesh.quaternion.copy(_cq);

    for (let i = 0; i < 4; i++) {
      const [x, z] = wheelXZ[i];
      const susp = vehicle.wheelSuspensionLength(i);
      const len = susp != null && Number.isFinite(susp) ? susp : t.suspensionRestLength;
      // Wheel centre = hard-point + suspension drop, expressed in chassis space.
      _wheelLocal.set(x, t.connectionY - len, z).applyQuaternion(_cq).add(_cp);
      wheelMeshes[i].position.copy(_wheelLocal);

      _wheelQuat.copy(_cq);
      if (FRONT_WHEELS.includes(i)) {
        _steerQ.setFromAxisAngle(_yAxis, vehicle.wheelSteering(i) ?? steerAngle);
        _wheelQuat.multiply(_steerQ);
      }
      _spinQ.setFromAxisAngle(_xAxis, vehicle.wheelRotation(i) ?? 0);
      _wheelQuat.multiply(_spinQ);
      wheelMeshes[i].quaternion.copy(_wheelQuat);
    }
  }

  function speedKmh(): number {
    return vehicle.currentVehicleSpeed() * 3.6;
  }

  function recover() {
    // Upright at the current heading, lifted a touch, velocities zeroed.
    const r = chassis.rotation();
    const yaw = Math.atan2(2 * (r.w * r.y + r.x * r.z), 1 - 2 * (r.y * r.y + r.z * r.z));
    const p = chassis.translation();
    chassis.setRotation(yawQuat(yaw), true);
    chassis.setTranslation({ x: p.x, y: p.y + 1.0, z: p.z }, true);
    chassis.setLinvel({ x: 0, y: 0, z: 0 }, true);
    chassis.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  function disposeVisuals() {
    scene.remove(chassisMesh);
    for (const m of wheelMeshes) scene.remove(m);
    try {
      chassisMesh.geometry.dispose();
      (chassisMesh.material as THREE.Material).dispose();
      cabin.geometry.dispose();
      (cabin.material as THREE.Material).dispose();
      wheelGeo.dispose();
      wheelMat.dispose();
    } catch {
      /* ignore */
    }
  }

  function dispose() {
    try {
      world.removeVehicleController(vehicle);
    } catch {
      /* ignore */
    }
    try {
      world.removeRigidBody(chassis);
    } catch {
      /* ignore */
    }
    disposeVisuals();
  }

  return {
    applyControls,
    step,
    syncMeshes,
    chassisBody: () => chassis,
    speedKmh,
    recover,
    disposeVisuals,
    dispose,
  };
}

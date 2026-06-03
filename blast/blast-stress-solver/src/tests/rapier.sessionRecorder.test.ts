/**
 * Tests for the per-frame session recorder. Drives the real recorder against a
 * fake destructible core + Rapier world (no DOM, no WASM): verifies the wrapped
 * step/input methods capture inputs, the columnar body trace round-trips through
 * encode → export → decode with exact values, the topology diff emits the right
 * migrate/detach/destroy/bodyRemoved deltas, and that gzip packing works.
 */
import { describe, expect, it } from 'vitest';
import {
  createSessionRecorder,
  decodeSimRecording,
  gzipJson,
  BODY_STRIDE,
  type RecordableCore,
} from '../rapier/sessionRecorder';
import type { ChunkData, Vec3 } from '../rapier/types';

type FakeBody = {
  handle: number;
  fixed: boolean;
  t: Vec3;
  r: { x: number; y: number; z: number; w: number };
  lv: Vec3;
  av: Vec3;
};

function bodyView(b: FakeBody) {
  return {
    handle: b.handle,
    isFixed: () => b.fixed,
    translation: () => b.t,
    rotation: () => b.r,
    linvel: () => b.lv,
    angvel: () => b.av,
  };
}

function makeFakeCore() {
  const bodies: FakeBody[] = [];
  const chunks: ChunkData[] = [];
  let activeBonds = 100;
  const projectiles: { length: number } = { length: 0 };
  const log: { projectiles: unknown[]; forces: unknown[]; gravities: number[] } = {
    projectiles: [],
    forces: [],
    gravities: [],
  };

  const core: RecordableCore = {
    world: {
      forEachRigidBody: (cb) => {
        for (const b of bodies) cb(bodyView(b));
      },
    },
    chunks,
    getActiveBondsCount: () => activeBonds,
    getRigidBodyCount: () => bodies.filter((b) => !b.fixed).length,
    projectiles,
    step: () => {
      /* physics is faked by the test mutating bodies directly */
    },
    enqueueProjectile: (s) => log.projectiles.push(s),
    applyExternalForce: (n, p, f) => log.forces.push({ n, p, f }),
    setGravity: (g) => log.gravities.push(g),
  };

  return {
    core,
    bodies,
    chunks,
    log,
    projectiles,
    setActiveBonds: (n: number) => {
      activeBonds = n;
    },
  };
}

function newChunk(nodeIndex: number, bodyHandle: number | null): ChunkData {
  return {
    nodeIndex,
    size: { x: 1, y: 1, z: 1 },
    isSupport: false,
    baseLocalOffset: { x: 0, y: 0, z: 0 },
    localOffset: { x: 0, y: 0, z: 0 },
    colliderHandle: null,
    bodyHandle,
    active: true,
    detached: false,
  };
}

describe('session recorder', () => {
  it('captures body kinematics columnar and round-trips exactly', () => {
    const fake = makeFakeCore();
    fake.bodies.push(
      { handle: 5, fixed: false, t: { x: 1, y: 2, z: 3 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0.5, y: 0, z: 0 }, av: { x: 0, y: 0.25, z: 0 } },
      { handle: 9, fixed: true, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } }, // ground (skipped)
    );
    fake.chunks.push(newChunk(0, 5));

    const rec = createSessionRecorder();
    rec.attach(fake.core);
    rec.start();

    // Frame 0
    fake.core.step(1 / 60);
    // Frame 1 — move the dynamic body.
    fake.bodies[0].t = { x: 4, y: 5, z: 6 };
    fake.bodies[0].lv = { x: 2, y: -1, z: 0 };
    fake.core.step(1 / 60);
    rec.stop();

    const data = rec.export();
    expect(data).not.toBeNull();
    const dec = decodeSimRecording(data!);

    expect(dec.durationFrames).toBe(2);
    // Fixed body excluded → 1 row per frame.
    expect(Array.from(dec.columns.bodyCount)).toEqual([1, 1]);
    expect(dec.bodies.length).toBe(2 * BODY_STRIDE);

    const f1 = dec.bodyInFrame(1, 5);
    expect(f1).not.toBeNull();
    expect(f1![0]).toBe(5); // handle
    expect(f1![1]).toBeCloseTo(4); // px
    expect(f1![2]).toBeCloseTo(5);
    expect(f1![3]).toBeCloseTo(6);
    expect(f1![8]).toBeCloseTo(2); // lvx
    expect(f1![9]).toBeCloseTo(-1); // lvy

    // Frame 0 still has the original position.
    const f0 = dec.bodyInFrame(0, 5)!;
    expect(f0![1]).toBeCloseTo(1);
  });

  it('logs projectile / force / gravity input events with frame + sim time', () => {
    const fake = makeFakeCore();
    fake.bodies.push({ handle: 1, fixed: false, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } });
    fake.chunks.push(newChunk(0, 1));

    const rec = createSessionRecorder();
    rec.attach(fake.core);
    rec.start();

    fake.core.step(0.1); // frame 0 → simTime 0.1
    fake.core.enqueueProjectile({ position: { x: 1, y: 2, z: 3 }, velocity: { x: 4, y: 5, z: 6 }, radius: 0.5, mass: 100 });
    fake.core.applyExternalForce(0, { x: 0, y: 1, z: 0 }, { x: 10, y: 0, z: 0 });
    fake.core.setGravity(-5);
    fake.core.step(0.1); // frame 1

    rec.stop();

    // Inputs were forwarded to the real core methods.
    expect(fake.log.projectiles).toHaveLength(1);
    expect(fake.log.forces).toHaveLength(1);
    expect(fake.log.gravities).toEqual([-5]);

    const data = rec.export()!;
    const proj = data.events.find((e) => e.type === 'projectile');
    const force = data.events.find((e) => e.type === 'force');
    const grav = data.events.find((e) => e.type === 'gravity');
    expect(proj).toMatchObject({ f: 1, type: 'projectile' });
    expect((proj as { t: number }).t).toBeCloseTo(0.1);
    expect(force).toMatchObject({ f: 1, node: 0, type: 'force' });
    expect(grav).toMatchObject({ f: 1, value: -5, type: 'gravity' });
  });

  it('emits topology deltas (migrate / detach / destroy / bodyRemoved)', () => {
    const fake = makeFakeCore();
    fake.bodies.push(
      { handle: 1, fixed: false, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } },
      { handle: 2, fixed: false, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } },
    );
    fake.chunks.push(newChunk(0, 1), newChunk(1, 1)); // both on body 1 initially

    const rec = createSessionRecorder();
    rec.attach(fake.core);
    rec.start();

    fake.core.step(1 / 60); // frame 0 — baseline confirmed

    // Chunk 1 migrates to a new body 2 and detaches; chunk 0 stays.
    fake.chunks[1].bodyHandle = 2;
    fake.chunks[1].detached = true;
    fake.core.step(1 / 60); // frame 1

    // Chunk 0's body 1 disappears (removed), chunk 0 destroyed.
    fake.chunks[0].destroyed = true;
    fake.chunks[0].bodyHandle = null;
    fake.core.step(1 / 60); // frame 2

    rec.stop();
    const data = rec.export()!;

    const migrate = data.events.find((e) => e.type === 'migrate');
    expect(migrate).toMatchObject({ node: 1, from: 1, to: 2, f: 1 });
    const detach = data.events.find((e) => e.type === 'detach');
    expect(detach).toMatchObject({ node: 1, f: 1 });
    const destroy = data.events.find((e) => e.type === 'destroy');
    expect(destroy).toMatchObject({ node: 0, f: 2 });
    const removed = data.events.find((e) => e.type === 'bodyRemoved');
    expect(removed).toMatchObject({ body: 1 });
  });

  it('restores wrapped core methods on detach', () => {
    const fake = makeFakeCore();
    const origStep = fake.core.step;
    const origEnqueue = fake.core.enqueueProjectile;
    const rec = createSessionRecorder();
    rec.attach(fake.core);
    expect(fake.core.step).not.toBe(origStep); // wrapped
    rec.detach();
    expect(fake.core.step).toBe(origStep); // restored
    expect(fake.core.enqueueProjectile).toBe(origEnqueue);
  });

  it('auto-stops at maxFrames', () => {
    const fake = makeFakeCore();
    fake.bodies.push({ handle: 1, fixed: false, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } });
    fake.chunks.push(newChunk(0, 1));
    let autoStopped = false;
    const rec = createSessionRecorder({ maxFrames: 3, onAutoStop: () => (autoStopped = true) });
    rec.attach(fake.core);
    rec.start();
    for (let i = 0; i < 10; i += 1) fake.core.step(1 / 60);
    expect(autoStopped).toBe(true);
    expect(rec.isRecording()).toBe(false);
    expect(rec.frameCount()).toBe(3);
  });

  it('gzip packs the bundle to a smaller blob', async () => {
    const fake = makeFakeCore();
    // A handful of smoothly-moving bodies compress well.
    for (let i = 0; i < 20; i += 1) {
      fake.bodies.push({ handle: i, fixed: false, t: { x: i, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 }, lv: { x: 0, y: 0, z: 0 }, av: { x: 0, y: 0, z: 0 } });
      fake.chunks.push(newChunk(i, i));
    }
    const rec = createSessionRecorder();
    rec.attach(fake.core);
    rec.start();
    for (let f = 0; f < 30; f += 1) {
      for (const b of fake.bodies) b.t = { x: b.t.x, y: b.t.y + 0.01, z: b.t.z };
      fake.core.step(1 / 60);
    }
    rec.stop();
    const data = rec.export()!;
    const json = JSON.stringify(data);
    const { blob, gzipped } = await gzipJson(data);
    if (gzipped) {
      expect(blob.size).toBeLessThan(json.length);
    } else {
      // Environment without CompressionStream — fallback returns raw JSON blob.
      expect(blob.size).toBe(json.length);
    }
  });
});

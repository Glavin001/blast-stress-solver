/**
 * Shared-memory pose channel codec (pure logic — no WASM, no workers).
 *
 * Pins the publish → mirror round trip the worker-physics architecture rides on:
 * poses/flags/body kinds/projectiles survive intact, mirrored chunk objects keep their
 * identity (the renderer's batched-sync cache keys on it), projectile mesh references
 * survive across frames by id, and the frame counter gates re-mirroring.
 */
import { describe, it, expect } from 'vitest';
import {
  poseChannelByteLength,
  viewPoseChannel,
  publishPoseFrame,
  mirrorPoseFrame,
  NO_BODY,
  type MirrorChunk,
  type MirrorProjectile,
} from '../rapier/poseChannel';

function fakeCore() {
  const bodies = new Map<number, { fixed: boolean; t: { x: number; y: number; z: number }; r: { x: number; y: number; z: number; w: number } }>();
  bodies.set(1, { fixed: true, t: { x: 0, y: 0, z: 0 }, r: { x: 0, y: 0, z: 0, w: 1 } });
  bodies.set(9, { fixed: false, t: { x: 5, y: 6, z: 7 }, r: { x: 0, y: 1, z: 0, w: 0 } });
  return {
    chunks: [
      { active: true, destroyed: false, bodyHandle: 1, worldPosition: { x: 1, y: 2, z: 3 }, worldQuaternion: { x: 0, y: 0, z: 0, w: 1 } },
      { active: true, destroyed: false, bodyHandle: 9, worldPosition: { x: -4, y: 0.5, z: 8 }, worldQuaternion: { x: 0, y: 0.6, z: 0, w: 0.8 } },
      { active: false, destroyed: true, bodyHandle: null, worldPosition: null, worldQuaternion: null },
    ],
    projectiles: [{ bodyHandle: 9, radius: 0.75, type: 'ball' }],
    world: {
      getRigidBody(h: number) {
        const b = bodies.get(h);
        if (!b) return null;
        return {
          isFixed: () => b.fixed,
          translation: () => b.t,
          rotation: () => b.r,
        };
      },
    },
    bodies,
  };
}

function freshMirror(n: number): MirrorChunk[] {
  return Array.from({ length: n }, (_, i) => ({
    nodeIndex: i,
    active: false,
    destroyed: false,
    detached: false,
    bodyHandle: null,
    worldPosition: null,
    worldQuaternion: null,
    baseLocalOffset: { x: 0, y: 0, z: 0 },
    localOffset: { x: 0, y: 0, z: 0 },
  }));
}

describe('pose channel codec', () => {
  it('round-trips chunk poses, flags, bodies, kinds and projectiles', () => {
    const core = fakeCore();
    const buf = new ArrayBuffer(poseChannelByteLength(8, 4));
    const views = viewPoseChannel(buf, 8, 4);

    const frame = publishPoseFrame(views, core as any);
    expect(frame).toBe(1);

    const chunks = freshMirror(3);
    const projectiles: MirrorProjectile[] = [];
    const kinds = new Map<number, number>();
    const consumed = mirrorPoseFrame(views, 0, chunks, projectiles, kinds, 12.5);
    expect(consumed).toBe(1);

    expect(chunks[0].active).toBe(true);
    expect(chunks[0].bodyHandle).toBe(1);
    expect(chunks[0].worldPosition).toEqual({ x: 1, y: 2, z: 3 });
    expect(kinds.get(1)).toBe(0); // fixed
    expect(kinds.get(9)).toBe(2); // dynamic

    expect(chunks[1].worldQuaternion!.y).toBeCloseTo(0.6, 6);
    expect(chunks[2].active).toBe(false);
    expect(chunks[2].destroyed).toBe(true);
    expect(chunks[2].bodyHandle).toBe(null);
    expect(chunks[2].worldPosition).toBe(null); // never had a pose

    expect(projectiles).toHaveLength(1);
    expect(projectiles[0].bodyHandle).toBe(9);
    expect(projectiles[0].px).toBe(5);
    expect(projectiles[0].radius).toBeCloseTo(0.75, 6);
    expect(projectiles[0].spawnTime).toBe(12.5);
  });

  it('mirrors in place: chunk object and pose identities are stable across frames', () => {
    const core = fakeCore();
    const buf = new ArrayBuffer(poseChannelByteLength(8, 4));
    const views = viewPoseChannel(buf, 8, 4);
    const chunks = freshMirror(3);
    const projectiles: MirrorProjectile[] = [];
    const kinds = new Map<number, number>();

    publishPoseFrame(views, core as any);
    mirrorPoseFrame(views, 0, chunks, projectiles, kinds, 0);
    const poseRef = chunks[0].worldPosition;
    const chunkRef = chunks[0];

    core.chunks[0].worldPosition!.x = 42;
    publishPoseFrame(views, core as any);
    mirrorPoseFrame(views, 1, chunks, projectiles, kinds, 0);

    expect(chunks[0]).toBe(chunkRef);
    expect(chunks[0].worldPosition).toBe(poseRef); // mutated, not replaced
    expect(poseRef!.x).toBe(42);
  });

  it('returns -1 (mirror untouched) when no new frame was published', () => {
    const core = fakeCore();
    const buf = new ArrayBuffer(poseChannelByteLength(8, 4));
    const views = viewPoseChannel(buf, 8, 4);
    const chunks = freshMirror(3);
    const kinds = new Map<number, number>();
    const frame = publishPoseFrame(views, core as any);
    expect(mirrorPoseFrame(views, frame, chunks, [], kinds, 0)).toBe(-1);
    expect(chunks[0].active).toBe(false); // untouched
  });

  it('keeps a surviving projectile entry (and its mesh reference) across frames, drops despawned ones', () => {
    const core = fakeCore();
    const buf = new ArrayBuffer(poseChannelByteLength(8, 4));
    const views = viewPoseChannel(buf, 8, 4);
    const chunks = freshMirror(3);
    const projectiles: MirrorProjectile[] = [];
    const kinds = new Map<number, number>();

    publishPoseFrame(views, core as any);
    mirrorPoseFrame(views, 0, chunks, projectiles, kinds, 1);
    const entry = projectiles[0];
    (entry as { mesh?: unknown }).mesh = { fake: true };

    publishPoseFrame(views, core as any);
    mirrorPoseFrame(views, 1, chunks, projectiles, kinds, 2);
    expect(projectiles[0]).toBe(entry); // same entry → mesh survives
    expect((projectiles[0] as { mesh?: unknown }).mesh).toEqual({ fake: true });

    core.projectiles.length = 0; // despawn
    publishPoseFrame(views, core as any);
    mirrorPoseFrame(views, 2, chunks, projectiles, kinds, 3);
    expect(projectiles).toHaveLength(0);
  });

  it('treats NO_BODY as null and clamps to capacity', () => {
    const core = fakeCore();
    const buf = new ArrayBuffer(poseChannelByteLength(2, 1)); // capacity below counts
    const views = viewPoseChannel(buf, 2, 1);
    publishPoseFrame(views, core as any);
    const chunks = freshMirror(2);
    const kinds = new Map<number, number>();
    mirrorPoseFrame(views, 0, chunks, [], kinds, 0);
    expect(views.header[1]).toBe(2); // clamped chunk count
    expect(views.bodies[0]).not.toBe(NO_BODY);
  });
});

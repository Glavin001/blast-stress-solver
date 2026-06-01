/**
 * Invariants for the damage system (`DestructibleDamageSystem`): health is monotonically
 * non-increasing, destruction is irreversible (no healing, no resurrection), and support
 * chunks are never destroyed. None of this was tested before. Pure logic (no WASM).
 */
import { describe, it, expect } from 'vitest';
import { DestructibleDamageSystem } from '../../dist/rapier.js';

function makeSystem() {
  // 3 chunks: two normal, one support. volume 1 + strengthPerVolume 10000 => maxHealth 10000.
  const chunks: any[] = [{}, {}, { isSupport: true }];
  const scenario: any = {
    nodes: [
      { centroid: { x: 0, y: 0, z: 0 }, mass: 1, volume: 1 },
      { centroid: { x: 1, y: 0, z: 0 }, mass: 1, volume: 1 },
      { centroid: { x: 2, y: 0, z: 0 }, mass: 0, volume: 1 },
    ],
  };
  return new DestructibleDamageSystem({
    chunks,
    scenario,
    materialScale: 1,
    options: { enabled: true, strengthPerVolume: 10000 },
  });
}

describe('DestructibleDamageSystem invariants', () => {
  it('health is monotonically non-increasing across damage + ticks', () => {
    const sys = makeSystem();
    expect(sys.getHealth(0)!.health).toBe(10000);
    sys.applyDirect(0, 3000);
    sys.tick(1 / 60);
    expect(sys.getHealth(0)!.health).toBe(7000);
    sys.applyDirect(0, 2000);
    sys.tick(1 / 60);
    expect(sys.getHealth(0)!.health).toBe(5000);
  });

  it('cannot be healed (negative damage is clamped to zero)', () => {
    const sys = makeSystem();
    sys.applyDirect(1, -99999);
    sys.tick(1 / 60);
    expect(sys.getHealth(1)!.health).toBe(10000);
  });

  it('destroys a chunk when health reaches zero', () => {
    const sys = makeSystem();
    sys.applyDirect(0, 99999);
    const destroyed = sys.tick(1 / 60);
    expect(destroyed).toContain(0);
    const h = sys.getHealth(0)!;
    expect(h.destroyed).toBe(true);
    expect(h.health).toBe(0);
  });

  it('destruction is irreversible — further damage (or healing) is a no-op', () => {
    const sys = makeSystem();
    sys.applyDirect(0, 99999);
    sys.tick(1 / 60);
    // Try to heal and to re-damage a destroyed chunk.
    sys.applyDirect(0, -100000);
    sys.applyDirect(0, 5000);
    const before = sys.getHealth(0)!;
    sys.tick(1 / 60);
    const after = sys.getHealth(0)!;
    expect(after.destroyed).toBe(true);
    expect(after.health).toBe(0);
    expect(after.health).toBe(before.health);
  });

  it('support chunks are never destroyed, even at zero health', () => {
    const sys = makeSystem();
    sys.applyDirect(2, 99999);
    const destroyed = sys.tick(1 / 60);
    expect(destroyed).not.toContain(2);
    expect(sys.getHealth(2)!.destroyed).toBe(false);
  });

  it('previewTick does not mutate state', () => {
    const sys = makeSystem();
    sys.applyDirect(1, 99999);
    const would = sys.previewTick(1 / 60);
    expect(would).toContain(1); // would be destroyed
    const h = sys.getHealth(1)!;
    expect(h.destroyed).toBe(false); // but state is untouched
    expect(h.health).toBe(10000);
  });
});

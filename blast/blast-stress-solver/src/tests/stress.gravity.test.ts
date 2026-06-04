import { describe, it, expect } from 'vitest';
import type * as Runtime from '..';
import {
  createColumnNodes,
  createColumnBond,
  columnStressSettings,
  gravityMagnitude
} from './gravityFixtures';

async function importRuntime(): Promise<typeof Runtime> {
  return (await import('../../dist/index.js')) as typeof Runtime;
}

describe('ExtStressSolver addActorGravity', () => {
  const createSolver = async (orientation: 'vertical' | 'horizontal') => {
    const { loadStressSolver } = await importRuntime();
    const rt = await loadStressSolver();
    return rt.createExtSolver({
      nodes: createColumnNodes(orientation),
      bonds: [createColumnBond(orientation)],
      settings: columnStressSettings
    });
  };

  it('distinguishes gravity directions based on actor rotation', async () => {
    const worldGravity = { x: 0, y: -gravityMagnitude, z: 0 };

    // Upright column: global gravity fractures the bond
    const verticalSolver = await createSolver('vertical');
    verticalSolver.addGravity(worldGravity);
    verticalSolver.update();
    const verticalFracture = verticalSolver.generateFractureCommands();
    expect(verticalFracture.fractures.length).toBeGreaterThan(0);
    verticalSolver.destroy();

    // Horizontal column without any per-actor gravity: remains intact
    const horizontalSolver = await createSolver('horizontal');
    horizontalSolver.update();
    const horizontalFracture = horizontalSolver.generateFractureCommands();
    expect(horizontalFracture.fractures.length).toBe(0);
    horizontalSolver.destroy();

    // Horizontal column with gravity transformed into local space using addActorGravity
    const rotatedSolver = await createSolver('horizontal');
    const [actor] = rotatedSolver.actors();
    expect(actor).toBeDefined();
    rotatedSolver.addActorGravity(actor!.actorIndex, { x: -gravityMagnitude, y: 0, z: 0 });
    rotatedSolver.update();
    const rotatedFracture = rotatedSolver.generateFractureCommands();
    expect(rotatedFracture.fractures.length).toBeGreaterThan(0);
    rotatedSolver.destroy();
  });

  it('addAllActorGravity rotates world gravity into each actor frame in one call', async () => {
    // The runtime built here exposes the batched entry point.
    const probe = await createSolver('horizontal');
    expect(probe.supportsBatchedGravity()).toBe(true);
    probe.destroy();

    // Upright column: pass world gravity with the identity rotation — the
    // batched call applies it unrotated, fracturing the axial bond.
    const verticalSolver = await createSolver('vertical');
    const identity = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1]);
    const applied = verticalSolver.addAllActorGravity({ x: 0, y: -gravityMagnitude, z: 0 }, identity, 2);
    expect(applied).toBeGreaterThan(0);
    verticalSolver.update();
    expect(verticalSolver.generateFractureCommands().fractures.length).toBeGreaterThan(0);
    verticalSolver.destroy();

    // Horizontal column: a 90° rotation about Z maps world -Y gravity into the
    // actor's local -X axis, fracturing the (local-X) bond — the same outcome
    // as the explicit per-actor addActorGravity case above, but computed in C++.
    const rotatedSolver = await createSolver('horizontal');
    const [actor] = rotatedSolver.actors();
    expect(actor).toBeDefined();
    // Quaternion for -90° about Z: (x,y,z,w) = (0, 0, -sin45, cos45).
    const h = Math.SQRT1_2;
    const rotations = new Float32Array(4 * (actor!.actorIndex + 1));
    rotations.set([0, 0, -h, h], actor!.actorIndex * 4);
    rotatedSolver.addAllActorGravity({ x: 0, y: -gravityMagnitude, z: 0 }, rotations, actor!.actorIndex + 1);
    rotatedSolver.update();
    expect(rotatedSolver.generateFractureCommands().fractures.length).toBeGreaterThan(0);
    rotatedSolver.destroy();
  });
});


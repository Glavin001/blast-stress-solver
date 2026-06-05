/**
 * CCD policy: continuous collision detection must be enabled for the PROJECTILE only, not for
 * every fracture fragment.
 *
 * Fragment CCD (added 2026-04-07 in "fix: batch split planning…", on by default until now) ran
 * swept collision tests on every debris body and, near the pile, clamped fast chunks to their
 * first predicted contact each frame — the "big chunks float/lag while small debris falls
 * normally" report, plus a large perf cost. The projectile still needs CCD so it can't tunnel
 * through the structure. These tests pin the default (projectile CCD on, fragment CCD off) and
 * verify the opt-in still works. Requires the WASM build.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import RAPIER from '@dimforge/rapier3d-compat';

const here = dirname(fileURLToPath(import.meta.url));
const runtimeAvailable = existsSync(resolve(here, '../../dist/stress_solver.wasm'));

let buildDestructibleCore: (opts: any) => Promise<any>;
beforeAll(async () => {
  if (!runtimeAvailable) return;
  await RAPIER.init();
  buildDestructibleCore = (await import('../../dist/rapier.js')).buildDestructibleCore;
});

// 4×3 grid anchored along its bottom row with weak vertical bonds -> cascades into many dynamic
// fragment bodies under gravity.
function cascadingGrid() {
  const rows = 4, cols = 3;
  const nodes: any[] = [], bonds: any[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    nodes.push({ centroid: { x: c, y: r + 0.5, z: 0 }, mass: r === 0 ? 0 : 5, volume: 1 });
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols - 1; c++) {
    const n = r * cols + c;
    bonds.push({ node0: n, node1: n + 1, centroid: { x: c + 0.5, y: r + 0.5, z: 0 }, normal: { x: 1, y: 0, z: 0 }, area: 2 });
  }
  for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols; c++) {
    const n = r * cols + c;
    bonds.push({ node0: n, node1: n + cols, centroid: { x: c, y: r + 1, z: 0 }, normal: { x: 0, y: 1, z: 0 }, area: 0.01 });
  }
  return { nodes, bonds } as any;
}

function classifyCcd(core: any) {
  let projectileCcd: boolean | null = null;
  const fragmentCcd: boolean[] = [];
  core.world.forEachRigidBody((b: any) => {
    if (typeof b.isCcdEnabled !== 'function' || b.isFixed()) return;
    if (b.handle === core.rootBodyHandle || b.handle === core.groundBodyHandle) return;
    const ud = b.userData as { projectile?: boolean } | undefined;
    if (ud && ud.projectile) projectileCcd = b.isCcdEnabled();
    else fragmentCcd.push(b.isCcdEnabled());
  });
  return { projectileCcd, fragmentCcd };
}

async function runCascadeWithProjectile(overrides: Record<string, unknown>) {
  const core = await buildDestructibleCore({
    scenario: cascadingGrid(), gravity: -20, materialScale: 0.001,
    snapshotMode: 'perBody', debrisCleanup: { mode: 'off' }, skipSingleBodies: false,
    ...overrides,
  });
  core.setFracturePolicy?.({ idleSkip: false });
  core.enqueueProjectile({ position: { x: 1, y: 6, z: 4 }, velocity: { x: 0, y: 0, z: -30 }, radius: 0.3, mass: 500, ttl: 5 });
  for (let i = 0; i < 80; i++) core.step(1 / 60);
  const result = classifyCcd(core);
  core.dispose();
  return result;
}

describe.skipIf(!runtimeAvailable)('CCD policy (projectile-only by default)', () => {
  it('by default: projectile has CCD, fragments do NOT', async () => {
    const { projectileCcd, fragmentCcd } = await runCascadeWithProjectile({});
    expect(projectileCcd).toBe(true); // projectile must keep CCD (no tunneling through the structure)
    expect(fragmentCcd.length).toBeGreaterThan(0); // the grid actually produced dynamic fragments
    expect(fragmentCcd.every((c) => c === false)).toBe(true); // ...none of which have CCD
  });

  it('opt-in: fractureBodyCcdEnabled:true re-enables CCD on fragments', async () => {
    const { projectileCcd, fragmentCcd } = await runCascadeWithProjectile({ fractureBodyCcdEnabled: true });
    expect(projectileCcd).toBe(true);
    expect(fragmentCcd.length).toBeGreaterThan(0);
    expect(fragmentCcd.every((c) => c === true)).toBe(true);
  });

  it('projectile CCD can be turned off independently', async () => {
    const { projectileCcd } = await runCascadeWithProjectile({ projectileCcdEnabled: false });
    expect(projectileCcd).toBe(false);
  });
});

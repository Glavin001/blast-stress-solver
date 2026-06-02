/**
 * Shared invariant spec for the destruction harness — the single source of truth
 * for the tolerances and continuity assertions used by the JS test suite, and the
 * documented mirror for the Rust suite
 * (`blast-stress-solver-rs/tests/kinematic_invariants_test.rs`).
 *
 * The headline invariant is **point-velocity continuity across a split**: a faithful
 * rigid fracture must preserve every migrated chunk's world-space point velocity.
 * The JS library already satisfies this (it transfers parent motion using Rapier's
 * real centre of mass); the Rust port does not for offset-COM fragments. Keeping the
 * spec here means both languages assert the same thing with the same numbers.
 */

/** Tolerance table — keep in sync with the Rust constants of the same name. */
export const TOL = {
  /** ‖child.velocityAtPoint(p) − parent.velocityAtPoint(p)‖ across a split (m/s). */
  pointVelocityContinuity: 1e-3,
  /** Chunk world-position continuity across a split (m). */
  worldPositionContinuity: 1e-3,
  /** Body origin translation continuity across a restore/migration. */
  translation: 1e-6,
  /** Body orientation continuity. */
  rotation: 1e-6,
  /** Body angular-velocity continuity. */
  angularVelocity: 1e-6,
  /** Relative tolerance for conserved scalar quantities (mass, etc.). */
  massRelative: 1e-4,
} as const;

export type Vec3 = { x: number; y: number; z: number };

export type SplitContinuityRecord = {
  phase: 'migration' | 'restore';
  sourceBodyHandle: number;
  targetBodyHandle: number;
  nodeIndices: number[];
  sourceBodyIsFixed: boolean;
  targetBodyIsFixed: boolean;
  translationError: number;
  rotationError: number;
  linearVelocityError: number;
  angularVelocityError: number;
  maxChunkWorldPositionError: number;
  maxChunkPointVelocityError: number;
};

/** The observability surface the destructible core exposes for tests. */
export type ContinuityObservableCore = {
  step: (dt?: number) => void;
  __debugSplitContinuityLog?: SplitContinuityRecord[];
  __clearDebugSplitContinuityLog?: () => void;
};

export function stepN(core: { step: (dt?: number) => void }, count: number, dt = 1 / 60): void {
  for (let index = 0; index < count; index += 1) {
    core.step(dt);
  }
}

/** All recorded continuity samples (both migration and restore phases). */
export function getContinuityRecords(core: ContinuityObservableCore): SplitContinuityRecord[] {
  return core.__debugSplitContinuityLog ?? [];
}

/** Continuity records for genuine dynamic→dynamic body transfers (distinct handles). */
export function getDynamicRecords(
  core: ContinuityObservableCore,
  phase?: 'migration' | 'restore',
): SplitContinuityRecord[] {
  return getContinuityRecords(core).filter(
    (record) =>
      (phase === undefined || record.phase === phase) &&
      !record.sourceBodyIsFixed &&
      !record.targetBodyIsFixed &&
      record.sourceBodyHandle !== record.targetBodyHandle &&
      record.nodeIndices.length > 0,
  );
}

/**
 * Assert the per-record continuity invariants. Mirrors the assertions in
 * `rapier.resim-continuity.test.ts` and the Rust kinematic invariant test.
 *
 * Note: raw COM `linearVelocityError` is intentionally only checked for finiteness
 * (a child's COM legitimately differs from its parent's); the *physically* correct,
 * un-cheatable invariant is point-velocity continuity at the migrated chunks.
 */
export function assertContinuity(
  expect: (value: unknown, message?: string) => any,
  records: SplitContinuityRecord[],
): void {
  expect(records.length).toBeGreaterThan(0);
  for (const record of records) {
    const where = `phase=${record.phase} src=${record.sourceBodyHandle} dst=${record.targetBodyHandle} nodes=[${record.nodeIndices.join(',')}]`;
    expect(Number.isFinite(record.linearVelocityError), where).toBe(true);
    expect(record.translationError, where).toBeLessThan(TOL.translation);
    expect(record.rotationError, where).toBeLessThan(TOL.rotation);
    expect(record.angularVelocityError, where).toBeLessThan(TOL.angularVelocity);
    expect(record.maxChunkWorldPositionError, where).toBeLessThan(TOL.worldPositionContinuity);
    expect(record.maxChunkPointVelocityError, where).toBeLessThan(TOL.pointVelocityContinuity);
  }
}

/** Assert that every continuity record's measured magnitudes are finite (no NaN/Inf). */
export function assertFiniteRecords(
  expect: (value: unknown) => any,
  records: SplitContinuityRecord[],
): void {
  for (const record of records) {
    expect(Number.isFinite(record.maxChunkPointVelocityError)).toBe(true);
    expect(Number.isFinite(record.maxChunkWorldPositionError)).toBe(true);
    expect(Number.isFinite(record.linearVelocityError)).toBe(true);
    expect(Number.isFinite(record.angularVelocityError)).toBe(true);
  }
}

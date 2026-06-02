/**
 * Lightweight, read-only destruction-metrics recorder.
 *
 * Samples per frame: active bonds, bonds broken this frame and cumulatively, rigid
 * body count, and the center-of-mass height of dynamic chunks (a topple/collapse
 * signal). Used by the web demo HUD, the JS sweep, and tests to answer "how many
 * bonds are breaking over time" without touching the core.
 */
import type { ChunkData } from './types';

/** Minimal structural view of the destructible core this recorder reads from. */
export type MetricsCoreLike = {
  getActiveBondsCount: () => number;
  getRigidBodyCount: () => number;
  chunks: ChunkData[];
};

export type DestructionSample = {
  frame: number;
  activeBonds: number;
  bondsBrokenThisFrame: number;
  bondsBrokenCumulative: number;
  rigidBodies: number;
  /** Average Y of active, non-support chunks (drops as the structure collapses). */
  comHeight: number;
};

export type BondBreakRecorder = {
  /** Record a sample for the given frame index and return it. */
  sample: (frame: number) => DestructionSample;
  history: () => DestructionSample[];
  /** Total bonds broken since the recorder was created. */
  totalBondsBroken: () => number;
  /** Peak rigid-body count seen across all samples. */
  peakRigidBodies: () => number;
  /** Net change in COM height from the first to the latest sample (negative = collapse). */
  comDrop: () => number;
};

function dynamicComHeight(chunks: ChunkData[]): number {
  let sum = 0;
  let count = 0;
  for (const c of chunks) {
    if (!c.active || c.isSupport) continue;
    const pos = c.worldPosition ?? c.baseWorldPosition ?? c.baseLocalOffset ?? c.localOffset;
    if (pos) {
      sum += pos.y;
      count += 1;
    }
  }
  return count > 0 ? sum / count : 0;
}

export function createBondBreakRecorder(core: MetricsCoreLike): BondBreakRecorder {
  const initialBonds = core.getActiveBondsCount();
  let prevActive = initialBonds;
  let cumulative = 0;
  let peakBodies = 0;
  let firstCom: number | null = null;
  let lastCom = 0;
  const samples: DestructionSample[] = [];

  return {
    sample(frame: number): DestructionSample {
      const activeBonds = core.getActiveBondsCount();
      const broken = Math.max(0, prevActive - activeBonds);
      prevActive = activeBonds;
      cumulative += broken;
      const rigidBodies = core.getRigidBodyCount();
      peakBodies = Math.max(peakBodies, rigidBodies);
      const comHeight = dynamicComHeight(core.chunks);
      if (firstCom === null) firstCom = comHeight;
      lastCom = comHeight;
      const s: DestructionSample = {
        frame,
        activeBonds,
        bondsBrokenThisFrame: broken,
        bondsBrokenCumulative: cumulative,
        rigidBodies,
        comHeight,
      };
      samples.push(s);
      return s;
    },
    history: () => samples,
    totalBondsBroken: () => cumulative,
    peakRigidBodies: () => peakBodies,
    comDrop: () => (firstCom === null ? 0 : lastCom - firstCom),
  };
}

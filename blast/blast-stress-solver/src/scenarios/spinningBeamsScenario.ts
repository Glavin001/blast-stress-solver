/**
 * Spinning-beams scenario — a showcase for centrifugal stress.
 *
 * Produces a set of free-floating (no support / no anchor) segmented beams, spaced apart so they
 * don't touch. Each beam is one connected component, so the core spawns it as a single dynamic
 * rigid body / actor. When the simulation gives a beam a hard spin (and gravity is zero), every
 * segment needs an inward centripetal acceleration to stay on its circular path — which the solver
 * supplies as compression in the beam's radial bonds. With centrifugal acceleration enabled the
 * beams shatter from the inside out; with it disabled they spin forever intact. That on/off
 * contrast is the whole point of this scene.
 *
 * The scene is deliberately anchor-free and gravity-agnostic: callers set `gravity: 0` and spin the
 * bodies themselves (see `centrifugal-spinner` demo / `centrifugal.showcase` test).
 */
import type { ScenarioBond, ScenarioDesc, ScenarioNode, Vec3 } from '../rapier/types';

export interface SpinningBeamsOptions {
  /** Number of independent beams. */
  beams?: number;
  /** Segments (nodes) per beam, laid out along local X. */
  segments?: number;
  /** Edge length of each cubic segment (m). Also used as the render/collider size via `spacing`. */
  segmentSize?: number;
  /** Empty gap between adjacent beams along Z (m). */
  gap?: number;
  /** Mass of each segment (kg). */
  massPerSegment?: number;
  /** Multiplier on each bond's contact area. Smaller area ⇒ higher stress ⇒ easier to break. */
  areaScale?: number;
  /**
   * Height (m) to float the beams above the origin. The core always spawns an invisible ground
   * plane at y=0; since the beams spin about the vertical axis they would otherwise sweep through
   * it like a blade and generate stray contacts. Lifting them clear keeps the only load
   * centrifugal.
   */
  height?: number;
}

export const DEFAULT_SPINNING_BEAMS_OPTIONS: Required<SpinningBeamsOptions> = {
  beams: 4,
  segments: 7,
  segmentSize: 0.6,
  gap: 1.4,
  massPerSegment: 1.0,
  areaScale: 1.0,
  height: 6.0,
};

/**
 * Build the spinning-beams scenario. Every node is dynamic (no support node), so each beam is a
 * free body that can be spun and fractured purely by centrifugal load.
 */
export function buildSpinningBeamsScenario(opts: SpinningBeamsOptions = {}): ScenarioDesc {
  const { beams, segments, segmentSize, gap, massPerSegment, areaScale, height } = {
    ...DEFAULT_SPINNING_BEAMS_OPTIONS,
    ...opts,
  };

  const nodes: ScenarioNode[] = [];
  const bonds: ScenarioBond[] = [];

  const volume = segmentSize * segmentSize * segmentSize;
  const area = segmentSize * segmentSize * areaScale; // cross-section shared by adjacent segments
  const beamPitch = segmentSize + gap;
  const beamCenter = (beams - 1) / 2;
  const segCenter = (segments - 1) / 2;

  for (let b = 0; b < beams; b++) {
    const z = (b - beamCenter) * beamPitch;
    const base = nodes.length;

    for (let i = 0; i < segments; i++) {
      const x = (i - segCenter) * segmentSize;
      nodes.push({ centroid: { x, y: height, z }, mass: massPerSegment, volume });
    }

    for (let i = 0; i < segments - 1; i++) {
      const a = base + i;
      const c = base + i + 1;
      const na = nodes[a].centroid;
      const nc = nodes[c].centroid;
      bonds.push({
        node0: a,
        node1: c,
        centroid: { x: (na.x + nc.x) * 0.5, y: height, z },
        normal: { x: 1, y: 0, z: 0 }, // bonds run along the beam axis
        area,
      });
    }
  }

  const spacing: Vec3 = { x: segmentSize, y: segmentSize, z: segmentSize };
  return { nodes, bonds, spacing };
}

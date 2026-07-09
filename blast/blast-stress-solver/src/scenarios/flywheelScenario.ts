/**
 * Flywheel scenario — a centrifugal-burst showcase with structure.
 *
 * Builds a single free-floating (no anchor) wheel lying flat in the XZ plane: a central hub, a ring
 * of `spokes` radial spoke arms, and a closed `rimBlocks` hoop welded onto the spoke tips. Because
 * every node is connected, the core spawns the whole wheel as one dynamic rigid body / actor. Spin
 * it hard about +Y (with gravity zero) and the stress solver's centrifugal term loads the rim and
 * spoke bonds — strongest out at the rim, where ω²r is largest — until they overstress and the wheel
 * tears itself apart, shedding rim arcs first and then whole spoke assemblies. This is the same
 * centrifugal mechanism the `spinningBeams` scenario isolates, but on a recognizable machine part so
 * the failure reads as an overspeed burst.
 *
 * The scene is anchor-free and gravity-agnostic: callers set `gravity: 0`, enable centrifugal via
 * `setSolverCentrifugalEnabled(true)`, and spin the body themselves (see the `flywheel-burst` demo
 * and `flywheel.showcase.test.ts`). Per-node sizes are emitted in `parameters.fragmentSizes` so the
 * hub renders chunky and the rim/spoke blocks render as a fine ring of cubes.
 */
import type { ScenarioBond, ScenarioDesc, ScenarioNode, Vec3 } from '../rapier/types';

export interface FlywheelOptions {
  /** Number of rim blocks around the circumference (more ⇒ smoother ring). */
  rimBlocks?: number;
  /** Number of radial spokes. Must divide `rimBlocks` so each spoke lands on a rim block. */
  spokes?: number;
  /** Segments per spoke, from the hub out to the rim. */
  spokeSegments?: number;
  /** Rim centerline radius (m). */
  rimRadius?: number;
  /** Hub radius — where the spokes begin (m). */
  hubRadius?: number;
  /** Height to float the wheel above the origin (m). The core spawns an invisible ground plane at
   *  y=0; lifting the wheel clear keeps the only load centrifugal. */
  height?: number;
  /** Mass of each rim block (kg). The rim carries the most centrifugal load, so this dominates. */
  rimBlockMass?: number;
  /** Mass of each spoke segment (kg). */
  spokeSegmentMass?: number;
  /** Mass of the hub (kg). */
  hubMass?: number;
  /** Multiplier on every bond's contact area. Smaller area ⇒ higher stress ⇒ easier to burst. */
  areaScale?: number;
}

export const DEFAULT_FLYWHEEL_OPTIONS: Required<FlywheelOptions> = {
  rimBlocks: 72,
  spokes: 6,
  spokeSegments: 4,
  rimRadius: 3.0,
  hubRadius: 0.5,
  height: 6.0,
  rimBlockMass: 6.0,
  spokeSegmentMass: 3.0,
  hubMass: 8.0,
  areaScale: 1.0,
};

type NodeRole = 'hub' | 'spoke' | 'rim';

export interface FlywheelScenario extends ScenarioDesc {
  /** Per-node role, parallel to `nodes`, so the demo can classify fragments (rim arc vs spoke). */
  roles: NodeRole[];
}

/**
 * Build the flywheel scenario. Every node is dynamic (no support node), so the whole wheel is one
 * free body that can be spun and burst purely by centrifugal load.
 */
export function buildFlywheelScenario(opts: FlywheelOptions = {}): FlywheelScenario {
  const o = { ...DEFAULT_FLYWHEEL_OPTIONS, ...opts };
  const rimBlocks = Math.max(8, Math.floor(o.rimBlocks));
  // Keep spokes a divisor of rimBlocks so each spoke tip lands exactly on a rim block.
  let spokes = Math.max(1, Math.floor(o.spokes));
  while (spokes > 1 && rimBlocks % spokes !== 0) spokes--;
  const spokeSegments = Math.max(1, Math.floor(o.spokeSegments));

  const nodes: ScenarioNode[] = [];
  const bonds: ScenarioBond[] = [];
  const roles: NodeRole[] = [];
  const fragmentSizes: Vec3[] = [];

  const { rimRadius, hubRadius, height } = o;
  // A node at angle θ sits at (r·cosθ, height, r·sinθ); its radial (outward) unit is (cosθ,0,sinθ).
  const at = (angle: number, r: number): Vec3 => ({
    x: r * Math.cos(angle),
    y: height,
    z: r * Math.sin(angle),
  });
  const radialDir = (angle: number): Vec3 => ({ x: Math.cos(angle), y: 0, z: Math.sin(angle) });

  // Block sizing for the renderer (kept cubic so the axis-aligned box meshes read cleanly at every
  // angle as the body rotates). The rim cube slightly exceeds the arc pitch so the ring looks solid.
  const arcPitch = (2 * Math.PI * rimRadius) / rimBlocks;
  const rimCube = arcPitch * 1.25;
  const spokeCube = arcPitch * 1.2;
  const hubCube = hubRadius * 1.8;
  const cube = (s: number): Vec3 => ({ x: s, y: s, z: s });

  // Bond cross-section areas. Small relative to the masses so realistic spins reach fatal stress.
  const rimArea = rimCube * rimCube * 0.12 * o.areaScale;
  const spokeArea = spokeCube * spokeCube * 0.5 * o.areaScale; // chunky: spokes outlast the rim
  const weldArea = spokeCube * spokeCube * 0.18 * o.areaScale; // spoke-tip-to-rim weld is the weak link

  // ── Hub (node 0) ──
  nodes.push({ centroid: at(0, 0), mass: o.hubMass, volume: hubCube ** 3 });
  roles.push('hub');
  fragmentSizes.push(cube(hubCube));

  // ── Spokes ──
  const spokeOuter = rimRadius - rimCube * 0.5; // spoke tip meets the rim's inner face
  const segLen = (spokeOuter - hubRadius) / spokeSegments;
  const spokeTip: number[] = [];
  for (let s = 0; s < spokes; s++) {
    const angle = (s * 2 * Math.PI) / spokes;
    const dir = radialDir(angle);
    let prev = 0; // hub
    for (let seg = 0; seg < spokeSegments; seg++) {
      const r = hubRadius + (seg + 0.5) * segLen;
      const index = nodes.length;
      nodes.push({ centroid: at(angle, r), mass: o.spokeSegmentMass, volume: spokeCube ** 3 });
      roles.push('spoke');
      fragmentSizes.push(cube(spokeCube));
      const jointR = hubRadius + seg * segLen;
      bonds.push({ node0: prev, node1: index, centroid: at(angle, jointR), normal: dir, area: spokeArea });
      prev = index;
    }
    spokeTip.push(prev);
  }

  // ── Rim hoop ──
  const rimBase = nodes.length;
  const rimNode = (slot: number) => rimBase + (slot % rimBlocks);
  for (let slot = 0; slot < rimBlocks; slot++) {
    const angle = (slot * 2 * Math.PI) / rimBlocks;
    nodes.push({ centroid: at(angle, rimRadius), mass: o.rimBlockMass, volume: rimCube ** 3 });
    roles.push('rim');
    fragmentSizes.push(cube(rimCube));
  }
  for (let slot = 0; slot < rimBlocks; slot++) {
    const a = nodes[rimNode(slot)].centroid;
    const b = nodes[rimNode(slot + 1)].centroid;
    const tangent = normalize({ x: b.x - a.x, y: b.y - a.y, z: b.z - a.z });
    bonds.push({
      node0: rimNode(slot),
      node1: rimNode(slot + 1),
      centroid: { x: (a.x + b.x) * 0.5, y: height, z: (a.z + b.z) * 0.5 },
      normal: tangent,
      area: rimArea,
    });
  }

  // ── Welds: each spoke tip onto its rim block ──
  const slotsPerSpoke = rimBlocks / spokes;
  for (let s = 0; s < spokes; s++) {
    const angle = (s * 2 * Math.PI) / spokes;
    bonds.push({
      node0: spokeTip[s],
      node1: rimNode(s * slotsPerSpoke),
      centroid: at(angle, spokeOuter),
      normal: radialDir(angle),
      area: weldArea,
    });
  }

  return {
    nodes,
    bonds,
    roles,
    spacing: cube(rimCube),
    parameters: { fragmentSizes },
  };
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z);
  return len > 0 ? { x: v.x / len, y: v.y / len, z: v.z / len } : { x: 1, y: 0, z: 0 };
}

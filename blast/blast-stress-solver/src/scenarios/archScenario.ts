/**
 * Stone arch scenario — a showcase for *compression* stress and the keystone principle.
 *
 * Produces a semicircular voussoir arch: a ring of wedge-shaped stone blocks spanning two fixed
 * abutments. The voussoirs are one connected, anchor-free component, so the core spawns the whole
 * ring as a single dynamic rigid body that simply *rests* (by contact) on the abutment supports —
 * exactly like a real dry-stone arch, which is held up by nothing but the compression passing from
 * stone to stone down to the ground (the "thrust line").
 *
 * Because the ring is not bonded to the supports, severing the crown joints (cut the keystone's
 * bonds) splits it into two unsupported half-arches whose centres of mass overhang their feet — so
 * they rotate off the abutments and the arch collapses. That is the whole point of the scene: the
 * keystone is what locks the ring, and the only thing keeping every block aloft is mutual
 * compression.
 *
 * The builder is pure data (no Three.js). Each voussoir's eight local hull corners are emitted in
 * `parameters.voussoirHulls` so a renderer can build proper wedge geometry (and convex-hull
 * colliders) from them; the test/headless path can ignore them and fall back to the box colliders
 * sized from `parameters.fragmentSizes`.
 */
import type { ScenarioBond, ScenarioDesc, ScenarioNode, Vec3 } from '../rapier/types';

export interface ArchScenarioOptions {
  /** Number of voussoirs (arch stones). Forced odd so there is a single centre keystone. */
  voussoirs?: number;
  /** Radius to the mid-line of the ring (m). */
  radius?: number;
  /** Radial thickness of each voussoir, intrados → extrados (m). */
  thickness?: number;
  /** Depth of the arch along Z (m). */
  depth?: number;
  /** Height of the springing line / abutment tops above the ground (m). Creates the span below. */
  springingHeight?: number;
  /** Stone density (kg/m³). Sandstone ≈ 2200. */
  density?: number;
}

export const DEFAULT_ARCH_OPTIONS: Required<ArchScenarioOptions> = {
  voussoirs: 15,
  radius: 7,
  thickness: 1.4,
  depth: 3.5,
  springingHeight: 4,
  density: 2200,
};

export interface ArchScenarioResult {
  scenario: ScenarioDesc;
  /** Node index of the centre keystone voussoir. */
  keystoneIndex: number;
  /** Number of voussoir nodes (the first `archNodeCount` nodes; the rest are abutment supports). */
  archNodeCount: number;
}

/**
 * Build the stone-arch scenario. The voussoir ring is fully dynamic (rests on the abutments by
 * contact); the two abutments are mass-0 support nodes pinned to the world.
 */
export function buildArchScenario(opts: ArchScenarioOptions = {}): ArchScenarioResult {
  const { voussoirs, radius, thickness, depth, springingHeight, density } = {
    ...DEFAULT_ARCH_OPTIONS,
    ...opts,
  };

  // Force an odd count so the crown is a single keystone.
  const N = voussoirs % 2 === 0 ? voussoirs + 1 : voussoirs;
  const dTheta = Math.PI / N; // angular width of one voussoir
  const Ri = radius - thickness / 2; // intrados radius
  const Ro = radius + thickness / 2; // extrados radius
  const halfDepth = depth / 2;

  const nodes: ScenarioNode[] = [];
  const bonds: ScenarioBond[] = [];
  const fragmentSizes: Vec3[] = [];
  const voussoirHulls: number[][] = [];
  const nodeColors: number[] = [];

  // World point on the arch at angle θ (measured from the +X springing) and radius r.
  const at = (theta: number, r: number, z: number): [number, number, number] => [
    r * Math.cos(theta),
    springingHeight + r * Math.sin(theta),
    z,
  ];

  for (let i = 0; i < N; i++) {
    const tc = dTheta * (i + 0.5); // centre angle of this voussoir
    const tA = tc - dTheta / 2;
    const tB = tc + dTheta / 2;

    // Eight hull corners: {θ-,θ+} × {intrados,extrados} × {front,back}.
    const corners: Array<[number, number, number]> = [
      at(tA, Ri, halfDepth), at(tB, Ri, halfDepth), at(tB, Ro, halfDepth), at(tA, Ro, halfDepth),
      at(tA, Ri, -halfDepth), at(tB, Ri, -halfDepth), at(tB, Ro, -halfDepth), at(tA, Ro, -halfDepth),
    ];

    // Centroid = mean of the corners; the hull is stored relative to it (local space).
    const c: [number, number, number] = [0, 0, 0];
    for (const p of corners) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    c[0] /= corners.length; c[1] /= corners.length; c[2] /= corners.length;

    const hull: number[] = [];
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const p of corners) {
      const lx = p[0] - c[0], ly = p[1] - c[1], lz = p[2] - c[2];
      hull.push(lx, ly, lz);
      if (lx < minX) minX = lx; if (lx > maxX) maxX = lx;
      if (ly < minY) minY = ly; if (ly > maxY) maxY = ly;
      if (lz < minZ) minZ = lz; if (lz > maxZ) maxZ = lz;
    }

    // True annular-sector volume (× depth), so masses scale physically with stone size.
    const volume = dTheta * ((Ro * Ro - Ri * Ri) / 2) * depth;

    nodes.push({ centroid: { x: c[0], y: c[1], z: c[2] }, mass: volume * density, volume });
    fragmentSizes.push({ x: Math.max(maxX - minX, 0.05), y: Math.max(maxY - minY, 0.05), z: Math.max(maxZ - minZ, 0.05) });
    voussoirHulls.push(hull);
    nodeColors.push(0x9a8c79); // warm sandstone (the demo tints the keystone separately)
  }

  // Radial-joint bonds between adjacent voussoirs. The bond normal is the *circumferential*
  // (tangential) direction at the joint, so the gravity load is carried as compression along it.
  const jointArea = thickness * depth;
  for (let i = 0; i < N - 1; i++) {
    const tj = dTheta * (i + 1); // joint angle between voussoir i and i+1
    const [jx, jy, jz] = at(tj, radius, 0);
    bonds.push({
      node0: i,
      node1: i + 1,
      centroid: { x: jx, y: jy, z: jz },
      normal: { x: -Math.sin(tj), y: Math.cos(tj), z: 0 },
      area: jointArea,
    });
  }

  const keystoneIndex = (N - 1) >> 1;
  const archNodeCount = N;

  // Two fixed abutments (mass 0). Each one's flat top sits exactly at the springing line, directly
  // under the foot of the end voussoir, so the ring rests on them. They are NOT bonded to the ring.
  const abutW = thickness * 1.6;
  const abutD = depth * 1.15;
  const abutH = springingHeight; // from ground (y=0) up to the springing line
  const addAbutment = (sideX: number) => {
    nodes.push({
      centroid: { x: sideX * radius, y: springingHeight - abutH / 2, z: 0 },
      mass: 0,
      volume: abutW * abutH * abutD,
    });
    fragmentSizes.push({ x: abutW, y: abutH, z: abutD });
    voussoirHulls.push([]); // no hull → renderer/collider fall back to a box
    nodeColors.push(0x5b6066); // cool grey granite piers
  };
  addAbutment(1);
  addAbutment(-1);

  const scenario: ScenarioDesc = {
    nodes,
    bonds,
    parameters: { fragmentSizes, voussoirHulls, nodeColors, keystoneIndex, archNodeCount },
  };

  return { scenario, keystoneIndex, archNodeCount };
}

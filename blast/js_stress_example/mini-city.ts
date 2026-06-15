/**
 * Mini-City Demo
 *
 * Composes a grid of independently fractured tower blocks into a SINGLE
 * physics world / stress graph, then lets you flatten the whole skyline.
 *
 * Each building is generated with `buildFracturedTowerScenario` (Voronoi-
 * fractured walls, columns, floor plates + foundation). The per-building
 * scenarios are then *merged* — node centroids and bond indices are offset
 * onto a city grid, and the `fragmentSizes` / `fragmentGeometries` parameter
 * arrays are concatenated in node order — so the runtime sees one big
 * ScenarioDesc made of many disconnected components (one independent
 * stress-solve per building) sharing one Rapier world and ground.
 *
 * Destruction:
 *   - Click            → throw a wrecking ball at the skyline
 *   - ☄ Meteor Storm   → staggered barrage raining across every block
 *   - ✸ Detonate       → demolition charges at every base + a radial shockwave
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import * as pinata from '@dgreenheck/three-pinata';
import { buildDestructibleCore, createFrameProfilerOverlay, createRecordingOverlay, buildSpatialCollisionTree } from 'blast-stress-solver/rapier';
import {
  createDestructibleThreeBundle,
  RapierDebugRenderer,
} from 'blast-stress-solver/three';
import { pipelineCoreOverrides, mountPipelineControls } from './pipeline-controls.js';
import { mountPhysicsControls, physicsCoreOverrides } from './physics-controls.js';
import { mountShooter } from './shooter-fps.js';
import {
  buildFracturedTowerScenario,
  buildHighRiseScenarioAsync,
  buildHouseScenario,
} from 'blast-stress-solver/scenarios';

type Vec3 = { x: number; y: number; z: number };
type ScenarioDesc = Awaited<ReturnType<typeof buildFracturedTowerScenario>>;

// What kind of building sits on a lot, and (for the tall ones) how it is built.
type BuildingKind = 'house' | 'midrise' | 'highrise';
type Construction = 'fractured' | 'skeleton';

// ── Config (driven by the control panel) ──────────────────────
const CONFIG = {
  city: {
    grid: 3, // grid x grid blocks
    street: 9, // gap between building footprints (m)
    widthMin: 6,
    widthMax: 9,
    maxFloors: 8, // tallest downtown high-rise (floors)
    fragments: 5, // Voronoi fragments per wall (floors/columns scale from this)
    seed: 7,
    massDensityMin: 600, // kg per m² per floor (min, randomised per building)
    massDensityMax: 1200, // kg per m² per floor (max)

    // ── Composition / zoning ──────────────────────────────────
    // The city is zoned by distance from its centre: a downtown core of tall buildings,
    // surrounded by a residential ring of low houses. Proportion sliders bias the mix
    // within each zone. All picks are seeded (mulberry32), so a seed reproduces the layout.
    downtownFrac: 0.45, // radius of the downtown core, as a fraction of the city half-extent
    highRiseShare: 0.4, // P(high-rise vs mid-rise) for a downtown lot
    suburbShopShare: 0.15, // P(a small mid-rise "shop" appearing among the suburban houses)
    emptyLotShare: 0.08, // P(a cell is left empty — a park/plaza — for street variety)
    houseMaxFloors: 1, // a lot with ≤ this many floors becomes a house
    midRiseMaxFloors: 4, // above houseMaxFloors and ≤ this → mid-rise; taller → high-rise
    // How tall buildings are constructed (per band). 'fractured' = the simpler Voronoi-
    // fractured concrete tower; 'skeleton' = the realistic high-rise (concrete frame +
    // frangible drywall infill).
    midRiseConstruction: 'fractured' as Construction,
    highRiseConstruction: 'skeleton' as Construction,
  },
  projectile: { radius: 0.6, mass: 1500, speed: 60 },
  destroy: {
    meteors: 36,
    meteorMass: 3000,
    meteorRadius: 1.2,
    meteorSpeedMin: 50, // each meteor's speed is picked uniformly in [min, max] m/s
    meteorSpeedMax: 70,
    meteorAngleMin: 0, // incoming angle from vertical, degrees (0 = straight down) in [min, max]
    meteorAngleMax: 35,
    blastForce: 6, // radial shockwave strength (× per-node mass)
  },
  solver: { gravity: -9.81, materialScale: 1e10 },
  physics: {
    debrisCollisionMode: 'all',
    friction: 0.25,
    restitution: 0,
    contactForceScale: 30,
  },
  optimization: {
    // Damping defaults to off; 'afterGroundCollision' would damp small debris only after it lands ('always' floats falling debris).
    smallBodyDampingMode: 'off',
    debrisCleanupMode: 'afterGroundCollision',
    debrisTtlMs: 8000,
    maxCollidersForDebris: 2,
    // Island-aware solve: solve each disconnected group independently and skip groups that have
    // settled (no input change + already converged). On by default; toggled live in the sidebar.
    islandSolver: true,
    // Lazy intact colliders: intact buildings keep their per-fragment colliders disabled (out of
    // the broadphase) until a mover is about to hit them, then enable just-in-time. Huge idle win
    // on big cities; identical while intact and on approach. On by default; toggled live.
    lazyIntactColliders: true,
    // Render LOD: draw a still-intact (un-hit) building as a single proxy box instead of all its
    // fragment instances once it is far enough from the camera. Collapses the per-frame batched
    // instance/triangle load on a settled city; reverts to full fragments when hit or approached.
    intactProxies: true,
    intactProxyFar: 70, // proxy a building beyond this distance (m) from the camera
    intactProxyNear: 45, // reveal its real fragments again within this distance (m) — hysteresis
  },
  features: { debug: false, lodTree: false },
};

// ── Three.js setup ────────────────────────────────────────────
const canvas = document.getElementById('demo-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);
scene.fog = new THREE.FogExp2(0x0a0d13, 0.004);

const camera = new THREE.PerspectiveCamera(
  55,
  canvas.clientWidth / canvas.clientHeight,
  0.1,
  1200,
);
camera.position.set(60, 40, 80);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 8, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
dirLight.position.set(40, 70, 50);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -90;
dirLight.shadow.camera.right = 90;
dirLight.shadow.camera.top = 90;
dirLight.shadow.camera.bottom = -20;
dirLight.shadow.camera.far = 300;
scene.add(dirLight);

const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(600, 600),
  new THREE.MeshStandardMaterial({ color: 0x161a28, roughness: 0.95, metalness: 0.05 }),
);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.02;
groundMesh.receiveShadow = true;
scene.add(groundMesh);

// Subtle street grid so the city layout reads clearly.
const streetGrid = new THREE.GridHelper(600, 60, 0x2a3350, 0x1c2336);
(streetGrid.material as THREE.Material).transparent = true;
(streetGrid.material as THREE.Material).opacity = 0.35;
streetGrid.position.y = 0.01;
scene.add(streetGrid);

const stats = new Stats();
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0';
stats.dom.style.left = '0';
(document.querySelector('.viewport') as HTMLElement)?.appendChild(stats.dom);

// ── Stats / perf ──────────────────────────────────────────────
let _physicsMs = 0;
let _renderMs = 0;
const EMA = 0.12;
let initialBonds = 0;

function setText(id: string, value: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function updateStatus(core: any) {
  setText('stat-bodies', String(core.getRigidBodyCount()));
  setText('stat-bonds', `${core.getActiveBondsCount()} / ${initialBonds}`);
  setText('stat-projectiles', String(core.projectiles.length));
  const detached = core.chunks.filter((c: any) => c.detached).length;
  setText('stat-chunks', `${detached} detached`);
  setText('stat-fragments', String(core.chunks.length));
  const isl = core.getIslandSolverStats?.();
  setText('stat-islands', isl?.enabled ? `${isl.islandsSkipped}/${isl.islandCount} skipped` : 'off');
  const lz = core.getLazyColliderStats?.();
  setText('stat-lazy', lz?.enabled ? `${lz.dormantCount} dormant / ${lz.explodedCount} hit · ${lz.activeLeafFragments} active` : 'off');
  const proxies = visualsRef?.intactProxies;
  setText(
    'stat-proxies',
    CONFIG.optimization.intactProxies && proxies ? `${proxies.proxiedCount()} as box` : 'off',
  );
}
function updatePerf() {
  setText('stat-physics-ms', _physicsMs.toFixed(1) + ' ms');
  setText('stat-render-ms', _renderMs.toFixed(1) + ' ms');
  setText('stat-draw-calls', String(renderer.info.render.calls));
  setText('stat-triangles', renderer.info.render.triangles.toLocaleString());
}

// ── Deterministic RNG (so a given seed reproduces a skyline) ───
function mulberry32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Scenario merge ────────────────────────────────────────────
// Offset every building onto the city grid and concatenate into one scenario.
// Geometry/size live in scenario.parameters and are per-node, so they are
// concatenated in node order. Fragment geometry is local to each node's
// centroid, so only centroids (and bond indices) need offsetting.
type Part = { scenario: ScenarioDesc; offset: Vec3; massMultiplier?: number };

// Offset every fragment index in a CollisionGroup subtree by `base` (deep copy, no mutation of the
// shared template tree) so the per-tower authored hierarchy survives the merge into one scenario.
function offsetCollisionGroup(g: any, base: number): any {
  if (g.children) return { children: g.children.map((c: any) => offsetCollisionGroup(c, base)) };
  return { fragments: (g.fragments as number[]).map((i) => i + base) };
}

function mergeScenarios(parts: Part[]): ScenarioDesc {
  const nodes: any[] = [];
  const bonds: any[] = [];
  const fragmentSizes: any[] = [];
  const fragmentGeometries: any[] = [];
  const collisionTree: any[] = [];
  let base = 0;

  for (const { scenario, offset, massMultiplier = 1 } of parts) {
    const params = (scenario.parameters ?? {}) as any;
    const sizes = (params.fragmentSizes ?? []) as Vec3[];
    const geos = (params.fragmentGeometries ?? []) as unknown[];

    scenario.nodes.forEach((n: any, i: number) => {
      nodes.push({
        centroid: {
          x: n.centroid.x + offset.x,
          y: n.centroid.y + offset.y,
          z: n.centroid.z + offset.z,
        },
        mass: n.mass != null && n.mass > 0 ? n.mass * massMultiplier : n.mass,
        volume: n.volume,
      });
      fragmentSizes.push(sizes[i]);
      fragmentGeometries.push(geos[i]);
    });

    for (const b of scenario.bonds as any[]) {
      bonds.push({
        node0: b.node0 + base,
        node1: b.node1 + base,
        centroid: {
          x: b.centroid.x + offset.x,
          y: b.centroid.y + offset.y,
          z: b.centroid.z + offset.z,
        },
        normal: { ...b.normal },
        area: b.area,
      });
    }

    // Carry each building's collision-LOD tree, offsetting its fragment indices into the
    // merged node array (one root per building). Fractured towers ship an AUTHORED semantic
    // tree (building → floor → element); house / high-rise parts don't, so we synthesize a
    // spatial one for them here. Either way every part contributes a tree, so localized hits
    // only materialize the struck region's colliders regardless of building type.
    let tree = (scenario as any).collisionTree as any[] | undefined;
    if (!tree) tree = buildSpatialCollisionTree(scenario, { leafMaxFragments: 24 });
    for (const root of tree) collisionTree.push(offsetCollisionGroup(root, base));

    base += scenario.nodes.length;
  }

  return {
    nodes,
    bonds,
    parameters: { fragmentSizes, fragmentGeometries },
    ...(collisionTree.length ? { collisionTree } : {}),
  } as ScenarioDesc;
}

// ── City description (for camera, targeting, detonation) ──────
type Building = {
  cx: number;
  cz: number;
  width: number;
  height: number;
  nodeStart: number;
  nodeCount: number;
};

const buildings: Building[] = [];
let cityRadius = 40;
let cityMaxHeight = 24;
// World centroids + masses of every node (built once at merge); used by the shockwave.
let nodeCentroids: Vec3[] = [];
let nodeMasses: number[] = [];

// How many distinct fracture patterns to keep per footprint shape. Repeated
// buildings reuse one of these instead of re-fracturing — a few variants is
// plenty for the skyline to read as varied (nobody notices two identical
// towers in a 400-building city). Bump for more variety at higher build cost.
const VARIANTS_PER_SHAPE = 3;

async function buildCity(): Promise<ScenarioDesc> {
  buildings.length = 0;
  const {
    grid, street, widthMin, widthMax, maxFloors, fragments, seed,
    massDensityMin, massDensityMax, downtownFrac, highRiseShare, suburbShopShare,
    emptyLotShare, houseMaxFloors, midRiseMaxFloors, midRiseConstruction, highRiseConstruction,
  } = CONFIG.city;
  const rng = mulberry32(seed * 2654435761);

  const cell = widthMax + street; // grid pitch
  const span = (grid - 1) * cell;
  const half = span / 2;
  const maxDist = Math.max(1e-3, Math.hypot(half, half)); // corner distance, for zoning
  const FLOOR_HEIGHT = 3;

  const parts: Part[] = [];
  let nodeBase = 0;
  cityMaxHeight = 0;
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  const total = grid * grid;

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const randInt = (lo: number, hi: number) => lo + Math.floor(rng() * (Math.max(lo, hi) - lo + 1));

  // Fracture is the dominant startup cost (Voronoi tessellation + O(n²) bond
  // detection per building), and it runs serially on the main thread. But
  // buildings only vary by (kind, construction, width, floorCount) — a handful of
  // distinct shapes. So we build at most a few variants per shape and reuse them:
  // a grid² city does ≤ shapes × variants fractures regardless of grid size.
  //
  // Sharing a cached scenario across instances is safe: mergeScenarios reads it
  // and emits fresh, offset node/bond objects without mutating the source, and
  // BatchedMesh.addGeometry copies vertex data, so the shared geometry refs are
  // never mutated per-instance (and sharing them keeps memory flat for huge
  // cities).
  const templateCache = new Map<string, ScenarioDesc[]>();
  let fractureCount = 0;

  // ── Building builders ─────────────────────────────────────────
  // Simpler Voronoi-fractured concrete tower (the original mini-city building).
  const fractureTower = (width: number, floorCount: number, floorHeight: number) =>
    buildFracturedTowerScenario({
      width,
      depth: width,
      floorCount,
      floorHeight,
      thickness: 0.3,
      floorThickness: 0.2,
      columnSize: 0.6,
      columnsX: 2,
      columnsZ: 2,
      fragmentCountPerWall: fragments,
      fragmentCountPerFloor: fragments,
      fragmentCountPerColumn: Math.max(2, Math.round(fragments / 2)),
      deckMass: width * width * floorCount * 320,
      pinata: pinata as any,
    });

  // Realistic high-rise: stiff concrete frame + frangible drywall infill. Kept
  // deliberately coarse (few column lines / slab divisions, infill left as boxes) so a
  // whole city of these stays within a sane collider budget; per-fragment masses come
  // from real densities, so it needs no mass multiplier.
  const skeletonTower = (width: number, floorCount: number, floorHeight: number) =>
    buildHighRiseScenarioAsync({
      width,
      depth: width,
      floorCount,
      floorHeight,
      columnsX: clamp(Math.round(width / 5), 2, 4),
      columnsZ: clamp(Math.round(width / 5), 2, 4),
      slabDivX: clamp(Math.round(width / 4), 2, 5),
      slabDivZ: clamp(Math.round(width / 4), 2, 5),
      fractureInfill: false,
      fractureColumns: false,
      bondMode: 'proximity',
      pinata: pinata as any,
    });

  // One-storey wood-framed house (boxes; furniture off to keep the city light).
  const houseLot = (width: number) =>
    Promise.resolve(
      buildHouseScenario({
        width,
        depth: width,
        wallHeight: FLOOR_HEIGHT,
        roofRise: width * 0.18,
        furniture: false,
      }),
    );

  type Spec = { kind: BuildingKind; construction: Construction; width: number; floorCount: number };

  // Decide what occupies a lot from its zone (distance from centre) + the proportion
  // sliders. Returns null for an empty lot (a park / plaza). All draws are seeded.
  const pickSpec = (d: number): Spec | null => {
    if (rng() < emptyLotShare) return null;
    if (d <= downtownFrac) {
      // Downtown core — tall buildings.
      const highRise = rng() < highRiseShare;
      const kind: BuildingKind = highRise ? 'highrise' : 'midrise';
      const construction = highRise ? highRiseConstruction : midRiseConstruction;
      const floorCount = highRise
        ? randInt(midRiseMaxFloors + 1, maxFloors)
        : randInt(Math.max(2, houseMaxFloors + 1), midRiseMaxFloors);
      const width = Math.round(widthMin + rng() * (widthMax - widthMin));
      return { kind, construction, width, floorCount };
    }
    // Residential ring — mostly houses, with the occasional small "shop".
    if (rng() < suburbShopShare) {
      const floorCount = randInt(Math.max(2, houseMaxFloors + 1), Math.min(3, midRiseMaxFloors));
      const width = Math.round(widthMin + rng() * (widthMax - widthMin) * 0.5);
      return { kind: 'midrise', construction: midRiseConstruction, width, floorCount };
    }
    const width = Math.round(widthMin + rng() * 2); // houses sit on smaller footprints
    return { kind: 'house', construction: 'fractured', width, floorCount: houseMaxFloors };
  };

  const buildPart = (spec: Spec): Promise<ScenarioDesc> => {
    if (spec.kind === 'house') return houseLot(spec.width);
    return spec.construction === 'skeleton'
      ? skeletonTower(spec.width, spec.floorCount, FLOOR_HEIGHT)
      : fractureTower(spec.width, spec.floorCount, FLOOR_HEIGHT);
  };

  for (let r = 0; r < grid; r++) {
    for (let c = 0; c < grid; c++) {
      const idx = r * grid + c;
      const cx = -half + c * cell;
      const cz = -half + r * cell;
      const d = Math.hypot(cx, cz) / maxDist; // 0 at centre → 1 at the corners

      const spec = pickSpec(d);
      if (!spec) {
        // Empty lot (park / plaza) — leave the gap for street variety.
        if (hint && idx % 16 === 0) hint.textContent = `Assembling city… (${idx + 1}/${total})`;
        continue;
      }
      const { kind, construction, width, floorCount } = spec;
      const height = floorCount * FLOOR_HEIGHT;
      cityMaxHeight = Math.max(cityMaxHeight, height);

      // Deterministic builders (houses, box high-rises) produce identical instances, so
      // one cached variant suffices; only the random Voronoi tower benefits from several.
      const variantCap = kind !== 'house' && construction === 'fractured' ? VARIANTS_PER_SHAPE : 1;
      const buildKey = `${kind}|${construction}|${width}|${floorCount}`;
      let variants = templateCache.get(buildKey);
      if (!variants) {
        variants = [];
        templateCache.set(buildKey, variants);
      }

      let scenario: ScenarioDesc;
      if (variants.length < variantCap) {
        // Cache miss — pay the build cost once, then yield so the browser paints the
        // progress hint (only the expensive path yields).
        fractureCount += 1;
        if (hint) hint.textContent = `Building city… (${idx + 1}/${total})`;
        await new Promise((res) => setTimeout(res, 0));
        scenario = await buildPart(spec);
        variants.push(scenario);
      } else {
        // Cache hit — reuse a built variant. Refresh the hint only occasionally; yielding
        // on every cell would cost grid² timer ticks.
        scenario = variants[(rng() * variants.length) | 0];
        if (hint && idx % 16 === 0) hint.textContent = `Assembling city… (${idx + 1}/${total})`;
      }

      // Jitter each building a little within its cell so rows don't look perfectly rigid.
      const jit = Math.min(street * 0.5, 3);
      const ox = cx + (rng() - 0.5) * jit;
      const oz = cz + (rng() - 0.5) * jit;

      // Houses / box high-rises carry real per-fragment densities already (default ×1);
      // only the fractured tower (which bakes a 320 kg/m² base density) is rescaled to the
      // requested mass-density range.
      const massMultiplier =
        kind !== 'house' && construction === 'fractured'
          ? (massDensityMin + rng() * (massDensityMax - massDensityMin)) / 320
          : 1;
      parts.push({ scenario, offset: { x: ox, y: 0, z: oz }, massMultiplier });

      buildings.push({ cx: ox, cz: oz, width, height, nodeStart: nodeBase, nodeCount: scenario.nodes.length });
      nodeBase += scenario.nodes.length;
    }
  }

  cityRadius = Math.max(40, half + widthMax);

  console.log(
    `Mini-city: ${buildings.length}/${total} lots built from ${fractureCount} unique builds ` +
      `(${templateCache.size} distinct shapes)`,
  );

  const merged = mergeScenarios(parts);
  nodeCentroids = merged.nodes.map((n: any) => ({ ...n.centroid }));
  nodeMasses = merged.nodes.map((n: any) => n.mass ?? 0);
  return merged;
}

// ── Scene lifecycle ───────────────────────────────────────────
let coreRef: Awaited<ReturnType<typeof buildDestructibleCore>> | null = null;
// Reusable, self-mounting live frame-profiler overlay (bottom-left): per-phase
// physics cost + A/B comparison.
const profiler = createFrameProfilerOverlay();
// Session recorder — ● Record captures every dynamic body's per-frame
// position/orientation + linear/angular velocity, every input (clicks, meteor
// storm, detonation, forces) and every fracture/topology change into a single
// gzipped bug-report bundle (⬇ Save). The profiler trace rides along in the
// bundle. Zero allocation on the hot path.
const recorder = createRecordingOverlay({
  mount: document.getElementById('recorder-slot') ?? undefined,
  exportName: 'mini-city-recording',
  getProfilerExport: () => profiler.exportData(),
});
let visualsRef: ReturnType<typeof createDestructibleThreeBundle> | null = null;
let shooter: ReturnType<typeof mountShooter> | null = null;
let cityGroup: THREE.Group | null = null;
let rapierDebug: RapierDebugRenderer | null = null;
let rebuilding = false;

// ── Collision-LOD tree visualizer ─────────────────────────────
// Draws the hierarchical collision groups: a faint white box per building (root) and a colored
// wireframe box per leaf region — dim blue while dormant (collider disabled / out of broadphase),
// bright orange once enabled (materialized by a nearby mover). Watch regions light up as you shoot.
class LodTreeViz {
  private group = new THREE.Group();
  private leafBoxes: THREE.LineSegments[] = [];
  private dormant = new THREE.LineBasicMaterial({ color: 0x2a5cff, transparent: true, opacity: 0.18 });
  private active = new THREE.LineBasicMaterial({ color: 0xff8a00, transparent: true, opacity: 0.95 });
  private rootMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12 });
  constructor(private scene: THREE.Scene) { this.group.visible = false; scene.add(this.group); }
  private boxAt(min: any, max: any, mat: THREE.Material): THREE.LineSegments {
    const w = Math.max(0.05, max.x - min.x), h = Math.max(0.05, max.y - min.y), d = Math.max(0.05, max.z - min.z);
    const seg = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)), mat);
    seg.position.set((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2);
    return seg;
  }
  rebuild(core: any) {
    this.group.clear();
    this.leafBoxes = [];
    const nodes = core?.getCollisionLodNodes?.() ?? [];
    for (const n of nodes) {
      if (n.depth === 0) this.group.add(this.boxAt(n.aabbMin, n.aabbMax, this.rootMat)); // building outline
      if (n.leaf) { const b = this.boxAt(n.aabbMin, n.aabbMax, this.dormant); this.group.add(b); this.leafBoxes.push(b); }
    }
  }
  update(core: any) {
    if (!this.group.visible) return;
    const nodes = (core?.getCollisionLodNodes?.() ?? []).filter((n: any) => n.leaf);
    for (let i = 0; i < this.leafBoxes.length && i < nodes.length; i++) {
      this.leafBoxes[i].material = nodes[i].enabled ? this.active : this.dormant;
    }
  }
  setEnabled(on: boolean) { this.group.visible = on; }
  dispose() { this.scene.remove(this.group); this.group.clear(); }
}
let lodViz: LodTreeViz | null = null;

async function initScene() {
  const hint = document.querySelector('.viewport-hint') as HTMLElement | null;
  if (hint) hint.textContent = 'Building city…';

  const scenario = await buildCity();
  // Hierarchical collision-LOD so a localized hit only materializes the struck region's colliders.
  // mergeScenarios carries each tower's AUTHORED tree (building → floor → wall/slab/column →
  // fragments) from buildFracturedTowerScenario; if that's ever absent we fall back to a generic
  // spatial split. Orthogonal to the bond graph — cannot change fracture output.
  if (!(scenario as any).collisionTree) {
    (scenario as any).collisionTree = buildSpatialCollisionTree(scenario, { leafMaxFragments: 24 });
  }
  console.log(
    `Mini-city: ${buildings.length} buildings, ${scenario.nodes.length} nodes, ${scenario.bonds.length} bonds`,
  );

  const core = await buildDestructibleCore({
    scenario,
    gravity: CONFIG.solver.gravity,
    materialScale: CONFIG.solver.materialScale,
    contactForceScale: CONFIG.physics.contactForceScale,
    ...physicsCoreOverrides(),
    // Lazy intact colliders is mini-city's own option (not part of the shared physics controls).
    lazyIntactColliders: CONFIG.optimization.lazyIntactColliders,
    ...pipelineCoreOverrides(),
  });

  const group = new THREE.Group();
  scene.add(group);
  const visuals = createDestructibleThreeBundle({
    core,
    scenario,
    root: group,
    useBatchedMesh: true,
    batchedMeshOptions: { enableBVH: false, bvhMargin: 5 },
    includeDebugLines: true,
    intactProxies: {
      enabled: CONFIG.optimization.intactProxies,
      farDistance: CONFIG.optimization.intactProxyFar,
      nearDistance: CONFIG.optimization.intactProxyNear,
    },
  });

  rapierDebug?.dispose();
  rapierDebug = new RapierDebugRenderer(scene, core.world as any, {
    enabled: CONFIG.features.debug,
  });

  lodViz?.dispose();
  lodViz = new LodTreeViz(scene);
  lodViz.rebuild(core);
  lodViz.setEnabled(CONFIG.features.lodTree);

  coreRef = core;
  core.setIslandSolver?.({ enabled: CONFIG.optimization.islandSolver }); // persist the toggle across rebuilds
  core.setLazyIntactColliders?.(CONFIG.optimization.lazyIntactColliders);
  recorder.attach(core, { scenario, meta: { demo: 'mini-city', config: CONFIG } });
  profiler.attach(core);
  visualsRef = visuals;
  cityGroup = group;
  initialBonds = core.getActiveBondsCount();

  // Frame the whole skyline.
  controls.target.set(0, cityMaxHeight * 0.35, 0);
  camera.position.set(cityRadius * 1.15, cityRadius * 0.85 + 12, cityRadius * 1.6);
  controls.update();

  if (hint) hint.textContent = 'Click the city · or use ☄ / ✸ to demolish it all';
}

async function rebuild() {
  if (rebuilding) return;
  rebuilding = true;
  barrage.length = 0;
  visualsRef?.dispose();
  coreRef?.dispose();
  if (cityGroup) scene.remove(cityGroup);
  coreRef = null;
  visualsRef = null;
  cityGroup = null;
  try {
    await initScene();
  } finally {
    rebuilding = false;
  }
}

// ── Projectiles ───────────────────────────────────────────────
// Click-to-shoot (ball + sticky-explosive modes) and the first-person camera
// are handled by the shared shooter module, mounted during boot below. The
// meteor storm + demolition-charge buttons still drive the core directly.

// ── City-wide destruction ─────────────────────────────────────
// Staggered projectile queue so a barrage spreads over several frames
// (more dramatic, and avoids a single-frame spawn spike).
const barrage: { at: number; spawn: any }[] = [];

function meteorStorm() {
  const core = coreRef;
  if (!core || !buildings.length) return;
  const rng = mulberry32((Math.random() * 1e9) | 0);
  const count = CONFIG.destroy.meteors;
  const now = performance.now();
  const interval = 1800 / Math.max(1, count); // spread over ~1.8s

  for (let i = 0; i < count; i++) {
    // Aim at a random building, biased toward upper floors for collapse.
    const b = buildings[(rng() * buildings.length) | 0];
    const target = {
      x: b.cx + (rng() - 0.5) * b.width,
      y: b.height * (0.4 + rng() * 0.6),
      z: b.cz + (rng() - 0.5) * b.width,
    };
    // Incoming angle, measured from vertical (0° = straight down), picked in [min, max].
    // Azimuth is random so the storm comes from all sides.
    const DEG = Math.PI / 180;
    const aLo = CONFIG.destroy.meteorAngleMin * DEG;
    const aHi = Math.max(aLo, CONFIG.destroy.meteorAngleMax * DEG);
    const theta = aLo + rng() * (aHi - aLo);
    const phi = rng() * Math.PI * 2;
    const sinT = Math.sin(theta);
    // Unit vector from the target up-and-out to the spawn point; the velocity is its reverse.
    const ux = sinT * Math.cos(phi);
    const uy = Math.cos(theta);
    const uz = sinT * Math.sin(phi);
    const dist = 70 + rng() * 30; // how far out (along the incoming ray) the meteor starts
    const spawn = { x: target.x + ux * dist, y: target.y + uy * dist, z: target.z + uz * dist };
    const lo = CONFIG.destroy.meteorSpeedMin;
    const hi = Math.max(lo, CONFIG.destroy.meteorSpeedMax);
    const speed = lo + rng() * (hi - lo);
    barrage.push({
      at: now + i * interval,
      spawn: {
        position: spawn,
        velocity: { x: -ux * speed, y: -uy * speed, z: -uz * speed },
        radius: CONFIG.destroy.meteorRadius,
        mass: CONFIG.destroy.meteorMass,
        ttl: 6000,
      },
    });
  }
}

function detonate() {
  const core = coreRef;
  if (!core || !buildings.length) return;
  const rng = mulberry32((Math.random() * 1e9) | 0);

  // 1) Demolition charges — fast upward impulses at each building base.
  for (const b of buildings) {
    const charges = 2;
    for (let i = 0; i < charges; i++) {
      core.enqueueProjectile({
        position: {
          x: b.cx + (rng() - 0.5) * b.width * 0.5,
          y: 0.6,
          z: b.cz + (rng() - 0.5) * b.width * 0.5,
        },
        velocity: {
          x: (rng() - 0.5) * 14,
          y: 26 + rng() * 10,
          z: (rng() - 0.5) * 14,
        },
        radius: 0.8,
        mass: 5000,
        ttl: 4000,
      });
    }
  }

  // 2) Radial shockwave — push every live node outward from the city centre,
  //    seeding stress in the bond graph so the whole skyline lets go.
  const k = CONFIG.destroy.blastForce;
  const chunks = core.chunks as any[];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (!chunk || !chunk.active || chunk.isSupport || chunk.bodyHandle == null) continue;
    const p = nodeCentroids[chunk.nodeIndex] ?? chunk.baseLocalOffset;
    if (!p) continue;
    let dx = p.x;
    let dz = p.z;
    const horiz = Math.hypot(dx, dz) || 1;
    dx /= horiz;
    dz /= horiz;
    const mass = nodeMasses[chunk.nodeIndex] || 1000;
    const mag = mass * k * 9.81;
    core.applyExternalForce(
      chunk.nodeIndex,
      { x: p.x, y: p.y, z: p.z },
      { x: dx * mag, y: mag * 0.6, z: dz * mag },
    );
  }
}

// ── Control panel wiring ──────────────────────────────────────
function bindSlider(
  id: string,
  obj: Record<string, any>,
  key: string,
  fmt?: (v: number) => string,
  onInput?: (v: number) => void,
) {
  const slider = document.getElementById(id) as HTMLInputElement | null;
  const display = document.getElementById(id + '-value');
  if (!slider) return;
  slider.value = String(obj[key]);
  if (display) display.textContent = fmt ? fmt(obj[key]) : String(obj[key]);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    obj[key] = v;
    if (display) display.textContent = fmt ? fmt(v) : String(v);
    onInput?.(v);
  });
}
function bindSelect(
  id: string,
  obj: Record<string, any>,
  key: string,
  onChange?: (v: string) => void,
) {
  const select = document.getElementById(id) as HTMLSelectElement | null;
  if (!select) return;
  select.value = String(obj[key]);
  select.addEventListener('change', () => {
    obj[key] = select.value;
    onChange?.(select.value);
  });
}
function bindToggle(
  id: string,
  obj: Record<string, any>,
  key: string,
  onChange?: (v: boolean) => void,
) {
  const box = document.getElementById(id) as HTMLInputElement | null;
  if (!box) return;
  box.checked = Boolean(obj[key]);
  box.addEventListener('change', () => {
    obj[key] = box.checked;
    onChange?.(box.checked);
  });
}

// City (deferred — needs Reset/Rebuild)
bindSlider('cfg-grid', CONFIG.city, 'grid', (v) => `${v}×${v}`);
bindSlider('cfg-street', CONFIG.city, 'street', (v) => v.toFixed(0) + ' m');
bindSlider('cfg-max-floors', CONFIG.city, 'maxFloors');
bindSlider('cfg-frags', CONFIG.city, 'fragments');
bindSlider('cfg-seed', CONFIG.city, 'seed');
bindSlider('cfg-mass-density-min', CONFIG.city, 'massDensityMin', (v) => v.toLocaleString() + ' kg/m²/fl');
bindSlider('cfg-mass-density-max', CONFIG.city, 'massDensityMax', (v) => v.toLocaleString() + ' kg/m²/fl');

// Composition / zoning (deferred — needs Rebuild). Sliders store a 0–1 fraction; show %.
const pct = (v: number) => Math.round(v * 100) + '%';
bindSlider('cfg-downtown', CONFIG.city, 'downtownFrac', pct);
bindSlider('cfg-highrise-share', CONFIG.city, 'highRiseShare', pct);
bindSlider('cfg-suburb-shops', CONFIG.city, 'suburbShopShare', pct);
bindSlider('cfg-empty-lots', CONFIG.city, 'emptyLotShare', pct);
bindSlider('cfg-midrise-floors', CONFIG.city, 'midRiseMaxFloors');
bindSelect('cfg-midrise-construction', CONFIG.city, 'midRiseConstruction');
bindSelect('cfg-highrise-construction', CONFIG.city, 'highRiseConstruction');

// Projectile (live at shoot time)
bindSlider('cfg-proj-radius', CONFIG.projectile, 'radius', (v) => v.toFixed(2) + ' m');
bindSlider('cfg-proj-mass', CONFIG.projectile, 'mass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-proj-speed', CONFIG.projectile, 'speed', (v) => v.toFixed(0) + ' m/s');

// Destruction
bindSlider('cfg-meteors', CONFIG.destroy, 'meteors', (v) => v.toFixed(0));
bindSlider('cfg-meteor-mass', CONFIG.destroy, 'meteorMass', (v) => v.toLocaleString() + ' kg');
bindSlider('cfg-meteor-speed-min', CONFIG.destroy, 'meteorSpeedMin', (v) => v.toFixed(0) + ' m/s');
bindSlider('cfg-meteor-speed-max', CONFIG.destroy, 'meteorSpeedMax', (v) => v.toFixed(0) + ' m/s');
bindSlider('cfg-meteor-angle-min', CONFIG.destroy, 'meteorAngleMin', (v) => v.toFixed(0) + '°');
bindSlider('cfg-meteor-angle-max', CONFIG.destroy, 'meteorAngleMax', (v) => v.toFixed(0) + '°');
bindSlider('cfg-blast', CONFIG.destroy, 'blastForce', (v) => v.toFixed(0) + '×');

// Solver
bindSlider('cfg-gravity', CONFIG.solver, 'gravity', (v) => v.toFixed(1) + ' m/s²', (v) =>
  coreRef?.setGravity(v),
);
{
  const slider = document.getElementById('cfg-material') as HTMLInputElement | null;
  const display = document.getElementById('cfg-material-value');
  if (slider) {
    const exp = Math.log10(CONFIG.solver.materialScale);
    slider.value = String(exp);
    if (display) display.textContent = `1e${exp.toFixed(0)}`;
    slider.addEventListener('input', () => {
      const e = parseFloat(slider.value);
      CONFIG.solver.materialScale = Math.pow(10, e);
      if (display) display.textContent = `1e${e.toFixed(1)}`;
    });
  }
}

// Shared Physics / Optimization controls (mini-city keeps its island-solver toggle + debug button).
mountPhysicsControls({
  getCore: () => coreRef,
  onRebuild: () => void rebuild(),
  include: { centrifugal: false, debug: false },
});
bindToggle('cfg-island-solver', CONFIG.optimization, 'islandSolver', (v) =>
  coreRef?.setIslandSolver?.({ enabled: v }),
);
bindToggle('cfg-lazy-colliders', CONFIG.optimization, 'lazyIntactColliders', (v) =>
  coreRef?.setLazyIntactColliders?.(v),
);
bindToggle('cfg-intact-proxies', CONFIG.optimization, 'intactProxies', (v) =>
  visualsRef?.intactProxies?.setEnabled(v),
);

// Actions
document.getElementById('btn-meteor')?.addEventListener('click', () => meteorStorm());
document.getElementById('btn-detonate')?.addEventListener('click', () => detonate());
document.getElementById('btn-reset')?.addEventListener('click', () => void rebuild());
document.getElementById('btn-debug')?.addEventListener('click', () => {
  CONFIG.features.debug = !CONFIG.features.debug;
  rapierDebug?.setEnabled(CONFIG.features.debug);
  const btn = document.getElementById('btn-debug')!;
  btn.textContent = CONFIG.features.debug ? '◈ Hide Debug' : '◇ Show Debug';
});
document.getElementById('btn-lod')?.addEventListener('click', () => {
  CONFIG.features.lodTree = !CONFIG.features.lodTree;
  lodViz?.setEnabled(CONFIG.features.lodTree);
  const btn = document.getElementById('btn-lod')!;
  btn.textContent = CONFIG.features.lodTree ? '◧ Hide LOD Tree' : '◫ Show LOD Tree';
});

// ── Render loop ───────────────────────────────────────────────
const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  profiler.render();
  stats.begin();
  const dt = Math.min(clock.getDelta(), 1 / 30);
  controls.update();

  if (coreRef && visualsRef) {
    // Flush any due barrage projectiles.
    if (barrage.length) {
      const now = performance.now();
      for (let i = barrage.length - 1; i >= 0; i--) {
        if (barrage[i].at <= now) {
          coreRef.enqueueProjectile(barrage[i].spawn);
          barrage.splice(i, 1);
        }
      }
    }

    const t0 = performance.now();
    coreRef.step(dt);
    _physicsMs += (performance.now() - t0 - _physicsMs) * EMA;
    visualsRef.update({ debug: CONFIG.features.debug, updateBVH: false, updateProjectiles: true, camera });
    rapierDebug?.update();
    lodViz?.update(coreRef);
    updateStatus(coreRef);
    recorder.render();
  }

  shooter?.update();

  const t1 = performance.now();
  renderer.render(scene, camera);
  _renderMs += (performance.now() - t1 - _renderMs) * EMA;
  updatePerf();
  stats.end();
}

function onResize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

mountPipelineControls();
shooter = mountShooter({
  canvas,
  camera,
  controls,
  scene,
  getCore: () => coreRef,
  getBallParams: () => CONFIG.projectile,
});
initScene()
  .then(() => loop())
  .catch((err) => {
    console.error('Failed to initialize mini-city demo:', err);
    const hint = document.querySelector('.viewport-hint');
    if (hint) hint.textContent = `Error: ${err.message}`;
  });
